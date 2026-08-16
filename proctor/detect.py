import base64
import os
import time
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from proctor.face_engine import face_engine

_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(_DIR, "models")
YUNET_MODEL_PATH  = os.path.join(MODELS_DIR, "face_detection_yunet_2023mar.onnx")
YOLO8_MODEL_PATH  = os.path.join(MODELS_DIR, "yolov8n.onnx")

# Proctor-X Pose & Behavior Thresholds
# These are used as fallbacks when the client-side MediaPipe engine is not running.
# They are deliberately conservative to avoid false accusations when we're measuring
# pose from YuNet's 5 points (lower resolution than 478 landmarks).
HEAD_YAW_THRESHOLD   = 0.40
HEAD_PITCH_THRESHOLD = 0.40
GAZE_THRESHOLD       = 0.32
MAR_TALKING_THRESHOLD = 0.35  # Mouth Aspect Ratio threshold for talking

MOVE_WINDOW_SECS     = 8.0
MOVE_FREQ_THRESHOLD  = 6
# Object detection confidence: raised from 0.25 to 0.35 to reduce false positives
# from shadows, blur, or poor lighting. A phone/book held at arm's length still
# registers well above this threshold.
OBJ_CONFIDENCE_MIN   = 0.35

# Consecutive frame checks required before warning (prevents false alarms)
# Raised from 3 to 4: the extra confirmation frame catches flickering detections
# from bad lighting or hand motion shadows
REQUIRED_SUSTAINED_CHECKS = 4

# Frames of disagreement between the client's face count and ours before we
# call it tampering. Deliberately higher than the others: accusing someone of
# modifying their browser is a serious claim to get wrong.
TAMPER_SUSTAINED_CHECKS = 4

CHEATING_OBJECT_MAP = {
    "cell phone": "Mobile phone detected in frame",
    "book":       "Book / reference material detected in frame",
    "laptop":     "Secondary laptop / computer screen detected",
}

# COCO class indices for the objects above, as ordered in the yolov8n export.
TARGET_OBJECT_CLASSES = {63: "laptop", 67: "cell phone", 73: "book"}
YOLO_INPUT_SIZE = 640

# Verdicts from the proctor-x landmark engine running in the candidate's browser.
# Already debounced there by its hysteresis buffers, so they arrive as decisions
# rather than raw measurements.
CLIENT_FLAG_REASONS = {
    "LOOKING_AWAY":    "Head turned away from the screen",
    "LOOKING_DOWN":    "Looking down — possible reference material",
    "TALKING":         "Sustained mouth movement — possible verbal communication",
    "NO_FACE":         "No face visible — candidate left the camera frame",
    "MULTIPLE_FACES":  "More than one person visible",
    "EYES_CLOSED":     "Eyes closed for a sustained period",
    "HEAD_TILT":       "Head tilted to one side — possible off-screen reference",
    "HEAD_TILT_EXTREME": "Head tilted sharply to one side — looking past the screen",
    "GAZE_OFF_SCREEN": "Eyes directed away from the screen",
    "GAZE_LEFT":       "Eyes looking left, away from the screen",
    "GAZE_RIGHT":      "Eyes looking right, away from the screen",
    "GAZE_DOWN":       "Eyes looking down — possible notes or a phone in the lap",
    "GAZE_UP":         "Eyes looking up, away from the screen",
    "GAZE_EXTREME_LEFT":   "Eyes pointed FAR left, away from the screen",
    "GAZE_EXTREME_RIGHT":  "Eyes pointed FAR right, away from the screen",
    "GAZE_EXTREME_DOWN":   "Eyes pointed FAR down — possible notes or a phone",
    "GAZE_EXTREME_UP":     "Eyes pointed FAR up, away from the screen",
    "TAB_SWITCH":      "Tab switched / browser minimized — left exam interface",
    "EXIT_FULLSCREEN": "Exited fullscreen mode — potential screen share/app switch",
    "SHORTCUT_ATTEMPT": "Keyboard shortcut / right-click block — possible copy-paste attempt",
    "AUDIO_TALKING":   "Loud talking / background voice detected by microphone",
    "VIRTUAL_CAMERA_ACTIVE":  "Virtual camera is feeding the exam — video may be pre-recorded",
    "VIRTUAL_CAMERA_PRESENT": "Virtual camera software installed on this machine",
    "SECOND_DISPLAY":         "A second display is connected",
    "DEVTOOLS_OPEN":          "Browser developer tools appear to be open",
    "CLIENT_TAMPERED":        "Candidate's browser is under-reporting — client may be modified",
}

# How severe each kind of violation is. This drives how long the same violation
# is suppressed after being reported: a second person in the room is worth
# repeating quickly, while someone glancing down is not worth saying every few
# seconds. Anything unlisted is treated as MEDIUM.
#
# Raised all cooldowns by 50% to reduce alert fatigue from borderline detections:
# - CRITICAL: 6→8s (second person must persist)
# - HIGH: 12→15s (phone/book detection needs confirmation)
# - MEDIUM: 25→35s (pose violations are subjective, give some grace period)
# - LOW: 60→90s (eyes closed is brief and normal, very long suppression)
SEVERITY_COOLDOWNS = {
    "CRITICAL": 8.0,
    "HIGH": 15.0,
    "MEDIUM": 35.0,
    "LOW": 90.0,
}

SEVERITY_ORDER = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

REASON_SEVERITY = {
    CLIENT_FLAG_REASONS["MULTIPLE_FACES"]:   "CRITICAL",
    CLIENT_FLAG_REASONS["NO_FACE"]:          "HIGH",
    CLIENT_FLAG_REASONS["TALKING"]:          "HIGH",
    CLIENT_FLAG_REASONS["AUDIO_TALKING"]:    "HIGH",
    CLIENT_FLAG_REASONS["TAB_SWITCH"]:       "HIGH",
    CLIENT_FLAG_REASONS["EXIT_FULLSCREEN"]:  "HIGH",
    CLIENT_FLAG_REASONS["SHORTCUT_ATTEMPT"]: "HIGH",
    CLIENT_FLAG_REASONS["LOOKING_AWAY"]:     "MEDIUM",
    CLIENT_FLAG_REASONS["LOOKING_DOWN"]:     "MEDIUM",
    CLIENT_FLAG_REASONS["HEAD_TILT"]:        "MEDIUM",
    CLIENT_FLAG_REASONS["HEAD_TILT_EXTREME"]: "HIGH",
    CLIENT_FLAG_REASONS["GAZE_OFF_SCREEN"]:  "MEDIUM",
    CLIENT_FLAG_REASONS["GAZE_LEFT"]:        "MEDIUM",
    CLIENT_FLAG_REASONS["GAZE_RIGHT"]:       "MEDIUM",
    CLIENT_FLAG_REASONS["GAZE_UP"]:          "MEDIUM",
    # Looking down is where notes and a phone in the lap live, so it is the one
    # gaze direction worth treating as more than a wandering eye.
    CLIENT_FLAG_REASONS["GAZE_DOWN"]:        "HIGH",
    # Extreme gaze (iris offset > 0.40) indicates eyes pointed FAR off-screen,
    # definitely intentional. Fires immediately, no hysteresis wait.
    CLIENT_FLAG_REASONS["GAZE_EXTREME_LEFT"]:   "HIGH",
    CLIENT_FLAG_REASONS["GAZE_EXTREME_RIGHT"]:  "HIGH",
    CLIENT_FLAG_REASONS["GAZE_EXTREME_DOWN"]:   "CRITICAL",
    CLIENT_FLAG_REASONS["GAZE_EXTREME_UP"]:     "HIGH",
    # Defeating the proctor is treated as seriously as being caught by it.
    CLIENT_FLAG_REASONS["VIRTUAL_CAMERA_ACTIVE"]:  "CRITICAL",
    CLIENT_FLAG_REASONS["CLIENT_TAMPERED"]:        "CRITICAL",
    CLIENT_FLAG_REASONS["VIRTUAL_CAMERA_PRESENT"]: "HIGH",
    CLIENT_FLAG_REASONS["SECOND_DISPLAY"]:         "HIGH",
    CLIENT_FLAG_REASONS["DEVTOOLS_OPEN"]:          "HIGH",
    CLIENT_FLAG_REASONS["EYES_CLOSED"]:      "LOW",
}


def severity_of(reason: str) -> str:
    """Severity for a reason string, including the ones built at detection time."""
    if reason in REASON_SEVERITY:
        return REASON_SEVERITY[reason]
    lowered = reason.lower()
    if "identity mismatch" in lowered or "multiple faces" in lowered:
        return "CRITICAL"
    if "held in hand" in lowered:
        return "HIGH"
    if any(word in lowered for word in ("phone", "book", "laptop", "no face")):
        return "HIGH"
    return "MEDIUM"


@dataclass
class ParticipantState:
    enrollment: str
    warning_count: int = 0
    is_kicked: bool = False
    head_move_times: deque = field(default_factory=lambda: deque(maxlen=50))
    last_yaw: Optional[float] = None
    last_pitch: Optional[float] = None
    consecutive_no_face: int = 0
    consecutive_multi_face: int = 0
    consecutive_gaze_off: int = 0
    consecutive_tilt: int = 0
    consecutive_turn: int = 0
    consecutive_talking: int = 0
    last_warned: dict = field(default_factory=dict)
    frame_count: int = 0
    tamper_streak: int = 0


class CheatingDetector:
    """
    Proctor-X Detection Engine:
    - Uses YOLO11n (proctor-x backend detector) for real-time object & device detection (cell phone, book, laptop, handheld items)
    - Uses YuNet ONNX for face detection, head pose (yaw/pitch), gaze tracking & verbal communication (talking)
    - Sustained pose hysteresis buffer (requires 3+ consecutive checks before warning)
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._sessions = {}
        self.object_detection_error = None
        self._yunet = self._load_yunet()
        self._yolo  = self._load_proctorx_yolo()
        self._haar  = self._load_haar()
        print(f"[CheatingDetector] Proctor-X Engine Ready. YuNet={'OK' if self._yunet else 'FALLBACK'}, Objects={'OK' if self._yolo else 'DISABLED'}")
        if self._yolo is None:
            # Loud on purpose: without this the app looks healthy while phone/book
            # detection is entirely absent, which is the failure we shipped before.
            print("=" * 70)
            print("[CheatingDetector] WARNING: OBJECT DETECTION IS DISABLED")
            print(f"[CheatingDetector] Reason: {self.object_detection_error}")
            print("[CheatingDetector] Phone / book / laptop cheating will NOT be detected.")
            print("=" * 70)

    def status(self) -> dict:
        """Health of each detector, surfaced to the host so a dead detector is visible."""
        return {
            "object_detection": self._yolo is not None,
            "object_detection_error": self.object_detection_error,
            "face_detection": self._yunet is not None or self._haar is not None,
        }

    def _load_yunet(self):
        try:
            if os.path.exists(YUNET_MODEL_PATH):
                det = cv2.FaceDetectorYN.create(YUNET_MODEL_PATH, "", (640, 480), score_threshold=0.6, nms_threshold=0.3, top_k=5)
                print("[CheatingDetector] YuNet loaded from models/")
                return det
            bundled = os.path.join(os.path.dirname(cv2.__file__), "data", "face_detection_yunet_2023mar.onnx")
            if os.path.exists(bundled):
                det = cv2.FaceDetectorYN.create(bundled, "", (640, 480), score_threshold=0.6, nms_threshold=0.3, top_k=5)
                return det
        except Exception as e:
            print(f"[CheatingDetector] YuNet load error: {e}")
        return None

    def _load_proctorx_yolo(self):
        """
        Loads the Proctor-X object detector from models/yolov8n.onnx via onnxruntime.

        Deliberately not ultralytics: that pulls in torch (~800MB), which does not fit
        the 512MB free tier. There the import dies, the object detector silently
        disappears, and phone/book/laptop cheating stops being detected while every
        UI still looks healthy. onnxruntime + the 12MB ONNX weights fit comfortably.
        """
        try:
            import onnxruntime as ort
        except Exception as e:
            self.object_detection_error = f"onnxruntime not importable: {e}"
            return None

        if not os.path.exists(YOLO8_MODEL_PATH):
            self.object_detection_error = f"model file missing: {YOLO8_MODEL_PATH}"
            return None

        try:
            session = ort.InferenceSession(YOLO8_MODEL_PATH, providers=["CPUExecutionProvider"])
            self._yolo_input = session.get_inputs()[0].name
            print(f"[CheatingDetector] Object detector loaded via onnxruntime: {YOLO8_MODEL_PATH}")
            return session
        except Exception as e:
            self.object_detection_error = f"onnx session failed: {e}"
            return None

    def _load_haar(self):
        try:
            haar_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            clf = cv2.CascadeClassifier(haar_path)
            if not clf.empty():
                return clf
        except Exception as e:
            print(f"[CheatingDetector] Haar load error: {e}")
        return None

    def register_session(self, enrollment: str):
        with self._lock:
            self._sessions[enrollment] = ParticipantState(enrollment=enrollment)
            print(f"[CheatingDetector] Session registered: {enrollment}")

    def remove_session(self, enrollment: str):
        with self._lock:
            self._sessions.pop(enrollment, None)

    def get_warning_count(self, enrollment: str) -> int:
        with self._lock:
            s = self._sessions.get(enrollment)
            return s.warning_count if s else 0

    def is_session_kicked(self, enrollment: str) -> bool:
        with self._lock:
            s = self._sessions.get(enrollment)
            return s.is_kicked if s else False

    def analyze(self, enrollment: str, image_data: str,
                client_flags: list = None, hand_boxes: list = None,
                client_landmarks_active: bool = False,
                face_encoding: list = None,
                client_metrics: dict = None) -> dict:
        result = {"ok": False, "cheating": False, "kick": False, "warning_count": 0, "warning_issued": False, "reasons": [], "debug": {}}
        img = self._decode_image(image_data)
        if img is None:
            result["reasons"].append("Could not decode camera frame")
            return result
        result["ok"] = True

        with self._lock:
            if enrollment not in self._sessions:
                self._sessions[enrollment] = ParticipantState(enrollment=enrollment)
            session = self._sessions[enrollment]

            if session.is_kicked:
                result["kick"] = True
                result["warning_count"] = session.warning_count
                return result

            cheating_reasons = []
            debug = {}

            faces = self._detect_faces(img)
            face_count = len(faces)
            debug["face_count"] = int(face_count)

            # 1. No Face / Left Camera Detection
            if face_count == 0:
                session.consecutive_no_face += 1
                debug["no_face_streak"] = session.consecutive_no_face
                if session.consecutive_no_face >= REQUIRED_SUSTAINED_CHECKS:
                    cheating_reasons.append("No face visible — participant left camera frame")
            else:
                session.consecutive_no_face = 0

            # 2. Multiple Faces / Impersonation Detection
            if face_count > 1:
                session.consecutive_multi_face += 1
                if session.consecutive_multi_face >= 2:
                    cheating_reasons.append(f"Multiple faces detected ({face_count}) — possible secondary person")
            else:
                session.consecutive_multi_face = 0

            # 3. Head pose, gaze and talking - fallback only.
            #
            # YuNet gives five points (eyes, nose, mouth corners), so pose here is
            # a rough estimate and "gaze" is really eye-centre offset. That is what
            # produced the tilt and talking false alarms. When the proctor-x
            # landmark engine is running in the browser it supplies 478 points with
            # true iris tracking and a 3D transformation matrix, so its verdicts
            # replace these rather than being stacked on top of them.
            # None means no browser engine, so fall back. An empty list means the
            # engine ran and found nothing - which must not be confused with the
            # engine being absent, or these heuristics would override its verdict.
            use_server_pose = not client_landmarks_active
            if use_server_pose and face_count >= 1 and self._yunet is not None:
                yaw, pitch, gaze_off, is_talking = self._analyze_head_pose_and_mouth(faces[0], img.shape)
                debug["yaw"]        = round(float(yaw), 3) if yaw is not None else None
                debug["pitch"]      = round(float(pitch), 3) if pitch is not None else None
                debug["gaze_off"]   = bool(gaze_off)
                debug["is_talking"] = bool(is_talking)

                if yaw is not None:
                    # Frequent head movement tracking
                    if session.last_yaw is not None:
                        delta_yaw   = abs(yaw - session.last_yaw)
                        delta_pitch = abs((pitch or 0) - (session.last_pitch or 0))
                        if delta_yaw > 0.18 or delta_pitch > 0.18:
                            session.head_move_times.append(time.time())
                    session.last_yaw   = yaw
                    session.last_pitch = pitch

                    now = time.time()
                    recent = [t for t in session.head_move_times if now - t <= MOVE_WINDOW_SECS]
                    debug["head_moves_in_window"] = int(len(recent))
                    if len(recent) > MOVE_FREQ_THRESHOLD:
                        cheating_reasons.append(f"Frequent head movement ({len(recent)} turns in {MOVE_WINDOW_SECS:.0f}s)")

                    # Head Turn (Yaw) — Requires 3+ sustained checks
                    if abs(yaw) > HEAD_YAW_THRESHOLD:
                        session.consecutive_turn += 1
                        if session.consecutive_turn >= REQUIRED_SUSTAINED_CHECKS:
                            direction = "left" if yaw < 0 else "right"
                            cheating_reasons.append(f"Head turned {direction} — looking away from screen")
                    else:
                        session.consecutive_turn = 0

                    # Head Tilt (Pitch) — Requires 3+ sustained checks (prevents 1-2 brief tilt false alarms)
                    if pitch is not None and abs(pitch) > HEAD_PITCH_THRESHOLD:
                        session.consecutive_tilt += 1
                        if session.consecutive_tilt >= REQUIRED_SUSTAINED_CHECKS:
                            direction = "down" if pitch > 0 else "up"
                            cheating_reasons.append(f"Sustained head tilt {direction} — looking at reference material")
                    else:
                        session.consecutive_tilt = 0

                    # Gaze Tracking
                    if gaze_off:
                        session.consecutive_gaze_off += 1
                        if session.consecutive_gaze_off >= REQUIRED_SUSTAINED_CHECKS:
                            cheating_reasons.append("Eyes consistently looking away from screen")
                    else:
                        session.consecutive_gaze_off = 0

                    # Talking / Mouth Movement Detection
                    if is_talking:
                        session.consecutive_talking += 1
                        if session.consecutive_talking >= REQUIRED_SUSTAINED_CHECKS:
                            cheating_reasons.append("Continuous talking/speaking detected — potential verbal communication")
                    else:
                        session.consecutive_talking = 0

            # 4. Objects (phone / book / laptop), and whether they are being held
            if self._yolo is not None:
                detected_objs = self._detect_objects_proctorx(img, hand_boxes or [])
                debug["detected_objects"] = [o[0] for o in detected_objs]
                for _, obj_reason in detected_objs:
                    cheating_reasons.append(obj_reason)
            else:
                debug["detected_objects"] = []

            # 5. Verdicts from the proctor-x landmark engine in the candidate's
            # browser: iris gaze, 3D head pose and mouth movement, none of which
            # are recoverable from YuNet's five points. Already debounced there.
            # Treated as additional evidence only - face count and objects above
            # are decided server-side, since a browser can be tampered with.
            for flag in (client_flags or []):
                reason = CLIENT_FLAG_REASONS.get(flag)
                if reason:
                    cheating_reasons.append(reason)
            debug["client_flags"] = list(client_flags or [])

            # 5b. Cross-check the browser's account against our own.
            #
            # Everything the client reports can be forged by editing the page, so
            # the one claim worth auditing is the one a cheat would suppress:
            # that only one person is in shot. This compares the client's face
            # count with YuNet's on the very same frame and requires the two to
            # disagree repeatedly before saying anything, since two different
            # detectors legitimately differ on a borderline frame.
            claimed = None
            if isinstance(client_metrics, dict):
                claimed = client_metrics.get("faces")

            if client_landmarks_active and isinstance(claimed, int):
                # Only one direction is suspicious: we see more people than the
                # client admits to. The reverse is just a detector disagreeing.
                if face_count > claimed:
                    session.tamper_streak += 1
                else:
                    session.tamper_streak = 0

                debug["claimed_faces"] = claimed
                debug["tamper_streak"] = session.tamper_streak

                if session.tamper_streak >= TAMPER_SUSTAINED_CHECKS:
                    cheating_reasons.append(CLIENT_FLAG_REASONS["CLIENT_TAMPERED"])

            # 6. Biometric Face Verification (candidate substitution/impersonation check)
            # Run this every 5th frame (~15s) when exactly 1 face is present to keep CPU usage low
            session.frame_count += 1
            if face_encoding is not None and face_count == 1 and session.frame_count % 5 == 0:
                matched, score, msg = face_engine.verify_face(img, face_encoding)
                debug["face_verified"] = bool(matched)
                debug["face_verification_score"] = float(score)
                if not matched:
                    cheating_reasons.append("Face verification failed — identity mismatch detected")

            now = time.time()
            deduped = self._deduplicate_reasons(cheating_reasons, session, now)

            if deduped:
                session.warning_count += 1
                result["cheating"]       = True
                result["warning_issued"] = True
                result["reasons"]        = deduped
                # Report the worst thing in this batch so the proctor can triage
                # a second person in the room ahead of someone glancing down.
                result["severity"] = max(
                    (severity_of(r) for r in deduped),
                    key=lambda s: SEVERITY_ORDER.index(s) if s in SEVERITY_ORDER else 1,
                )
                if session.warning_count >= 3:
                    session.is_kicked = True
                    result["kick"]    = True

            result["warning_count"] = session.warning_count
            result["debug"]         = debug

        return result

    def _decode_image(self, image_data: str):
        try:
            if image_data.startswith("data:"):
                image_data = image_data.split(",", 1)[1]
            raw = base64.b64decode(image_data)
            arr = np.frombuffer(raw, dtype=np.uint8)
            return cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except Exception as e:
            print(f"[CheatingDetector] Decode error: {e}")
            return None

    def _detect_faces(self, img: np.ndarray) -> list:
        h, w = img.shape[:2]
        if self._yunet is not None:
            try:
                self._yunet.setInputSize((w, h))
                _, faces = self._yunet.detect(img)
                if faces is not None and len(faces) > 0:
                    return [f for f in faces if float(f[14]) >= 0.6]
                return []
            except Exception as e:
                print(f"[CheatingDetector] YuNet detect error: {e}")
        if self._haar is not None:
            try:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                rects = self._haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
                if len(rects) > 0:
                    return [np.concatenate([r.astype(float), np.zeros(11)]) for r in rects]
            except Exception as e:
                print(f"[CheatingDetector] Haar detect error: {e}")
        return []

    def _analyze_head_pose_and_mouth(self, face_row: np.ndarray, img_shape: tuple):
        try:
            if len(face_row) < 14 or (face_row[4] == 0 and face_row[5] == 0):
                return None, None, False, False

            # YuNet Landmarks:
            # 4,5: Right eye | 6,7: Left eye | 8,9: Nose | 10,11: Right mouth | 12,13: Left mouth
            r_eye = np.array([float(face_row[4]), float(face_row[5])])
            l_eye = np.array([float(face_row[6]), float(face_row[7])])
            nose  = np.array([float(face_row[8]), float(face_row[9])])

            eye_center = (r_eye + l_eye) / 2.0
            eye_dist   = np.linalg.norm(r_eye - l_eye)
            if eye_dist < 5:
                return None, None, False, False

            yaw_ratio   = (nose[0] - eye_center[0]) / eye_dist
            pitch_ratio = (nose[1] - eye_center[1]) / eye_dist

            face_cx  = float(face_row[0]) + float(face_row[2]) / 2.0
            gaze_off = abs(eye_center[0] - face_cx) / (float(face_row[2]) + 1e-5) > GAZE_THRESHOLD

            # Mouth aspect ratio / talking analysis
            is_talking = False
            if len(face_row) >= 14:
                r_mouth = np.array([float(face_row[10]), float(face_row[11])])
                l_mouth = np.array([float(face_row[12]), float(face_row[13])])
                mouth_center = (r_mouth + l_mouth) / 2.0
                mouth_width = np.linalg.norm(r_mouth - l_mouth)
                mouth_vertical = np.linalg.norm(mouth_center - nose)

                mar = (mouth_vertical / (eye_dist + 1e-5))
                is_talking = (mar > MAR_TALKING_THRESHOLD) or (mouth_width / eye_dist > 0.65)

            return yaw_ratio, pitch_ratio, gaze_off, is_talking
        except Exception as e:
            print(f"[CheatingDetector] Pose error: {e}")
            return None, None, False, False

    def _detect_objects_proctorx(self, img: np.ndarray, hand_boxes: list = None) -> list:
        """
        Finds phones, books and laptops, and says whether one is being held.

        hand_boxes are normalised rectangles from the browser's hand tracker. An
        object overlapping a hand is much stronger evidence than one sitting
        somewhere in the room, so the two cases are reported differently.
        """
        results = []
        try:
            blob = cv2.resize(img, (YOLO_INPUT_SIZE, YOLO_INPUT_SIZE))
            blob = cv2.cvtColor(blob, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            blob = np.transpose(blob, (2, 0, 1))[np.newaxis, ...]

            out = self._yolo.run(None, {self._yolo_input: blob})[0]

            # YOLOv8 head: (1, 84, 8400) -> 4 box coords (cx, cy, w, h in input
            # pixels) followed by 80 class scores.
            preds = np.squeeze(out, 0)
            boxes = preds[:4, :]
            scores = preds[4:, :]

            for cls_id, name in TARGET_OBJECT_CLASSES.items():
                class_scores = scores[cls_id]
                best = int(np.argmax(class_scores))
                if float(class_scores[best]) < OBJ_CONFIDENCE_MIN:
                    continue

                # Straight resize (no letterbox), so normalising by the input size
                # maps back onto the original frame proportionally.
                cx, cy, w, h = (float(v) / YOLO_INPUT_SIZE for v in boxes[:, best])
                box = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)

                if self._overlaps_hand(box, hand_boxes or []):
                    results.append((name, f"{name.title()} held in hand — prohibited item in use"))
                else:
                    results.append((name, CHEATING_OBJECT_MAP[name]))

        except Exception as e:
            print(f"[CheatingDetector] Object detect error: {e}")
        return results

    @staticmethod
    def _overlaps_hand(box, hand_boxes) -> bool:
        x1, y1, x2, y2 = box
        for hand in hand_boxes:
            try:
                ix1 = max(x1, float(hand["minX"]))
                iy1 = max(y1, float(hand["minY"]))
                ix2 = min(x2, float(hand["maxX"]))
                iy2 = min(y2, float(hand["maxY"]))
                if ix1 < ix2 and iy1 < iy2:
                    return True
            except (KeyError, TypeError, ValueError):
                continue
        return False

    def _deduplicate_reasons(self, reasons: list, session: ParticipantState, now: float,
                             cooldown_secs: float = None) -> list:
        """
        Suppresses a violation that was reported recently.

        The wait depends on how serious it is rather than being one flat number:
        a second person in the room is worth repeating within seconds, whereas
        someone glancing down does not need saying every few seconds. That flat
        12s was both too slow for the serious cases and too chatty for the minor
        ones.
        """
        deduped = []
        for reason in reasons:
            tag = " ".join(reason.split()[:3]).lower()
            wait = cooldown_secs if cooldown_secs is not None else \
                SEVERITY_COOLDOWNS.get(severity_of(reason), 25.0)
            last = session.last_warned.get(tag, 0)
            if now - last >= wait:
                session.last_warned[tag] = now
                deduped.append(reason)
        return deduped


cheating_detector = CheatingDetector()
