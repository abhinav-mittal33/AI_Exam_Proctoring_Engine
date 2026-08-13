import base64
import os
import time
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(_DIR, "models")
YUNET_MODEL_PATH  = os.path.join(MODELS_DIR, "face_detection_yunet_2023mar.onnx")
YOLO8_MODEL_PATH  = os.path.join(MODELS_DIR, "yolov8n.onnx")

# Proctor-X Pose & Behavior Thresholds
HEAD_YAW_THRESHOLD   = 0.40
HEAD_PITCH_THRESHOLD = 0.40
GAZE_THRESHOLD       = 0.32
MAR_TALKING_THRESHOLD = 0.35  # Mouth Aspect Ratio threshold for talking

MOVE_WINDOW_SECS     = 8.0
MOVE_FREQ_THRESHOLD  = 6
OBJ_CONFIDENCE_MIN   = 0.32  # Proctor-X conf threshold

# Consecutive frame checks required before warning (prevents false alarms)
REQUIRED_SUSTAINED_CHECKS = 3

CHEATING_OBJECT_MAP = {
    "cell phone": "Mobile phone detected in frame",
    "book":       "Book / reference material detected in frame",
    "laptop":     "Secondary laptop / computer screen detected",
}

# COCO class indices for the objects above, as ordered in the yolov8n export.
TARGET_OBJECT_CLASSES = {63: "laptop", 67: "cell phone", 73: "book"}
YOLO_INPUT_SIZE = 640


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

    def analyze(self, enrollment: str, image_data: str) -> dict:
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

            # 3. Head Pose, Gaze & Talking Analysis
            if face_count >= 1 and self._yunet is not None:
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

            # 4. Proctor-X YOLO11 Detection Engine (Cell Phone, Book, Laptop, Handheld objects)
            if self._yolo is not None:
                detected_objs = self._detect_objects_proctorx(img)
                debug["detected_objects"] = [o[0] for o in detected_objs]
                for _, obj_reason in detected_objs:
                    cheating_reasons.append(obj_reason)
            else:
                debug["detected_objects"] = []

            now = time.time()
            deduped = self._deduplicate_reasons(cheating_reasons, session, now)

            if deduped:
                session.warning_count += 1
                result["cheating"]       = True
                result["warning_issued"] = True
                result["reasons"]        = deduped
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

    def _detect_objects_proctorx(self, img: np.ndarray) -> list:
        """
        Proctor-X object detection logic:
        Detects cell phone, book, laptop, and prohibited handheld items.
        """
        results = []
        try:
            blob = cv2.resize(img, (YOLO_INPUT_SIZE, YOLO_INPUT_SIZE))
            blob = cv2.cvtColor(blob, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            blob = np.transpose(blob, (2, 0, 1))[np.newaxis, ...]

            out = self._yolo.run(None, {self._yolo_input: blob})[0]

            # YOLOv8 head: (1, 84, 8400) -> 4 box coords followed by 80 class scores.
            # Only presence matters here, not boxes, so the per-class max over all
            # anchors is enough and no NMS is needed.
            scores = np.squeeze(out, 0)[4:, :]

            for cls_id, name in TARGET_OBJECT_CLASSES.items():
                if float(scores[cls_id].max()) >= OBJ_CONFIDENCE_MIN:
                    results.append((name, CHEATING_OBJECT_MAP[name]))

        except Exception as e:
            print(f"[CheatingDetector] Object detect error: {e}")
        return results

    def _deduplicate_reasons(self, reasons: list, session: ParticipantState, now: float, cooldown_secs: float = 12.0) -> list:
        deduped = []
        for reason in reasons:
            tag  = " ".join(reason.split()[:3]).lower()
            last = session.last_warned.get(tag, 0)
            if now - last >= cooldown_secs:
                session.last_warned[tag] = now
                deduped.append(reason)
        return deduped


cheating_detector = CheatingDetector()
