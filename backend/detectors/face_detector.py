import cv2
import numpy as np
import mediapipe as mp
from backend.config.settings import settings

class FaceDetector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=5,
            refine_landmarks=True,
            min_detection_confidence=0.35,
            min_tracking_confidence=0.35
        )

    def detect_faces(self, frame: np.ndarray) -> dict:
        if frame is None or frame.size == 0:
            return {
                "face_count": 0,
                "state": "NO_FACE",
                "bboxes": [],
                "landmarks": [],
                "confidence": 0.0,
                "presence_confidence": 0.0,
                "distance_state": "NO_FACE",
                "centering_state": "NO_FACE",
                "face_width_pct": 0.0,
                "alignment_passed": False
            }

        h, w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        if not results.multi_face_landmarks:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mean_brightness = float(np.mean(gray))
            laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

            if mean_brightness < 20.0 or laplacian_var < 8.0:
                return {
                    "face_count": 0,
                    "state": "FACE_DETECTION_UNCERTAIN",
                    "bboxes": [],
                    "landmarks": [],
                    "confidence": 0.40,
                    "presence_confidence": 0.40,
                    "distance_state": "UNCERTAIN",
                    "centering_state": "UNCERTAIN",
                    "face_width_pct": 0.0,
                    "alignment_passed": False
                }

            return {
                "face_count": 0,
                "state": "NO_FACE",
                "bboxes": [],
                "landmarks": [],
                "confidence": 0.0,
                "presence_confidence": 0.0,
                "distance_state": "NO_FACE",
                "centering_state": "NO_FACE",
                "face_width_pct": 0.0,
                "alignment_passed": False
            }

        face_count = len(results.multi_face_landmarks)
        bboxes = []
        landmarks_list = []

        for face_landmarks in results.multi_face_landmarks:
            pts = []
            xs = []
            ys = []
            for lm in face_landmarks.landmark:
                px, py, pz = int(lm.x * w), int(lm.y * h), lm.z
                pts.append((px, py, pz))
                xs.append(px)
                ys.append(py)

            x_min, x_max = max(0, min(xs)), min(w, max(xs))
            y_min, y_max = max(0, min(ys)), min(h, max(ys))
            bboxes.append([x_min, y_min, x_max - x_min, y_max - y_min])
            landmarks_list.append(pts)

        state = "ONE_FACE" if face_count == 1 else ("MULTIPLE_FACES" if face_count > 1 else "NO_FACE")
        presence_confidence = 0.95

        primary_box = bboxes[0]
        box_x, box_y, box_w, box_h = primary_box
        face_width_pct = round((box_w / float(w)) * 100.0, 1)

        # Distance rules: > 45% = TOO_CLOSE, < 15% = TOO_FAR, 15-45% = OPTIMAL
        if face_width_pct > 45.0:
            distance_state = "TOO_CLOSE"
        elif face_width_pct < 15.0:
            distance_state = "TOO_FAR"
        else:
            distance_state = "OPTIMAL"

        # Centering rules: X in [30%, 70%], Y in [15%, 80%]
        x_center_ratio = (box_x + box_w / 2.0) / float(w)
        y_center_ratio = (box_y + box_h / 2.0) / float(h)

        if 0.30 <= x_center_ratio <= 0.70 and 0.15 <= y_center_ratio <= 0.80:
            centering_state = "CENTERED"
        else:
            centering_state = "OFF_CENTER"

        alignment_passed = (
            face_count == 1 and
            distance_state == "OPTIMAL" and
            centering_state == "CENTERED"
        )

        return {
            "face_count": face_count,
            "state": state,
            "bboxes": bboxes,
            "landmarks": landmarks_list,
            "confidence": round(presence_confidence, 2),
            "presence_confidence": round(presence_confidence, 2),
            "distance_state": distance_state,
            "centering_state": centering_state,
            "face_width_pct": face_width_pct,
            "alignment_passed": alignment_passed
        }

face_detector = FaceDetector()
