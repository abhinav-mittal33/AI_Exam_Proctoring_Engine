import cv2
import numpy as np
import mediapipe as mp

class FaceDetector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=5,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def detect_faces(self, frame: np.ndarray) -> dict:
        """
        Detects faces in frame using MediaPipe FaceMesh.
        Returns face_count, state (NO_FACE, ONE_FACE, MULTIPLE_FACES), bounding boxes,
        and 468 3D landmark point arrays for each detected face.
        """
        if frame is None:
            return {"face_count": 0, "state": "NO_FACE", "bboxes": [], "landmarks": [], "confidence": 0.0}

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        if not results.multi_face_landmarks:
            return {"face_count": 0, "state": "NO_FACE", "bboxes": [], "landmarks": [], "confidence": 0.0}

        h, w = frame.shape[:2]
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

        if face_count == 0:
            state = "NO_FACE"
        elif face_count == 1:
            state = "ONE_FACE"
        else:
            state = "MULTIPLE_FACES"

        return {
            "face_count": face_count,
            "state": state,
            "bboxes": bboxes,
            "landmarks": landmarks_list,
            "confidence": 0.95 if face_count > 0 else 0.0
        }

face_detector = FaceDetector()
