import numpy as np

class GazeDetector:
    def __init__(self):
        pass

    def detect_gaze(self, frame: np.ndarray, landmarks_3d: list = None) -> dict:
        """
        Detects gaze direction using MediaPipe Iris Landmarks (468-477).
        """
        if frame is None or landmarks_3d is None or len(landmarks_3d) < 478:
            return {"direction": "GAZE_CENTER", "horizontal_ratio": 0.5, "vertical_ratio": 0.5, "is_away": False, "confidence": 0.8}

        # Left Eye (Outer 33, Inner 133, Iris Center 468)
        left_eye_left = landmarks_3d[33][0]
        left_eye_right = landmarks_3d[133][0]
        left_iris_x = landmarks_3d[468][0]

        # Right Eye (Inner 362, Outer 263, Iris Center 473)
        right_eye_left = landmarks_3d[362][0]
        right_eye_right = landmarks_3d[263][0]
        right_iris_x = landmarks_3d[473][0]

        # Calculate horizontal ratios
        left_dist = max(1, left_eye_right - left_eye_left)
        right_dist = max(1, right_eye_right - right_eye_left)

        left_ratio = (left_iris_x - left_eye_left) / left_dist
        right_ratio = (right_iris_x - right_eye_left) / right_dist

        avg_ratio = (left_ratio + right_ratio) / 2.0

        direction = "GAZE_CENTER"
        is_away = False
        confidence = 0.85

        if avg_ratio < 0.35:
            direction = "GAZE_LEFT"
            is_away = True
            confidence = min(0.98, max(0.70, round(float((0.35 - avg_ratio) * 3.0 + 0.70), 2)))
        elif avg_ratio > 0.65:
            direction = "GAZE_RIGHT"
            is_away = True
            confidence = min(0.98, max(0.70, round(float((avg_ratio - 0.65) * 3.0 + 0.70), 2)))

        return {
            "direction": direction,
            "horizontal_ratio": round(float(avg_ratio), 3),
            "vertical_ratio": 0.50,
            "is_away": is_away,
            "confidence": confidence
        }

gaze_detector = GazeDetector()
