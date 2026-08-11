import numpy as np

class GazeDetector:
    def __init__(self):
        pass

    def detect_gaze(self, frame: np.ndarray, landmarks_3d: list = None, is_usable_frame: bool = True) -> dict:
        """
        Detects gaze direction with calibrated thresholds preventing false positives during normal screen reading.
        """
        if not is_usable_frame or frame is None or landmarks_3d is None or len(landmarks_3d) < 468:
            return {"direction": "GAZE_UNKNOWN", "horizontal_ratio": 0.5, "vertical_ratio": 0.5, "is_away": False, "confidence": 0.0}

        left_eye_left = landmarks_3d[33][0]
        left_eye_right = landmarks_3d[133][0]
        left_eye_top = landmarks_3d[159][1]
        left_eye_bottom = landmarks_3d[145][1]

        right_eye_left = landmarks_3d[362][0]
        right_eye_right = landmarks_3d[263][0]
        right_eye_top = landmarks_3d[386][1]
        right_eye_bottom = landmarks_3d[374][1]

        if len(landmarks_3d) >= 474:
            left_iris_x = landmarks_3d[468][0]
            left_iris_y = landmarks_3d[468][1]
            right_iris_x = landmarks_3d[473][0]
            right_iris_y = landmarks_3d[473][1]
        else:
            left_iris_x = (landmarks_3d[468][0] if len(landmarks_3d) > 468 else (left_eye_left + left_eye_right) / 2.0)
            left_iris_y = (landmarks_3d[159][1] + landmarks_3d[145][1]) / 2.0
            right_iris_x = (landmarks_3d[473][0] if len(landmarks_3d) > 473 else (right_eye_left + right_eye_right) / 2.0)
            right_iris_y = (landmarks_3d[386][1] + landmarks_3d[374][1]) / 2.0

        left_width = max(1.0, float(left_eye_right - left_eye_left))
        right_width = max(1.0, float(right_eye_right - right_eye_left))

        left_h_ratio = (left_iris_x - left_eye_left) / left_width
        right_h_ratio = (right_iris_x - right_eye_left) / right_width
        avg_h_ratio = (left_h_ratio + right_h_ratio) / 2.0

        left_height = max(1.0, float(left_eye_bottom - left_eye_top))
        left_v_ratio = (left_iris_y - left_eye_top) / left_height

        direction = "GAZE_CENTER"
        is_away = False
        confidence = 0.85

        # Calibrated gaze thresholds for screen corners: LEFT < 0.30, RIGHT > 0.70, DOWN > 0.72, UP < 0.28
        if avg_h_ratio < 0.30:
            direction = "GAZE_LEFT"
            is_away = True
            confidence = min(0.98, max(0.70, round(float((0.30 - avg_h_ratio) * 4.0 + 0.70), 2)))
        elif avg_h_ratio > 0.70:
            direction = "GAZE_RIGHT"
            is_away = True
            confidence = min(0.98, max(0.70, round(float((avg_h_ratio - 0.70) * 4.0 + 0.70), 2)))
        elif left_v_ratio > 0.72:
            direction = "GAZE_DOWN"
            is_away = True
            confidence = min(0.95, max(0.70, round(float((left_v_ratio - 0.72) * 3.0 + 0.70), 2)))
        elif left_v_ratio < 0.28:
            direction = "GAZE_UP"
            is_away = True
            confidence = min(0.95, max(0.70, round(float((0.28 - left_v_ratio) * 3.0 + 0.70), 2)))

        return {
            "direction": direction,
            "horizontal_ratio": round(float(avg_h_ratio), 3),
            "vertical_ratio": round(float(left_v_ratio), 3),
            "is_away": is_away,
            "confidence": confidence
        }

gaze_detector = GazeDetector()
