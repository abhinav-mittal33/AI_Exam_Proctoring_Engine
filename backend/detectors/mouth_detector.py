import numpy as np

class MouthDetector:
    def __init__(self):
        pass

    def detect_mouth_movement(self, frame: np.ndarray, landmarks_3d: list = None, is_usable_frame: bool = True) -> dict:
        """
        Calculates Mouth Aspect Ratio (MAR). Returns MOUTH_UNKNOWN if frame is dark/blurred/unusable.
        """
        if not is_usable_frame or frame is None or landmarks_3d is None or len(landmarks_3d) < 300:
            return {"mar": 0.0, "mouth_open": False, "state": "MOUTH_UNKNOWN", "confidence": 0.0}

        top_lip = np.array([landmarks_3d[13][0], landmarks_3d[13][1]])
        bottom_lip = np.array([landmarks_3d[14][0], landmarks_3d[14][1]])
        left_corner = np.array([landmarks_3d[61][0], landmarks_3d[61][1]])
        right_corner = np.array([landmarks_3d[291][0], landmarks_3d[291][1]])

        vert_dist = np.linalg.norm(top_lip - bottom_lip)
        horiz_dist = np.linalg.norm(left_corner - right_corner)

        mar = float(vert_dist / max(1.0, horiz_dist))
        mouth_open = mar > 0.32
        state = "MOUTH_MOVEMENT" if mouth_open else "NORMAL"
        conf = min(0.98, max(0.65, round(mar * 2.0, 2))) if mouth_open else 0.90

        return {
            "mar": round(mar, 3),
            "mouth_open": mouth_open,
            "state": state,
            "confidence": conf
        }

mouth_detector = MouthDetector()
