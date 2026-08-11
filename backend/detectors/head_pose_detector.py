import cv2
import numpy as np

class HeadPoseDetector:
    def __init__(self):
        self.model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye corner
            (225.0, 170.0, -135.0),      # Right eye corner
            (-150.0, -150.0, -125.0),    # Left mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ], dtype=np.float64)

    def estimate_pose(self, frame: np.ndarray, landmarks_3d: list = None, is_usable_frame: bool = True) -> dict:
        """
        Estimates yaw, pitch, roll from 3D landmarks. Returns HEAD_POSE_UNKNOWN if frame unusable.
        """
        if not is_usable_frame or frame is None or landmarks_3d is None or len(landmarks_3d) < 468:
            return {"yaw": 0.0, "pitch": 0.0, "roll": 0.0, "direction": "HEAD_POSE_UNKNOWN", "confidence": 0.0}

        height, width = frame.shape[:2]

        image_points = np.array([
            (landmarks_3d[1][0], landmarks_3d[1][1]),
            (landmarks_3d[152][0], landmarks_3d[152][1]),
            (landmarks_3d[33][0], landmarks_3d[33][1]),
            (landmarks_3d[263][0], landmarks_3d[263][1]),
            (landmarks_3d[61][0], landmarks_3d[61][1]),
            (landmarks_3d[291][0], landmarks_3d[291][1])
        ], dtype=np.float64)

        focal_length = width
        center = (width / 2, height / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)

        dist_coeffs = np.zeros((4, 1))

        success, rvec, tvec = cv2.solvePnP(
            self.model_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return {"yaw": 0.0, "pitch": 0.0, "roll": 0.0, "direction": "HEAD_POSE_UNKNOWN", "confidence": 0.0}

        rmat, _ = cv2.Rodrigues(rvec)
        proj_matrix = np.hstack((rmat, tvec))
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)

        pitch, yaw, roll = [float(angle[0]) for angle in euler_angles]

        direction = "CENTER"
        if yaw > 22.0:
            direction = "HEAD_TURNED_RIGHT"
        elif yaw < -22.0:
            direction = "HEAD_TURNED_LEFT"
        elif pitch > 18.0:
            direction = "HEAD_TURNED_DOWN"
        elif pitch < -18.0:
            direction = "HEAD_TURNED_UP"

        abs_max_angle = max(abs(yaw), abs(pitch), abs(roll))
        conf = min(0.98, round(float(abs_max_angle / 45.0), 2)) if direction != "CENTER" else 0.95

        return {
            "yaw": round(yaw, 2),
            "pitch": round(pitch, 2),
            "roll": round(roll, 2),
            "direction": direction,
            "confidence": max(0.60, conf)
        }

head_pose_detector = HeadPoseDetector()
