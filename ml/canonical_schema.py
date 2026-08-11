import time
from typing import Dict, Any

class CanonicalTimestep:
    """
    Standardized feature vector schema for a single processed video frame.
    """
    def __init__(self, frame_quality: str = "GOOD", brightness: float = 120.0, blur: float = 150.0, frame_valid: bool = True):
        self.timestamp = time.time()
        
        # Frame Quality
        self.frame_quality = frame_quality
        self.brightness = float(brightness)
        self.blur = float(blur)
        self.frame_valid = bool(frame_valid)
        
        # Face Metrics
        self.face_count = 0
        self.face_confidence = 0.0
        self.face_presence = False
        self.face_size = 0.0
        self.face_center_x = 0.5
        self.face_center_y = 0.5
        self.face_tracking_stability = 1.0
        
        # Head Pose Metrics
        self.yaw = 0.0
        self.pitch = 0.0
        self.roll = 0.0
        self.head_pose_valid = True
        
        # Gaze Metrics
        self.gaze_x = 0.5
        self.gaze_y = 0.5
        self.gaze_direction = "GAZE_CENTER"
        self.gaze_valid = True
        
        # Mouth Metrics
        self.mar = 0.05
        self.mouth_activity = False
        self.mouth_valid = True
        
        # Object Metrics
        self.phone_present = False
        self.phone_confidence = 0.0
        self.phone_bbox = [0.0, 0.0, 0.0, 0.0]
        self.phone_tracking_stability = 0.0
        
        self.book_present = False
        self.book_confidence = 0.0
        self.laptop_present = False
        self.laptop_confidence = 0.0
        
        # Camera & System Metrics
        self.camera_connected = True
        self.frame_age_ms = 0.0

    def to_dict(self) -> dict:
        return self.__dict__

    def to_flat_vector(self) -> list:
        return [
            1.0 if self.frame_valid else 0.0,
            self.brightness / 255.0,
            min(1.0, self.blur / 500.0),
            float(self.face_count),
            self.face_confidence,
            1.0 if self.face_presence else 0.0,
            self.face_size,
            self.face_center_x,
            self.face_center_y,
            self.face_tracking_stability,
            self.yaw / 90.0,
            self.pitch / 90.0,
            self.roll / 90.0,
            1.0 if self.head_pose_valid else 0.0,
            self.gaze_x,
            self.gaze_y,
            1.0 if self.gaze_valid else 0.0,
            self.mar,
            1.0 if self.mouth_activity else 0.0,
            1.0 if self.phone_present else 0.0,
            self.phone_confidence,
            self.phone_bbox[0], self.phone_bbox[1], self.phone_bbox[2], self.phone_bbox[3],
            self.phone_tracking_stability,
            1.0 if self.book_present else 0.0,
            self.book_confidence,
            1.0 if self.laptop_present else 0.0,
            self.laptop_confidence,
            1.0 if self.camera_connected else 0.0
        ]
