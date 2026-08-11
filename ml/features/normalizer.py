import numpy as np
from typing import List, Dict

class FeatureNormalizer:
    def __init__(self):
        self.feature_names = [
            "frame_valid", "brightness_norm", "blur_norm", "face_count",
            "face_confidence", "face_presence", "face_size", "face_center_x", "face_center_y",
            "face_tracking_stability", "yaw_norm", "pitch_norm", "roll_norm", "head_pose_valid",
            "gaze_x", "gaze_y", "gaze_valid", "mar", "mouth_activity", "phone_present",
            "phone_confidence", "phone_x", "phone_y", "phone_w", "phone_h", "phone_stability",
            "book_present", "book_confidence", "laptop_present", "laptop_confidence", "camera_connected"
        ]

    def normalize_timestep(self, timestep_dict: dict) -> np.ndarray:
        """
        Converts canonical timestep dict into a normalized float32 feature array.
        """
        w, h = 640.0, 480.0
        
        frame_valid = 1.0 if timestep_dict.get("frame_valid", True) else 0.0
        brightness_norm = min(1.0, max(0.0, float(timestep_dict.get("brightness", 120.0)) / 255.0))
        blur_norm = min(1.0, max(0.0, float(timestep_dict.get("blur", 150.0)) / 500.0))
        
        face_cnt = float(timestep_dict.get("face_count", 0))
        face_conf = float(timestep_dict.get("face_confidence", 0.0))
        face_pres = 1.0 if face_cnt > 0 else 0.0
        face_sz = float(timestep_dict.get("face_size", 0.0))
        face_cx = float(timestep_dict.get("face_center_x", 0.5))
        face_cy = float(timestep_dict.get("face_center_y", 0.5))
        face_stab = float(timestep_dict.get("face_tracking_stability", 1.0))
        
        yaw_norm = float(timestep_dict.get("yaw", 0.0)) / 90.0
        pitch_norm = float(timestep_dict.get("pitch", 0.0)) / 90.0
        roll_norm = float(timestep_dict.get("roll", 0.0)) / 90.0
        head_valid = 1.0 if timestep_dict.get("head_pose_valid", True) else 0.0
        
        gaze_x = float(timestep_dict.get("gaze_x", 0.5))
        gaze_y = float(timestep_dict.get("gaze_y", 0.5))
        gaze_valid = 1.0 if timestep_dict.get("gaze_valid", True) else 0.0
        
        mar = float(timestep_dict.get("mar", 0.05))
        mouth_act = 1.0 if timestep_dict.get("mouth_activity", False) else 0.0
        
        phone_pres = 1.0 if timestep_dict.get("phone_present", False) else 0.0
        phone_conf = float(timestep_dict.get("phone_confidence", 0.0))
        bbox = timestep_dict.get("phone_bbox", [0.0, 0.0, 0.0, 0.0])
        phone_x, phone_y, phone_w, phone_h = bbox[0] / w, bbox[1] / h, bbox[2] / w, bbox[3] / h
        phone_stab = float(timestep_dict.get("phone_tracking_stability", 0.0))
        
        book_pres = 1.0 if timestep_dict.get("book_present", False) else 0.0
        book_conf = float(timestep_dict.get("book_confidence", 0.0))
        laptop_pres = 1.0 if timestep_dict.get("laptop_present", False) else 0.0
        laptop_conf = float(timestep_dict.get("laptop_confidence", 0.0))
        cam_conn = 1.0 if timestep_dict.get("camera_connected", True) else 0.0
        
        vec = [
            frame_valid, brightness_norm, blur_norm, face_cnt, face_conf, face_pres,
            face_sz, face_cx, face_cy, face_stab, yaw_norm, pitch_norm, roll_norm,
            head_valid, gaze_x, gaze_y, gaze_valid, mar, mouth_act, phone_pres,
            phone_conf, phone_x, phone_y, phone_w, phone_h, phone_stab, book_pres,
            book_conf, laptop_pres, laptop_conf, cam_conn
        ]
        return np.array(vec, dtype=np.float32)

normalizer = FeatureNormalizer()
