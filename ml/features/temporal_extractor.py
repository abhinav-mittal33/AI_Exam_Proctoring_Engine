import numpy as np
from typing import List, Dict

class TemporalFeatureExtractor:
    def __init__(self):
        self.feature_names = [
            "phone_conf_mean", "phone_conf_max", "phone_conf_std", "phone_detection_ratio", "phone_track_duration",
            "book_conf_mean", "book_conf_max", "book_detection_ratio",
            "face_count_mean", "face_count_max", "multiple_face_ratio", "multiple_face_duration",
            "face_absent_ratio", "face_absent_duration",
            "head_yaw_mean", "head_yaw_max", "head_yaw_std", "head_pitch_mean", "head_pitch_max", "head_turn_duration",
            "gaze_away_ratio", "gaze_left_duration", "gaze_right_duration",
            "mouth_activity_ratio", "frame_quality_ratio"
        ]

    def extract_features(self, timesteps: List[dict]) -> dict:
        """
        Calculates 25 summary temporal features over a sliding window of timesteps.
        """
        if not timesteps or len(timesteps) == 0:
            return {name: 0.0 for name in self.feature_names}

        n = float(len(timesteps))
        start_t = timesteps[0].get("timestamp", 0.0)
        end_t = timesteps[-1].get("timestamp", 0.0)
        window_duration = max(0.1, end_t - start_t)

        # Phone statistics
        phone_confs = [float(ts.get("phone_confidence", 0.0)) for ts in timesteps]
        phone_present_cnt = sum(1 for ts in timesteps if ts.get("phone_present", False))
        phone_conf_mean = float(np.mean(phone_confs))
        phone_conf_max = float(np.max(phone_confs))
        phone_conf_std = float(np.std(phone_confs))
        phone_detection_ratio = phone_present_cnt / n
        phone_track_duration = phone_detection_ratio * window_duration

        # Book statistics
        book_confs = [float(ts.get("book_confidence", 0.0)) for ts in timesteps]
        book_present_cnt = sum(1 for ts in timesteps if ts.get("book_present", False))
        book_conf_mean = float(np.mean(book_confs))
        book_conf_max = float(np.max(book_confs))
        book_detection_ratio = book_present_cnt / n

        # Face statistics
        face_counts = [float(ts.get("face_count", 0)) for ts in timesteps]
        face_count_mean = float(np.mean(face_counts))
        face_count_max = float(np.max(face_counts))
        multi_face_cnt = sum(1 for cnt in face_counts if cnt > 1)
        multiple_face_ratio = multi_face_cnt / n
        multiple_face_duration = multiple_face_ratio * window_duration

        face_absent_cnt = sum(1 for cnt in face_counts if cnt == 0)
        face_absent_ratio = face_absent_cnt / n
        face_absent_duration = face_absent_ratio * window_duration

        # Head Pose statistics
        yaws = [abs(float(ts.get("yaw", 0.0))) for ts in timesteps]
        pitches = [abs(float(ts.get("pitch", 0.0))) for ts in timesteps]
        head_yaw_mean = float(np.mean(yaws))
        head_yaw_max = float(np.max(yaws))
        head_yaw_std = float(np.std(yaws))
        head_pitch_mean = float(np.mean(pitches))
        head_pitch_max = float(np.max(pitches))
        
        turned_cnt = sum(1 for y, p in zip(yaws, pitches) if y > 22.0 or p > 18.0)
        head_turn_duration = (turned_cnt / n) * window_duration

        # Gaze statistics
        gaze_dirs = [ts.get("gaze_direction", "GAZE_CENTER") for ts in timesteps]
        gaze_away_cnt = sum(1 for d in gaze_dirs if d != "GAZE_CENTER" and d != "GAZE_UNKNOWN")
        gaze_away_ratio = gaze_away_cnt / n
        
        gaze_left_cnt = sum(1 for d in gaze_dirs if d == "GAZE_LEFT")
        gaze_right_cnt = sum(1 for d in gaze_dirs if d == "GAZE_RIGHT")
        gaze_left_duration = (gaze_left_cnt / n) * window_duration
        gaze_right_duration = (gaze_right_cnt / n) * window_duration

        # Mouth & Frame Quality
        mouth_act_cnt = sum(1 for ts in timesteps if ts.get("mouth_activity", False))
        mouth_activity_ratio = mouth_act_cnt / n

        valid_frame_cnt = sum(1 for ts in timesteps if ts.get("frame_valid", True))
        frame_quality_ratio = valid_frame_cnt / n

        res = {
            "phone_conf_mean": round(phone_conf_mean, 3),
            "phone_conf_max": round(phone_conf_max, 3),
            "phone_conf_std": round(phone_conf_std, 3),
            "phone_detection_ratio": round(phone_detection_ratio, 3),
            "phone_track_duration": round(phone_track_duration, 2),
            "book_conf_mean": round(book_conf_mean, 3),
            "book_conf_max": round(book_conf_max, 3),
            "book_detection_ratio": round(book_detection_ratio, 3),
            "face_count_mean": round(face_count_mean, 2),
            "face_count_max": round(face_count_max, 2),
            "multiple_face_ratio": round(multiple_face_ratio, 3),
            "multiple_face_duration": round(multiple_face_duration, 2),
            "face_absent_ratio": round(face_absent_ratio, 3),
            "face_absent_duration": round(face_absent_duration, 2),
            "head_yaw_mean": round(head_yaw_mean, 2),
            "head_yaw_max": round(head_yaw_max, 2),
            "head_yaw_std": round(head_yaw_std, 2),
            "head_pitch_mean": round(head_pitch_mean, 2),
            "head_pitch_max": round(head_pitch_max, 2),
            "head_turn_duration": round(head_turn_duration, 2),
            "gaze_away_ratio": round(gaze_away_ratio, 3),
            "gaze_left_duration": round(gaze_left_duration, 2),
            "gaze_right_duration": round(gaze_right_duration, 2),
            "mouth_activity_ratio": round(mouth_activity_ratio, 3),
            "frame_quality_ratio": round(frame_quality_ratio, 3)
        }
        return res

    def extract_feature_vector(self, timesteps: List[dict]) -> np.ndarray:
        feat_dict = self.extract_features(timesteps)
        return np.array([feat_dict[k] for k in self.feature_names], dtype=np.float32)

temporal_extractor = TemporalFeatureExtractor()
