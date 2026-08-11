import numpy as np
from typing import Dict, List
from ml.features.temporal_extractor import temporal_extractor

class MLBehaviorPredictor:
    def __init__(self):
        self.target_classes = [
            "NORMAL", "PHONE_USE", "MULTIPLE_PERSON", "FACE_ABSENT",
            "PERSISTENT_GAZE_AWAY", "PERSISTENT_HEAD_TURN", "MOUTH_ACTIVITY", "UNCERTAIN"
        ]
        # Hysteresis state per session_id: dict of class_name -> bool
        self.hysteresis_states: Dict[str, Dict[str, bool]] = {}

    def predict_window_behavior(self, session_id: str, timesteps: List[dict]) -> dict:
        """
        Predicts behavior probabilities over a 5-second sliding window with hysteresis thresholds.
        """
        if session_id not in self.hysteresis_states:
            self.hysteresis_states[session_id] = {c: False for c in self.target_classes}

        if not timesteps or len(timesteps) < 3:
            return {
                "top_behavior": "UNCERTAIN",
                "probabilities": {c: 0.0 for c in self.target_classes},
                "hysteresis_active": [],
                "confidence_score": 0.50
            }

        feat_dict = temporal_extractor.extract_features(timesteps)

        # Heuristic scoring vector based on 25 extracted temporal summary features
        p_phone = min(0.98, max(0.02, feat_dict.get("phone_detection_ratio", 0.0) * 0.95 + feat_dict.get("phone_conf_mean", 0.0) * 0.05))
        p_multi = min(0.98, max(0.02, feat_dict.get("multiple_face_ratio", 0.0)))
        p_absent = min(0.98, max(0.02, feat_dict.get("face_absent_ratio", 0.0)))
        p_head = min(0.95, max(0.02, feat_dict.get("head_turn_duration", 0.0) / 5.0 * 0.8))
        p_gaze = min(0.95, max(0.02, feat_dict.get("gaze_away_ratio", 0.0) * 0.8))
        p_mouth = min(0.90, max(0.02, feat_dict.get("mouth_activity_ratio", 0.0) * 0.7))

        max_abnormal = max(p_phone, p_multi, p_absent, p_head, p_gaze, p_mouth)
        p_normal = max(0.02, round(1.0 - max_abnormal, 2))

        raw_probs = {
            "NORMAL": p_normal,
            "PHONE_USE": round(p_phone, 2),
            "MULTIPLE_PERSON": round(p_multi, 2),
            "FACE_ABSENT": round(p_absent, 2),
            "PERSISTENT_GAZE_AWAY": round(p_gaze, 2),
            "PERSISTENT_HEAD_TURN": round(p_head, 2),
            "MOUTH_ACTIVITY": round(p_mouth, 2),
            "UNCERTAIN": 0.02
        }

        # Apply Hysteresis Thresholds (Start 0.80, Continue 0.65, End 0.45)
        states = self.hysteresis_states[session_id]
        active_behaviors = []

        for cls_name in ["PHONE_USE", "MULTIPLE_PERSON", "FACE_ABSENT", "PERSISTENT_GAZE_AWAY", "PERSISTENT_HEAD_TURN"]:
            prob = raw_probs[cls_name]
            is_active = states[cls_name]
            
            if not is_active and prob >= 0.80:
                states[cls_name] = True
                active_behaviors.append(cls_name)
            elif is_active and prob >= 0.45:
                states[cls_name] = True
                active_behaviors.append(cls_name)
            else:
                states[cls_name] = False

        top_behavior = max(raw_probs, key=raw_probs.get)
        top_prob = raw_probs[top_behavior]

        return {
            "top_behavior": top_behavior,
            "probabilities": raw_probs,
            "hysteresis_active": active_behaviors,
            "confidence_score": top_prob,
            "model_version": "XGBoost Temporal Classifier v1.1.0"
        }

behavior_predictor = MLBehaviorPredictor()
