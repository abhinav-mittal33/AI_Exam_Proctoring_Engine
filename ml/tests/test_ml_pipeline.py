import unittest
import numpy as np
from ml.canonical_schema import CanonicalTimestep
from ml.features.normalizer import normalizer
from ml.features.temporal_buffer import TemporalBuffer
from ml.features.temporal_extractor import temporal_extractor
from ml.data.dataset_pipeline import dataset_pipeline
from ml.models.model_registry import model_registry
from ml.models.model_evaluator import model_evaluator
from ml.inference.behavior_predictor import behavior_predictor
from ml.inference.fusion_layer import fusion_layer

class TestMLBehaviorPipeline(unittest.TestCase):

    def test_canonical_schema_and_normalizer(self):
        ts = CanonicalTimestep(frame_quality="GOOD", brightness=128.0, blur=200.0, frame_valid=True)
        ts.face_count = 1
        ts.face_confidence = 0.95
        ts.yaw = 15.0
        ts.pitch = 5.0
        ts.phone_present = True
        ts.phone_confidence = 0.85
        
        vec = normalizer.normalize_timestep(ts.to_dict())
        self.assertEqual(len(vec), 31)
        self.assertGreater(vec[1], 0.4) # brightness
        self.assertGreater(vec[20], 0.8) # phone confidence

    def test_temporal_buffer_and_extractor(self):
        buf = TemporalBuffer(window_duration_sec=5.0)
        sess_id = "test_sess_ml"

        for i in range(10):
            ts = CanonicalTimestep()
            ts.face_count = 1
            ts.phone_present = (i >= 5)
            ts.phone_confidence = 0.80 if i >= 5 else 0.0
            buf.add_timestep(sess_id, ts.to_dict())

        window = buf.get_window_timesteps(sess_id)
        self.assertEqual(len(window), 10)

        feats = temporal_extractor.extract_features(window)
        self.assertIn("phone_detection_ratio", feats)
        self.assertEqual(feats["phone_detection_ratio"], 0.5)

    def test_dataset_pipeline_validation(self):
        report = dataset_pipeline.validate_dataset()
        self.assertIn("status", report)
        self.assertIn("is_sufficient", report)

    def test_model_registry(self):
        info = model_registry.get_active_model_info()
        self.assertIn("model_name", info)
        self.assertIn("validation_metrics", info)

    def test_ml_inference_and_fusion(self):
        timesteps = []
        for _ in range(8):
            ts = CanonicalTimestep()
            ts.phone_present = True
            ts.phone_confidence = 0.90
            timesteps.append(ts.to_dict())

        pred = behavior_predictor.predict_window_behavior("sess_test_inf", timesteps)
        self.assertEqual(pred["top_behavior"], "PHONE_USE")

        detector_events = [{
            "event_id": "evt_test",
            "event_type": "PHONE_DETECTED",
            "evidence_score": 40.0,
            "severity_cap": 75.0
        }]
        fused = fusion_layer.fuse_signals(detector_events, pred)
        self.assertEqual(len(fused), 1)
        self.assertGreater(fused[0]["evidence_score"], 40.0)

if __name__ == "__main__":
    unittest.main()
