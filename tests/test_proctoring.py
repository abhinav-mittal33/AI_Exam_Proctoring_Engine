import unittest
import numpy as np
import time

from backend.detectors.quality_gate import quality_gate
from backend.detectors.face_detector import face_detector
from backend.detectors.head_pose_detector import head_pose_detector
from backend.detectors.gaze_detector import gaze_detector
from backend.detectors.mouth_detector import mouth_detector
from backend.detectors.object_detector import object_detector
from backend.tracking.temporal_tracker import temporal_tracker
from backend.scoring.event_evidence_engine import event_evidence_engine
from backend.scoring.risk_engine import risk_engine

class TestProctoring13SanitySuite(unittest.TestCase):

    def setUp(self):
        self.dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    def test_01_normal_student_sitting_low_score(self):
        session_id = "san_01"
        summary = risk_engine.get_session_summary(session_id)
        self.assertLessEqual(summary["overall_score"], 19.0)
        self.assertEqual(summary["risk_level"], "NORMAL")

    def test_02_brief_head_turn_negligible(self):
        session_id = "san_02"
        evt = temporal_tracker.process_signal(session_id, "HEAD_TURNED_RIGHT", is_active=True, confidence=0.90)
        self.assertIsNone(evt)

    def test_03_long_head_turn_capped_at_low_severity(self):
        ev = event_evidence_engine.calculate_evidence("HEAD_TURNED_RIGHT", "LOW", duration_sec=14.7, model_confidence=0.98, metadata={"yaw": 34.2})
        self.assertLessEqual(ev, 25.0) # Strictly capped at LOW severity cap 25/100!

    def test_04_brief_gaze_away_negligible(self):
        session_id = "san_04"
        evt = temporal_tracker.process_signal(session_id, "GAZE_AWAY", is_active=True, confidence=0.85)
        self.assertIsNone(evt)

    def test_05_long_gaze_away_capped_at_low_severity(self):
        ev = event_evidence_engine.calculate_evidence("GAZE_AWAY", "LOW", duration_sec=5.0, model_confidence=0.85, metadata={"horizontal_ratio": 0.25})
        self.assertLessEqual(ev, 25.0)

    def test_06_second_face_transient_ignored(self):
        session_id = "san_06"
        evt = temporal_tracker.process_signal(session_id, "MULTIPLE_FACES", is_active=True, confidence=0.95)
        self.assertIsNone(evt)

    def test_07_second_face_5s_high_contribution(self):
        ev = event_evidence_engine.calculate_evidence("MULTIPLE_FACES", "HIGH", duration_sec=5.0, model_confidence=0.95)
        self.assertGreater(ev, 30.0)
        self.assertLessEqual(ev, 75.0) # Capped at HIGH cap 75

    def test_08_phone_single_frame_ignored(self):
        session_id = "san_08"
        res = object_detector.detect_objects(self.dummy_frame, session_id)
        self.assertFalse(res["has_prohibited"])

    def test_09_phone_5s_high_contribution(self):
        ev = event_evidence_engine.calculate_evidence("PHONE_DETECTED", "HIGH", duration_sec=5.0, model_confidence=0.85, persistence_ratio=0.90)
        self.assertGreater(ev, 40.0)
        self.assertLessEqual(ev, 75.0)

    def test_10_face_missing_1s_no_violation(self):
        ev = event_evidence_engine.calculate_evidence("FACE_MISSING", "MODERATE", duration_sec=1.0, model_confidence=0.95)
        self.assertEqual(ev, 0.0)

    def test_11_face_missing_10s_moderate_contribution(self):
        ev = event_evidence_engine.calculate_evidence("FACE_MISSING", "MODERATE", duration_sec=10.0, model_confidence=0.95)
        self.assertGreater(ev, 30.0)
        self.assertLessEqual(ev, 50.0) # Capped at MODERATE cap 50

    def test_12_poor_lighting_uncertain_not_suspicious(self):
        dark_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        q = quality_gate.evaluate_quality(dark_frame)
        self.assertEqual(q["quality_state"], "TOO_DARK")
        self.assertFalse(q["is_usable"])

    def test_13_identity_mismatch_critical(self):
        ev = event_evidence_engine.calculate_evidence("IDENTITY_MISMATCH", "CRITICAL", duration_sec=1.0, model_confidence=0.99)
        self.assertGreaterEqual(ev, 70.0)
        self.assertLessEqual(ev, 100.0)

if __name__ == "__main__":
    unittest.main()
