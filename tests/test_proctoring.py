import unittest
import numpy as np
import cv2

from backend.detectors.face_detector import face_detector
from backend.detectors.head_pose_detector import head_pose_detector
from backend.detectors.gaze_detector import gaze_detector
from backend.detectors.mouth_detector import mouth_detector
from backend.detectors.object_detector import object_detector
from backend.tracking.temporal_tracker import temporal_tracker
from backend.scoring.risk_engine import risk_engine

class TestProctoringEngine(unittest.TestCase):

    def setUp(self):
        # Create 640x480 black image canvas
        self.dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    def test_face_detector(self):
        res = face_detector.detect_faces(self.dummy_frame)
        self.assertIn("face_count", res)
        self.assertIn("state", res)

    def test_head_pose_detector(self):
        res = head_pose_detector.estimate_pose(self.dummy_frame)
        self.assertIn("yaw", res)
        self.assertIn("pitch", res)
        self.assertIn("direction", res)

    def test_gaze_detector(self):
        bbox = [100, 100, 200, 200]
        res = gaze_detector.detect_gaze(self.dummy_frame, bbox)
        self.assertIn("direction", res)
        self.assertIn("is_away", res)

    def test_mouth_detector(self):
        bbox = [100, 100, 200, 200]
        res = mouth_detector.detect_mouth_movement(self.dummy_frame, bbox)
        self.assertIn("mar", res)
        self.assertIn("mouth_open", res)

    def test_object_detector(self):
        res = object_detector.detect_objects(self.dummy_frame)
        self.assertIn("detected_objects", res)
        self.assertIn("has_prohibited", res)

    def test_temporal_tracker_debouncing(self):
        session_id = "test_session_001"
        # Simulate candidate event
        res1 = temporal_tracker.process_signal(session_id, "GAZE_AWAY", is_active=True, severity="LOW")
        self.assertIsNone(res1)

    def test_risk_engine_scoring(self):
        session_id = "test_session_002"
        evt = {
            "event_id": "evt_test",
            "session_id": session_id,
            "event_type": "MULTIPLE_FACES",
            "severity": "HIGH",
            "confidence": 0.95,
            "duration_ms": 3000.0,
            "started_at": "2026-08-11 20:00:00",
            "ended_at": "2026-08-11 20:00:03",
            "metadata": {}
        }
        summary = risk_engine.record_event(session_id, evt)
        self.assertGreater(summary["overall_score"], 0.0)
        self.assertIn(summary["risk_level"], ["NORMAL", "LOW_RISK", "REVIEW", "HIGH_PRIORITY_REVIEW"])

if __name__ == "__main__":
    unittest.main()
