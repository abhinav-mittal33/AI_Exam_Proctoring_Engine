import cv2
import numpy as np
import base64
import time
from typing import Dict, List, Optional

from backend.detectors.quality_gate import quality_gate
from backend.detectors.face_detector import face_detector
from backend.detectors.head_pose_detector import head_pose_detector
from backend.detectors.gaze_detector import gaze_detector
from backend.detectors.mouth_detector import mouth_detector
from backend.detectors.object_detector import object_detector
from backend.detectors.audio_detector import audio_detector
from backend.tracking.temporal_tracker import temporal_tracker
from backend.scoring.risk_engine import risk_engine

from ml.canonical_schema import CanonicalTimestep
from ml.features.temporal_buffer import temporal_buffer
from ml.features.temporal_extractor import temporal_extractor
from ml.inference.behavior_predictor import behavior_predictor
from ml.inference.fusion_layer import fusion_layer

class ProctoringService:
    def __init__(self):
        self.frame_counters: Dict[str, int] = {}
        self.session_start_times: Dict[str, float] = {}

    def start_session(self, session_id: str):
        self.session_start_times[session_id] = time.time()
        self.frame_counters[session_id] = 0
        temporal_buffer.clear(session_id)

    def process_frame(self, session_id: str, image_b64: str, audio_energy: float = 0.0) -> dict:
        try:
            if "," in image_b64:
                image_b64 = image_b64.split(",")[1]
            img_bytes = base64.b64decode(image_b64)
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        except Exception as e:
            return {"success": False, "error": f"Invalid image frame: {str(e)}"}

        if frame is None or frame.size == 0:
            return {"success": False, "error": "Could not decode image"}

        now = time.time()
        if session_id not in self.session_start_times:
            self.session_start_times[session_id] = now

        start_time = self.session_start_times[session_id]
        elapsed_since_start = now - start_time
        is_in_grace_period = elapsed_since_start < 5.0

        frame_count = self.frame_counters.get(session_id, 0) + 1
        self.frame_counters[session_id] = frame_count

        # 1. MediaPipe FaceMesh Detection
        face_res = face_detector.detect_faces(frame)
        face_count = face_res["face_count"]
        state = face_res["state"]
        landmarks_list = face_res["landmarks"]
        bboxes = face_res["bboxes"]
        distance_state = face_res.get("distance_state", "OPTIMAL")
        centering_state = face_res.get("centering_state", "CENTERED")
        face_width_pct = face_res.get("face_width_pct", 0.0)
        base_alignment_passed = face_res.get("alignment_passed", False)

        # 2. Frame Quality Gate Pre-Detector Check
        quality_res = quality_gate.evaluate_quality(frame, bboxes)
        is_usable_frame = quality_res["is_usable"]

        primary_landmarks = landmarks_list[0] if len(landmarks_list) > 0 else None

        head_res = {"direction": "CENTER", "yaw": 0.0, "pitch": 0.0, "roll": 0.0, "confidence": 0.9}
        gaze_res = {"direction": "GAZE_CENTER", "horizontal_ratio": 0.5, "is_away": False, "confidence": 0.9}
        mouth_res = {"state": "NORMAL", "mar": 0.0, "mouth_open": False, "confidence": 0.9}
        object_res = {"has_prohibited": False, "detected_objects": []}

        # 3. Behavioral Detectors
        if primary_landmarks and is_usable_frame:
            head_res = head_pose_detector.estimate_pose(frame, primary_landmarks, is_usable_frame)
            gaze_res = gaze_detector.detect_gaze(frame, primary_landmarks, is_usable_frame)
            mouth_res = mouth_detector.detect_mouth_movement(frame, primary_landmarks, is_usable_frame)

        if is_usable_frame:
            object_res = object_detector.detect_objects(frame, session_id)

        alignment_passed = (
            base_alignment_passed and
            is_usable_frame and
            abs(head_res.get("yaw", 0.0)) <= 25.0 and
            abs(head_res.get("pitch", 0.0)) <= 20.0
        )

        # Populate Canonical Timestep Feature Schema for ML Buffer
        ts = CanonicalTimestep(
            frame_quality=quality_res["quality_state"],
            brightness=quality_res.get("brightness", 120.0),
            blur=quality_res.get("blur_var", 150.0),
            frame_valid=is_usable_frame
        )
        ts.face_count = face_count
        ts.face_presence = face_count > 0
        ts.face_size = face_width_pct / 100.0
        ts.yaw = head_res.get("yaw", 0.0)
        ts.pitch = head_res.get("pitch", 0.0)
        ts.roll = head_res.get("roll", 0.0)
        ts.gaze_x = gaze_res.get("horizontal_ratio", 0.5)
        ts.gaze_y = gaze_res.get("vertical_ratio", 0.5)
        ts.gaze_direction = gaze_res.get("direction", "GAZE_CENTER")
        ts.mar = mouth_res.get("mar", 0.05)
        ts.mouth_activity = mouth_res.get("mouth_open", False)

        for obj in object_res.get("detected_objects", []):
            oname = obj.get("object_name", "").lower()
            if "phone" in oname:
                ts.phone_present = True
                ts.phone_confidence = obj.get("confidence", 0.80)
                ts.phone_bbox = obj.get("bbox", [0.0, 0.0, 0.0, 0.0])
                ts.phone_tracking_stability = obj.get("persistence_ratio", 0.80)
            elif "book" in oname:
                ts.book_present = True
                ts.book_confidence = obj.get("confidence", 0.80)
            elif "laptop" in oname:
                ts.laptop_present = True
                ts.laptop_confidence = obj.get("confidence", 0.80)

        temporal_buffer.add_timestep(session_id, ts.to_dict())

        # Extract 25 temporal features and run ML behavior prediction
        win_timesteps = temporal_buffer.get_window_timesteps(session_id)
        temp_features = temporal_extractor.extract_features(win_timesteps)
        ml_prediction = behavior_predictor.predict_window_behavior(session_id, win_timesteps)

        if not is_in_grace_period:
            if is_usable_frame and state != "FACE_DETECTION_UNCERTAIN":
                temporal_tracker.process_signal(
                    session_id, "FACE_MISSING", is_active=(face_count == 0), confidence=0.95, severity="MODERATE"
                )

            if is_usable_frame:
                temporal_tracker.process_signal(
                    session_id, "MULTIPLE_FACES", is_active=(face_count > 1), confidence=0.95, severity="HIGH", metadata={"face_count": face_count}
                )

            if primary_landmarks and is_usable_frame:
                head_dir = head_res["direction"]
                is_turned = head_dir in ["HEAD_TURNED_LEFT", "HEAD_TURNED_RIGHT", "HEAD_TURNED_UP", "HEAD_TURNED_DOWN"]
                temporal_tracker.process_signal(
                    session_id, head_dir if is_turned else "HEAD_TURNED_RIGHT", is_active=is_turned, confidence=head_res["confidence"], severity="LOW", metadata=head_res
                )

                is_gaze_away = gaze_res["is_away"]
                temporal_tracker.process_signal(
                    session_id, "GAZE_AWAY", is_active=is_gaze_away, confidence=gaze_res["confidence"], severity="LOW", metadata=gaze_res
                )

                is_mouth_open = mouth_res["mouth_open"]
                temporal_tracker.process_signal(
                    session_id, "MOUTH_MOVEMENT", is_active=is_mouth_open, confidence=mouth_res["confidence"], severity="LOW", metadata=mouth_res
                )

            if is_usable_frame:
                has_obj = object_res["has_prohibited"]
                for obj in object_res["detected_objects"]:
                    evt_type = obj.get("event_type", "PROHIBITED_OBJECT_DETECTED")
                    temporal_tracker.process_signal(
                        session_id, evt_type, is_active=has_obj, confidence=obj["confidence"], severity="HIGH", metadata=obj
                    )

            audio_res = audio_detector.analyze_audio(energy=audio_energy)
            if audio_res["is_voice"]:
                temporal_tracker.process_signal(
                    session_id, "AUDIO_ACTIVITY", is_active=True, confidence=audio_res["confidence"], severity="LOW", metadata=audio_res
                )

            flushed_completed = temporal_tracker.flush_expired_events(session_id)
            for completed_evt in flushed_completed:
                risk_engine.record_event(session_id, completed_evt)

            active_snapshot = temporal_tracker.get_active_events_snapshot(session_id)
            for active_evt in active_snapshot:
                risk_engine.record_event(session_id, active_evt)

        summary = risk_engine.get_session_summary(session_id)
        fused_events = fusion_layer.fuse_signals(summary["events"], ml_prediction)

        return {
            "success": True,
            "session_id": session_id,
            "frame_number": frame_count,
            "is_in_grace_period": is_in_grace_period,
            "grace_seconds_remaining": max(0, round(5.0 - elapsed_since_start, 1)),
            "frame_quality": quality_res["quality_state"],
            "current_status": {
                "face_count": face_count,
                "face_state": state,
                "distance_state": distance_state,
                "centering_state": centering_state,
                "face_width_pct": face_width_pct,
                "alignment_passed": alignment_passed,
                "head_direction": head_res["direction"],
                "yaw": head_res.get("yaw", 0.0),
                "pitch": head_res.get("pitch", 0.0),
                "gaze_direction": gaze_res["direction"],
                "gaze_ratio": gaze_res.get("horizontal_ratio", 0.5),
                "mouth_state": mouth_res["state"],
                "mar": mouth_res.get("mar", 0.0),
                "prohibited_objects": object_res["detected_objects"]
            },
            "ml_prediction": ml_prediction,
            "temporal_features": temp_features,
            "new_events": fused_events if not is_in_grace_period else [],
            "risk_score": summary["overall_score"] if not is_in_grace_period else 0.0,
            "risk_level": summary["risk_level"] if not is_in_grace_period else "NORMAL",
            "risk_breakdown": summary["breakdown"] if not is_in_grace_period else {
                "category_scores": {"OBJECT": 0.0, "FACE": 0.0, "GAZE": 0.0, "HEAD": 0.0, "MOUTH": 0.0, "IDENTITY": 0.0},
                "decay_adjustment": 0.0,
                "correlation_adjustment": 0.0
            }
        }

proctoring_service = ProctoringService()
