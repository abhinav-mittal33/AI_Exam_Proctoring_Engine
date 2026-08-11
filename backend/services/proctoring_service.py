import cv2
import numpy as np
import base64
import time
from typing import Dict, List, Optional

from backend.detectors.face_detector import face_detector
from backend.detectors.head_pose_detector import head_pose_detector
from backend.detectors.gaze_detector import gaze_detector
from backend.detectors.mouth_detector import mouth_detector
from backend.detectors.object_detector import object_detector
from backend.detectors.audio_detector import audio_detector
from backend.tracking.temporal_tracker import temporal_tracker
from backend.scoring.risk_engine import risk_engine

class ProctoringService:
    def __init__(self):
        self.frame_counters: Dict[str, int] = {}

    def process_frame(self, session_id: str, image_b64: str, audio_energy: float = 0.0) -> dict:
        """
        Multi-Rate Processing Pipeline:
        - 30 FPS: MediaPipe FaceMesh detection & face count
        - 10 FPS (every 3rd frame): Real 3D Head Pose, Iris Gaze, Mouth Aspect Ratio (MAR)
        - 5 FPS (every 6th frame): Ultralytics YOLOv8 Object Detection (Cell Phone, Book, Laptop)
        """
        try:
            if "," in image_b64:
                image_b64 = image_b64.split(",")[1]
            img_bytes = base64.b64decode(image_b64)
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        except Exception as e:
            return {"success": False, "error": f"Invalid image frame: {str(e)}"}

        if frame is None:
            return {"success": False, "error": "Could not decode image"}

        frame_count = self.frame_counters.get(session_id, 0) + 1
        self.frame_counters[session_id] = frame_count

        completed_events = []

        # 1. 30 FPS: MediaPipe FaceMesh Detection
        face_res = face_detector.detect_faces(frame)
        face_count = face_res["face_count"]
        state = face_res["state"]
        landmarks_list = face_res["landmarks"]

        # Signal: FACE_MISSING
        evt = temporal_tracker.process_signal(
            session_id, "FACE_MISSING", is_active=(face_count == 0), confidence=0.95, severity="MODERATE"
        )
        if evt: completed_events.append(evt)

        # Signal: MULTIPLE_FACES
        evt = temporal_tracker.process_signal(
            session_id, "MULTIPLE_FACES", is_active=(face_count > 1), confidence=0.95, severity="HIGH", metadata={"face_count": face_count}
        )
        if evt: completed_events.append(evt)

        primary_landmarks = landmarks_list[0] if len(landmarks_list) > 0 else None

        head_res = {"direction": "CENTER", "yaw": 0.0, "pitch": 0.0, "roll": 0.0, "confidence": 0.9}
        gaze_res = {"direction": "GAZE_CENTER", "horizontal_ratio": 0.5, "is_away": False, "confidence": 0.9}
        mouth_res = {"state": "NORMAL", "mar": 0.0, "mouth_open": False, "confidence": 0.9}
        object_res = {"has_prohibited": False, "detected_objects": []}

        # 2. 10 FPS: Head Pose, Gaze, Mouth (every 3rd frame)
        if frame_count % 3 == 0 and primary_landmarks:
            # 3D Head Pose
            head_res = head_pose_detector.estimate_pose(frame, primary_landmarks)
            head_dir = head_res["direction"]
            is_turned = head_dir != "CENTER"
            evt = temporal_tracker.process_signal(
                session_id, head_dir if is_turned else "HEAD_TURNED_RIGHT", is_active=is_turned, confidence=head_res["confidence"], severity="LOW", metadata=head_res
            )
            if evt: completed_events.append(evt)

            # Iris Gaze
            gaze_res = gaze_detector.detect_gaze(frame, primary_landmarks)
            is_gaze_away = gaze_res["is_away"]
            evt = temporal_tracker.process_signal(
                session_id, "GAZE_AWAY", is_active=is_gaze_away, confidence=gaze_res["confidence"], severity="LOW", metadata=gaze_res
            )
            if evt: completed_events.append(evt)

            # Mouth Aspect Ratio (MAR)
            mouth_res = mouth_detector.detect_mouth_movement(frame, primary_landmarks)
            is_mouth_open = mouth_res["mouth_open"]
            evt = temporal_tracker.process_signal(
                session_id, "MOUTH_MOVEMENT", is_active=is_mouth_open, confidence=mouth_res["confidence"], severity="LOW", metadata=mouth_res
            )
            if evt: completed_events.append(evt)

        # 3. 5 FPS: YOLOv8 Object Detection (every 6th frame)
        if frame_count % 6 == 0:
            object_res = object_detector.detect_objects(frame)
            for obj in object_res["detected_objects"]:
                evt_type = obj.get("event_type", "PROHIBITED_OBJECT_DETECTED")
                evt = temporal_tracker.process_signal(
                    session_id, evt_type, is_active=True, confidence=obj["confidence"], severity="HIGH", metadata=obj
                )
                if evt: completed_events.append(evt)

        # 4. Audio VAD
        audio_res = audio_detector.analyze_audio(energy=audio_energy)
        if audio_res["is_voice"]:
            evt = temporal_tracker.process_signal(
                session_id, "AUDIO_ACTIVITY", is_active=True, confidence=audio_res["confidence"], severity="LOW", metadata=audio_res
            )
            if evt: completed_events.append(evt)

        # Flush completed events
        flushed = temporal_tracker.flush_expired_events(session_id)
        completed_events.extend(flushed)

        # Record events in Risk Engine
        for completed_evt in completed_events:
            risk_engine.record_event(session_id, completed_evt)

        summary = risk_engine.get_session_summary(session_id)

        return {
            "success": True,
            "session_id": session_id,
            "frame_number": frame_count,
            "current_status": {
                "face_count": face_count,
                "face_state": state,
                "head_direction": head_res["direction"],
                "yaw": head_res.get("yaw", 0.0),
                "pitch": head_res.get("pitch", 0.0),
                "gaze_direction": gaze_res["direction"],
                "gaze_ratio": gaze_res.get("horizontal_ratio", 0.5),
                "mouth_state": mouth_res["state"],
                "mar": mouth_res.get("mar", 0.0),
                "prohibited_objects": object_res["detected_objects"]
            },
            "new_events": completed_events,
            "risk_score": summary["overall_score"],
            "risk_level": summary["risk_level"]
        }

proctoring_service = ProctoringService()
