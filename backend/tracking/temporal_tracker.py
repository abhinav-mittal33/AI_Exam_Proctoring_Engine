import time
import uuid
from typing import Dict, List, Optional
from backend.config.settings import settings

class ActiveEventState:
    def __init__(self, event_id: str, session_id: str, event_type: str, severity: str, confidence: float, metadata: dict = None):
        self.event_id = event_id
        self.session_id = session_id
        self.event_type = event_type
        self.severity = severity
        self.start_time = time.time()
        self.last_seen = time.time()
        self.confidence_list = [confidence]
        self.metadata = metadata or {}

    def update(self, confidence: float, metadata: dict = None):
        self.last_seen = time.time()
        self.confidence_list.append(confidence)
        if metadata:
            self.metadata.update(metadata)

    def duration_seconds(self) -> float:
        return time.time() - self.start_time

    def mean_confidence(self) -> float:
        return float(sum(self.confidence_list) / len(self.confidence_list))

class TemporalTracker:
    def __init__(self):
        # Session ID -> dict of event_type -> ActiveEventState
        self.active_sessions: Dict[str, Dict[str, ActiveEventState]] = {}
        self.min_durations = {
            "FACE_MISSING": settings.FACE_MISSING_GRACE_PERIOD,
            "MULTIPLE_FACES": settings.MULTIPLE_FACE_MIN_DURATION,
            "HEAD_TURNED_LEFT": settings.HEAD_POSE_EVENT_DURATION,
            "HEAD_TURNED_RIGHT": settings.HEAD_POSE_EVENT_DURATION,
            "HEAD_TURNED_UP": settings.HEAD_POSE_EVENT_DURATION,
            "HEAD_TURNED_DOWN": settings.HEAD_POSE_EVENT_DURATION,
            "GAZE_AWAY": settings.GAZE_EVENT_DURATION,
            "MOUTH_MOVEMENT": settings.MOUTH_EVENT_DURATION,
            "PHONE_DETECTED": settings.OBJECT_MIN_PERSISTENCE,
            "PROHIBITED_OBJECT_DETECTED": settings.OBJECT_MIN_PERSISTENCE,
            "AUDIO_ACTIVITY": 2.0
        }
        self.cooldown_timeout = 1.5

    def _build_measured_value_str(self, event_type: str, metadata: dict) -> str:
        if "HEAD" in event_type:
            yaw = metadata.get("yaw", 0.0)
            pitch = metadata.get("pitch", 0.0)
            return f"yaw: {yaw:+.1f}°, pitch: {pitch:+.1f}°"
        elif "GAZE" in event_type:
            h_ratio = metadata.get("horizontal_ratio", 0.5)
            return f"horizontal iris ratio: {h_ratio:.2f}"
        elif "MOUTH" in event_type:
            mar = metadata.get("mar", 0.0)
            return f"MAR: {mar:.3f}"
        elif "PHONE" in event_type or "OBJECT" in event_type:
            obj_name = metadata.get("object_name", "object")
            return f"{obj_name} detected"
        elif "FACE" in event_type:
            cnt = metadata.get("face_count", 0)
            return f"face count: {cnt}"
        return "event active"

    def get_active_events_snapshot(self, session_id: str) -> List[dict]:
        """
        Returns list of ongoing active events that have met minimum duration thresholds.
        """
        if session_id not in self.active_sessions:
            return []

        now = time.time()
        result = []
        for event_type, active_evt in self.active_sessions[session_id].items():
            dur = active_evt.duration_seconds()
            min_req = self.min_durations.get(event_type, 1.5)
            if dur >= min_req:
                persistence = active_evt.metadata.get("persistence_ratio", 1.0)
                result.append({
                    "event_id": active_evt.event_id,
                    "session_id": session_id,
                    "event_type": active_evt.event_type,
                    "severity": active_evt.severity,
                    "severity_cap": 25.0 if active_evt.severity == "LOW" else (50.0 if active_evt.severity == "MODERATE" else (75.0 if active_evt.severity == "HIGH" else 100.0)),
                    "confidence": round(active_evt.mean_confidence(), 2),
                    "duration_sec": round(dur, 1),
                    "duration_ms": round(dur * 1000.0, 1),
                    "temporal_persistence": round(float(persistence), 2),
                    "measured_value": self._build_measured_value_str(active_evt.event_type, active_evt.metadata),
                    "raw_detector_output": active_evt.metadata,
                    "model_version": "MediaPipe 0.10.14 / YOLOv8n",
                    "threshold_config_version": "v1.2.0",
                    "started_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(active_evt.start_time)),
                    "ended_at": "ONGOING",
                    "ended_at_ts": now,
                    "is_active": True
                })
        return result

    def process_signal(self, session_id: str, event_type: str, is_active: bool, confidence: float = 1.0, severity: str = "LOW", metadata: dict = None) -> Optional[dict]:
        now = time.time()
        if session_id not in self.active_sessions:
            self.active_sessions[session_id] = {}

        session_events = self.active_sessions[session_id]
        completed_event = None

        if is_active:
            if event_type not in session_events:
                evt_id = f"evt_{uuid.uuid4().hex[:8]}"
                session_events[event_type] = ActiveEventState(evt_id, session_id, event_type, severity, confidence, metadata)
            else:
                session_events[event_type].update(confidence, metadata)
        else:
            if event_type in session_events:
                active_evt = session_events[event_type]
                idle_duration = now - active_evt.last_seen
                
                if idle_duration >= self.cooldown_timeout:
                    dur = active_evt.duration_seconds() - idle_duration
                    min_req = self.min_durations.get(event_type, 1.5)
                    
                    if dur >= min_req:
                        persistence = active_evt.metadata.get("persistence_ratio", 1.0)
                        completed_event = {
                            "event_id": active_evt.event_id,
                            "session_id": session_id,
                            "event_type": active_evt.event_type,
                            "severity": active_evt.severity,
                            "severity_cap": 25.0 if active_evt.severity == "LOW" else (50.0 if active_evt.severity == "MODERATE" else (75.0 if active_evt.severity == "HIGH" else 100.0)),
                            "confidence": round(active_evt.mean_confidence(), 2),
                            "duration_sec": round(dur, 1),
                            "duration_ms": round(dur * 1000.0, 1),
                            "temporal_persistence": round(float(persistence), 2),
                            "measured_value": self._build_measured_value_str(active_evt.event_type, active_evt.metadata),
                            "raw_detector_output": active_evt.metadata,
                            "model_version": "MediaPipe 0.10.14 / YOLOv8n",
                            "threshold_config_version": "v1.2.0",
                            "started_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(active_evt.start_time)),
                            "ended_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)),
                            "ended_at_ts": now,
                            "is_active": False
                        }
                    del session_events[event_type]

        return completed_event

    def flush_expired_events(self, session_id: str) -> List[dict]:
        if session_id not in self.active_sessions:
            return []

        now = time.time()
        completed = []
        expired_keys = []
        session_events = self.active_sessions[session_id]

        for event_type, active_evt in session_events.items():
            if now - active_evt.last_seen >= self.cooldown_timeout:
                dur = active_evt.duration_seconds() - (now - active_evt.last_seen)
                min_req = self.min_durations.get(event_type, 1.5)
                if dur >= min_req:
                    persistence = active_evt.metadata.get("persistence_ratio", 1.0)
                    completed.append({
                        "event_id": active_evt.event_id,
                        "session_id": session_id,
                        "event_type": active_evt.event_type,
                        "severity": active_evt.severity,
                        "severity_cap": 25.0 if active_evt.severity == "LOW" else (50.0 if active_evt.severity == "MODERATE" else (75.0 if active_evt.severity == "HIGH" else 100.0)),
                        "confidence": round(active_evt.mean_confidence(), 2),
                        "duration_sec": round(dur, 1),
                        "duration_ms": round(dur * 1000.0, 1),
                        "temporal_persistence": round(float(persistence), 2),
                        "measured_value": self._build_measured_value_str(active_evt.event_type, active_evt.metadata),
                        "raw_detector_output": active_evt.metadata,
                        "model_version": "MediaPipe 0.10.14 / YOLOv8n",
                        "threshold_config_version": "v1.2.0",
                        "started_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(active_evt.start_time)),
                        "ended_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)),
                        "ended_at_ts": now,
                        "is_active": False
                    })
                expired_keys.append(event_type)

        for k in expired_keys:
            del session_events[k]

        return completed

temporal_tracker = TemporalTracker()
