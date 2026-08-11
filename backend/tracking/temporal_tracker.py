import time
import uuid
from typing import Dict, List, Optional

class ActiveEventState:
    def __init__(self, event_type: str, severity: str, confidence: float, metadata: dict = None):
        self.event_id = f"evt_{uuid.uuid4().hex[:8]}"
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
        # Maps event_type -> ActiveEventState
        self.active_events: Dict[str, ActiveEventState] = {}
        # Min duration before logging event (seconds)
        self.min_durations = {
            "FACE_MISSING": 3.0,
            "MULTIPLE_FACES": 2.0,
            "HEAD_TURNED_LEFT": 2.0,
            "HEAD_TURNED_RIGHT": 2.0,
            "HEAD_TURNED_UP": 2.0,
            "HEAD_TURNED_DOWN": 2.0,
            "GAZE_AWAY": 2.0,
            "MOUTH_MOVEMENT": 1.5,
            "PHONE_DETECTED": 1.0,
            "PROHIBITED_OBJECT_DETECTED": 1.0,
            "AUDIO_ACTIVITY": 2.0
        }
        # Cooldown timeout before declaring event END (seconds)
        self.cooldown_timeout = 1.5

    def process_signal(self, session_id: str, event_type: str, is_active: bool, confidence: float = 1.0, severity: str = "LOW", metadata: dict = None) -> Optional[dict]:
        """
        Process a detector signal frame by frame.
        Returns a completed ProctorEvent dict if an event lifecycle finishes or crosses threshold.
        """
        now = time.time()
        completed_event = None

        if is_active:
            if event_type not in self.active_events:
                # Start tracking candidate event
                self.active_events[event_type] = ActiveEventState(event_type, severity, confidence, metadata)
            else:
                # Update existing active event
                self.active_events[event_type].update(confidence, metadata)
        else:
            # Check if an active event has ended due to cooldown
            if event_type in self.active_events:
                active_evt = self.active_events[event_type]
                idle_duration = now - active_evt.last_seen
                
                if idle_duration >= self.cooldown_timeout:
                    dur = active_evt.duration_seconds() - idle_duration
                    min_req = self.min_durations.get(event_type, 1.5)
                    
                    if dur >= min_req:
                        completed_event = {
                            "event_id": active_evt.event_id,
                            "session_id": session_id,
                            "event_type": active_evt.event_type,
                            "severity": active_evt.severity,
                            "confidence": round(active_evt.mean_confidence(), 3),
                            "duration_ms": round(dur * 1000.0, 1),
                            "started_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(active_evt.start_time)),
                            "ended_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)),
                            "metadata": active_evt.metadata
                        }
                    del self.active_events[event_type]

        return completed_event

    def flush_expired_events(self, session_id: str) -> List[dict]:
        """
        Flushes and returns any active events that have exceeded grace thresholds.
        """
        now = time.time()
        completed = []
        expired_keys = []

        for event_type, active_evt in self.active_events.items():
            if now - active_evt.last_seen >= self.cooldown_timeout:
                dur = active_evt.duration_seconds() - (now - active_evt.last_seen)
                min_req = self.min_durations.get(event_type, 1.5)
                if dur >= min_req:
                    completed.append({
                        "event_id": active_evt.event_id,
                        "session_id": session_id,
                        "event_type": active_evt.event_type,
                        "severity": active_evt.severity,
                        "confidence": round(active_evt.mean_confidence(), 3),
                        "duration_ms": round(dur * 1000.0, 1),
                        "started_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(active_evt.start_time)),
                        "ended_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)),
                        "metadata": active_evt.metadata
                    })
                expired_keys.append(event_type)

        for k in expired_keys:
            del self.active_events[k]

        return completed

temporal_tracker = TemporalTracker()
