import time
from typing import Dict, List
from backend.config.settings import settings

class RiskEngine:
    def __init__(self):
        self.session_scores: Dict[str, float] = {}
        self.last_update_times: Dict[str, float] = {}
        self.session_events: Dict[str, List[dict]] = {}

    def record_event(self, session_id: str, event: dict) -> dict:
        """
        Record a completed proctoring event and update the cumulative risk score.
        """
        if session_id not in self.session_scores:
            self.session_scores[session_id] = 0.0
            self.last_update_times[session_id] = time.time()
            self.session_events[session_id] = []

        # Apply score decay first
        self._apply_decay(session_id)

        event_type = event.get("event_type", "")
        weight = settings.RISK_WEIGHTS.get(event_type, 5)
        duration_factor = min(3.0, max(1.0, event.get("duration_ms", 1000) / 2000.0))
        added_points = weight * duration_factor

        self.session_scores[session_id] = min(100.0, self.session_scores[session_id] + added_points)
        self.session_events[session_id].append(event)
        self.last_update_times[session_id] = time.time()

        return self.get_session_summary(session_id)

    def _apply_decay(self, session_id: str):
        now = time.time()
        last_t = self.last_update_times.get(session_id, now)
        elapsed = now - last_t
        if elapsed > 0:
            decay_rate = 0.5 # decay 0.5 points per second during quiet period
            decay_amount = elapsed * decay_rate
            self.session_scores[session_id] = max(0.0, self.session_scores[session_id] - decay_amount)
            self.last_update_times[session_id] = now

    def get_session_summary(self, session_id: str) -> dict:
        if session_id not in self.session_scores:
            return {
                "session_id": session_id,
                "overall_score": 0.0,
                "risk_level": "NORMAL",
                "event_count": 0,
                "events": []
            }

        self._apply_decay(session_id)
        score = round(self.session_scores[session_id], 1)

        if score <= 20.0:
            risk_level = "NORMAL"
        elif score <= 40.0:
            risk_level = "LOW_RISK"
        elif score <= 65.0:
            risk_level = "REVIEW"
        else:
            risk_level = "HIGH_PRIORITY_REVIEW"

        events = self.session_events.get(session_id, [])

        return {
            "session_id": session_id,
            "overall_score": score,
            "risk_level": risk_level,
            "event_count": len(events),
            "events": events
        }

risk_engine = RiskEngine()
