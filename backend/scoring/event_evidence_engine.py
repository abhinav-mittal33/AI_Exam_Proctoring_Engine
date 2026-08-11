import math
from typing import Dict, Any
from backend.config.settings import settings

class EventEvidenceEngine:
    def __init__(self):
        self.severity_caps = {
            "LOW": 25.0,
            "MODERATE": 50.0,
            "HIGH": 75.0,
            "CRITICAL": 100.0
        }
        self.base_severities = settings.BASE_SEVERITIES

    def calculate_evidence(self, event_type: str, severity: str, duration_sec: float, model_confidence: float, persistence_ratio: float = 1.0, metadata: dict = None) -> float:
        """
        Calculates bounded event evidence score E in [0, severity_cap].
        Formula: E_raw = S_base * (1 - exp(-duration / tau)) * M_magnitude * M_persistence * C_model
        Capped strictly by self.severity_caps[severity].
        """
        metadata = metadata or {}
        base_score = self.base_severities.get(severity, 10.0)

        # 1. Grace Period Filter for FACE_MISSING & MULTIPLE_FACES
        if event_type == "FACE_MISSING" and duration_sec < settings.FACE_MISSING_GRACE_PERIOD:
            return 0.0
        if event_type == "MULTIPLE_FACES" and duration_sec < settings.MULTIPLE_FACE_MIN_DURATION:
            return 0.0

        # 2. Saturating Duration Function: 1 - exp(-duration / tau)
        tau = 5.0 # saturation time constant
        duration_mult = 1.0 + (1.0 - math.exp(-max(0.0, duration_sec) / tau)) * 1.5

        # 3. Monotonic Magnitude Multiplier
        magnitude_mult = 1.0
        if "HEAD_TURNED" in event_type:
            yaw = abs(metadata.get("yaw", 0.0))
            pitch = abs(metadata.get("pitch", 0.0))
            max_angle = max(yaw, pitch)
            threshold = settings.HEAD_YAW_EVENT
            if max_angle > threshold:
                magnitude_mult = 1.0 + (max_angle - threshold) / threshold
        elif event_type == "MOUTH_MOVEMENT":
            mar = metadata.get("mar", 0.0)
            if mar > settings.MAR_THRESHOLD:
                magnitude_mult = 1.0 + (mar - settings.MAR_THRESHOLD) / settings.MAR_THRESHOLD
        elif event_type == "GAZE_AWAY":
            h_ratio = metadata.get("horizontal_ratio", 0.5)
            dev = abs(h_ratio - 0.5)
            magnitude_mult = 1.0 + max(0.0, dev - 0.12) * 2.0

        # 4. Persistence Ratio (e.g. 18/20 frames = 0.90)
        persistence_mult = min(1.0, max(0.4, persistence_ratio))

        # 5. Model Confidence Scaling
        conf_scale = min(1.0, max(0.3, model_confidence))

        # Raw calculated evidence score
        raw_evidence = base_score * duration_mult * magnitude_mult * persistence_mult * conf_scale

        # Apply Strict Policy Severity Cap
        cap = self.severity_caps.get(severity, 25.0)
        final_evidence = min(cap, max(0.0, raw_evidence))

        return round(float(final_evidence), 1)

event_evidence_engine = EventEvidenceEngine()
