import os
import json
import time
import logging

class DebugLogger:
    def __init__(self):
        self.logger = logging.getLogger("proctoring_debug")
        self.logger.setLevel(logging.DEBUG)
        
        # Log to file if directory exists
        log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "proctoring_debug.log")
        
        handler = logging.FileHandler(log_path)
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        handler.setFormatter(formatter)
        if not self.logger.handlers:
            self.logger.addHandler(handler)

    def log_frame_event(self, session_id: str, frame_num: int, quality: str, detector: str, raw_val: str, conf: float, persistence: float, event_state: str, duration: float, severity: str, evidence: float, cat_contrib: float, corr_discount: float, decay: float, final_score: float):
        log_entry = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "session_id": session_id,
            "frame_number": frame_num,
            "frame_quality": quality,
            "detector": detector,
            "raw_measurement": raw_val,
            "raw_confidence": round(conf, 2),
            "temporal_persistence": round(persistence, 2),
            "event_state": event_state,
            "duration_sec": round(duration, 1),
            "severity": severity,
            "event_evidence": round(evidence, 1),
            "category_contribution": round(cat_contrib, 1),
            "correlation_discount": round(corr_discount, 1),
            "decay_adjustment": round(decay, 1),
            "final_session_score": round(final_score, 1)
        }
        self.logger.debug(json.dumps(log_entry))

debug_logger = DebugLogger()
