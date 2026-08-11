import time
import math
from typing import Dict, List
from backend.config.settings import settings
from backend.scoring.event_evidence_engine import event_evidence_engine
from backend.utils.debug_logger import debug_logger

class RiskEngine:
    def __init__(self):
        self.session_events: Dict[str, List[dict]] = {}
        self.session_scores: Dict[str, float] = {}
        self.last_update_times: Dict[str, float] = {}

    def get_category_for_event(self, event_type: str) -> str:
        if "FACE" in event_type: return "FACE"
        if "GAZE" in event_type: return "GAZE"
        if "HEAD" in event_type: return "HEAD"
        if "MOUTH" in event_type: return "MOUTH"
        if "PHONE" in event_type or "OBJECT" in event_type: return "OBJECT"
        if "IDENTITY" in event_type: return "IDENTITY"
        if "CAMERA" in event_type or "FROZEN" in event_type: return "CAMERA"
        return "FACE"

    def record_event(self, session_id: str, event: dict) -> dict:
        if session_id not in self.session_events:
            self.session_events[session_id] = []
            self.session_scores[session_id] = 0.0
            self.last_update_times[session_id] = time.time()

        event_type = event.get("event_type", "")
        severity = event.get("severity", "LOW")
        duration_sec = event.get("duration_sec", event.get("duration_ms", 1000.0) / 1000.0)
        model_conf = event.get("confidence", 0.90)
        persistence = event.get("temporal_persistence", event.get("persistence", 1.0))
        metadata = event.get("raw_detector_output", event.get("metadata", {}))

        evidence_score = event_evidence_engine.calculate_evidence(
            event_type, severity, duration_sec, model_conf, persistence, metadata
        )
        event["evidence_score"] = evidence_score
        event["event_evidence"] = evidence_score

        # Deduplicate ongoing/continuous event
        existing_list = self.session_events[session_id]
        replaced = False
        for i, existing in enumerate(existing_list):
            if existing.get("event_id") == event.get("event_id"):
                existing_list[i] = event
                replaced = True
                break

        if not replaced:
            existing_list.append(event)

        self.last_update_times[session_id] = time.time()
        return self.get_session_summary(session_id)

    def calculate_session_risk(self, session_id: str) -> dict:
        if session_id not in self.session_events or len(self.session_events[session_id]) == 0:
            return {
                "overall_score": 0.0,
                "risk_level": "NORMAL",
                "breakdown": {
                    "category_scores": {"OBJECT": 0.0, "FACE": 0.0, "GAZE": 0.0, "HEAD": 0.0, "MOUTH": 0.0, "IDENTITY": 0.0},
                    "decay_adjustment": 0.0,
                    "correlation_adjustment": 0.0
                }
            }

        now = time.time()
        events = self.session_events[session_id]

        category_contributions: Dict[str, float] = {
            "OBJECT": 0.0, "FACE": 0.0, "GAZE": 0.0, "HEAD": 0.0, "MOUTH": 0.0, "IDENTITY": 0.0, "CAMERA": 0.0
        }
        total_decay_adjustment = 0.0
        total_correlation_discount = 0.0

        sorted_events = sorted(events, key=lambda x: x.get("duration_sec", 0), reverse=True)
        processed_events = []

        for evt in sorted_events:
            evt_type = evt.get("event_type", "")
            cat = self.get_category_for_event(evt_type)
            ev_score = evt.get("evidence_score", 0.0)
            ended_t = evt.get("ended_at_ts", now)

            # 20-minute Half-Life Temporal Decay
            elapsed_sec = max(0.0, now - ended_t)
            half_life_sec = settings.RISK_DECAY_HALF_LIFE * 60.0
            decay_factor = math.pow(2.0, -elapsed_sec / half_life_sec)
            decayed_evidence = ev_score * decay_factor
            total_decay_adjustment += (ev_score - decayed_evidence)

            # 50% Correlation Discount for co-occurring HEAD_TURNED + GAZE_AWAY
            effective_evidence = decayed_evidence
            if cat == "GAZE":
                for prev in processed_events:
                    if self.get_category_for_event(prev.get("event_type", "")) == "HEAD":
                        effective_evidence *= (1.0 - settings.CORRELATION_DISCOUNT)
                        total_correlation_discount += (decayed_evidence - effective_evidence)
                        break

            # Initial contribution caps per severity: LOW: 25, MODERATE: 50, HIGH: 75, CRITICAL: 100
            severity = evt.get("severity", "LOW")
            cap_by_severity = settings.SEVERITY_CAPS.get(severity, 25.0)
            capped_contribution = min(cap_by_severity, effective_evidence)

            # Accumulate within category cap
            current_cat_total = category_contributions.get(cat, 0.0)
            cat_cap = settings.CATEGORY_CAPS.get(cat, 35.0)
            category_contributions[cat] = min(cat_cap, current_cat_total + capped_contribution)
            processed_events.append(evt)

        # Bounded Aggregation Formula: Risk = 100 * (1 - prod(1 - c_i / 100))
        comp_product = 1.0
        for cat, contribution in category_contributions.items():
            comp_product *= (1.0 - contribution / 100.0)

        raw_score = 100.0 * (1.0 - comp_product)

        # Risk Smoothing: alpha = 0.25
        prev_score = self.session_scores.get(session_id, raw_score)
        smoothed_score = settings.RISK_SMOOTHING_ALPHA * raw_score + (1.0 - settings.RISK_SMOOTHING_ALPHA) * prev_score
        self.session_scores[session_id] = smoothed_score

        final_score = round(float(smoothed_score), 1)

        # Risk Level Classification
        if final_score <= 19.0:
            risk_level = "NORMAL"
        elif final_score <= 39.0:
            risk_level = "LOW_RISK"
        elif final_score <= 59.0:
            risk_level = "REVIEW"
        elif final_score <= 79.0:
            risk_level = "HIGH_PRIORITY_REVIEW"
        else:
            risk_level = "CRITICAL_REVIEW"

        # Format category contributions rounded
        formatted_cat_scores = {k: round(v, 1) for k, v in category_contributions.items()}

        # Log developer debug info
        debug_logger.log_frame_event(
            session_id, len(events), "GOOD", "SessionEvidenceEngine",
            f"Active events: {len(events)}", 1.0, 1.0, "CALCULATED",
            0.0, "SESSION", final_score, sum(formatted_cat_scores.values()),
            total_correlation_discount, total_decay_adjustment, final_score
        )

        return {
            "overall_score": final_score,
            "risk_level": risk_level,
            "breakdown": {
                "category_scores": formatted_cat_scores,
                "decay_adjustment": round(float(total_decay_adjustment), 1),
                "correlation_adjustment": round(float(total_correlation_discount), 1)
            }
        }

    def get_session_summary(self, session_id: str) -> dict:
        risk_res = self.calculate_session_risk(session_id)
        events = self.session_events.get(session_id, [])

        return {
            "session_id": session_id,
            "overall_score": risk_res["overall_score"],
            "risk_level": risk_res["risk_level"],
            "breakdown": risk_res["breakdown"],
            "event_count": len(events),
            "events": events
        }

risk_engine = RiskEngine()
