from typing import Dict, List

class MLDetectorFusionLayer:
    def __init__(self):
        pass

    def fuse_signals(self, detector_events: List[dict], ml_prediction: dict) -> List[dict]:
        """
        Fuses raw detector events with ML behavior model output to produce calibrated event evidence outputs.
        """
        fused_events = []
        ml_probs = ml_prediction.get("probabilities", {})
        top_behavior = ml_prediction.get("top_behavior", "NORMAL")

        for evt in detector_events:
            fused = dict(evt)
            evt_type = evt.get("event_type", "")
            base_evidence = evt.get("evidence_score", 0.0)

            # Adjust evidence based on ML behavior confidence boost
            if evt_type == "PHONE_DETECTED" and ml_probs.get("PHONE_USE", 0.0) > 0.70:
                ml_boost = ml_probs["PHONE_USE"] * 15.0
                fused["evidence_score"] = min(evt.get("severity_cap", 75.0), base_evidence + ml_boost)
                fused["ml_confirmation"] = "HIGH_CONFIRMATION"
            elif evt_type == "MULTIPLE_FACES" and ml_probs.get("MULTIPLE_PERSON", 0.0) > 0.70:
                ml_boost = ml_probs["MULTIPLE_PERSON"] * 15.0
                fused["evidence_score"] = min(evt.get("severity_cap", 75.0), base_evidence + ml_boost)
                fused["ml_confirmation"] = "HIGH_CONFIRMATION"
            else:
                fused["ml_confirmation"] = "STANDARD"

            fused["ml_behavior_score"] = ml_probs.get(top_behavior, 0.0)
            fused_events.append(fused)

        # Handle ML Suggestion when ML is confident but detector is absent
        if ml_probs.get("PHONE_USE", 0.0) > 0.85 and not any(e["event_type"] == "PHONE_DETECTED" for e in fused_events):
            fused_events.append({
                "event_id": "evt_ml_sug_phone",
                "event_type": "ML_SUGGESTION_PHONE_USE",
                "severity": "LOW",
                "severity_cap": 25.0,
                "confidence": round(ml_probs["PHONE_USE"], 2),
                "duration_sec": 5.0,
                "temporal_persistence": 0.85,
                "measured_value": f"ML behavior confidence: {ml_probs['PHONE_USE']:.2f}",
                "evidence_score": 15.0,
                "model_version": ml_prediction.get("model_version", "v1.1.0"),
                "threshold_config_version": "v1.2.0"
            })

        return fused_events

fusion_layer = MLDetectorFusionLayer()
