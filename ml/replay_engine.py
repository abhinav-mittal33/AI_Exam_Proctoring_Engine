import os
import cv2
import json
import time
from typing import Dict, List

from backend.services.proctoring_service import proctoring_service
from ml.canonical_schema import CanonicalTimestep
from ml.features.temporal_buffer import temporal_buffer
from ml.features.temporal_extractor import temporal_extractor
from ml.inference.behavior_predictor import behavior_predictor
from ml.inference.fusion_layer import fusion_layer

class ReplayEngine:
    def __init__(self):
        pass

    def replay_video_file(self, video_path: str, session_id: str = "replay_test_session") -> dict:
        """
        Runs a recorded MP4 video file through the exact production proctoring + ML pipeline offline.
        """
        if not os.path.exists(video_path):
            return {"success": False, "error": f"Video file not found at {video_path}"}

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"success": False, "error": "Could not open video file"}

        proctoring_service.start_session(session_id)
        
        frames_processed = 0
        timeline_history = []
        final_result = {}

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frames_processed += 1
            
            # Encode frame to JPEG Base64
            _, buffer = cv2.imencode('.jpg', frame)
            img_b64 = buffer.tobytes()
            import base64
            b64_str = base64.b64encode(img_b64).decode('utf-8')
            
            # Process frame in Proctoring Service
            res = proctoring_service.process_frame(session_id, b64_str, audio_energy=0.01)
            
            # Populate canonical timestep for ML buffer
            ts = CanonicalTimestep(
                frame_quality=res.get("frame_quality", "GOOD"),
                frame_valid=True
            )
            st = res.get("current_status", {})
            ts.face_count = st.get("face_count", 0)
            ts.face_presence = ts.face_count > 0
            ts.yaw = st.get("yaw", 0.0)
            ts.pitch = st.get("pitch", 0.0)
            ts.gaze_direction = st.get("gaze_direction", "GAZE_CENTER")
            ts.mar = st.get("mar", 0.05)
            
            objs = st.get("prohibited_objects", [])
            for o in objs:
                if "phone" in o.get("object_name", "").lower():
                    ts.phone_present = True
                    ts.phone_confidence = o.get("confidence", 0.80)
                elif "book" in o.get("object_name", "").lower():
                    ts.book_present = True
                    ts.book_confidence = o.get("confidence", 0.80)

            temporal_buffer.add_timestep(session_id, ts.to_dict())
            
            # Predict window behavior
            win_timesteps = temporal_buffer.get_window_timesteps(session_id)
            ml_pred = behavior_predictor.predict_window_behavior(session_id, win_timesteps)
            
            # Fuse signals
            fused_events = fusion_layer.fuse_signals(res.get("new_events", []), ml_pred)
            res["fused_events"] = fused_events
            res["ml_prediction"] = ml_pred
            
            timeline_history.append({
                "frame": frames_processed,
                "risk_score": res.get("risk_score", 0.0),
                "top_behavior": ml_pred["top_behavior"],
                "active_events": len(fused_events)
            })
            
            final_result = res

        cap.release()

        return {
            "success": True,
            "session_id": session_id,
            "total_frames_processed": frames_processed,
            "final_risk_score": final_result.get("risk_score", 0.0),
            "final_risk_level": final_result.get("risk_level", "NORMAL"),
            "final_breakdown": final_result.get("risk_breakdown", {}),
            "final_ml_prediction": final_result.get("ml_prediction", {}),
            "timeline_history": timeline_history[-20:] # Last 20 samples
        }

replay_engine = ReplayEngine()
