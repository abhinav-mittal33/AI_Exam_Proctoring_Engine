import time
from fastapi import APIRouter, HTTPException, Body
from backend.schemas.events import ProctorEventCreate, RiskSummaryResponse
from backend.services.proctoring_service import proctoring_service
from backend.scoring.risk_engine import risk_engine

router = APIRouter(prefix="/api/proctor", tags=["Proctoring Events"])

@router.post("/events/process-frame")
def process_video_frame(payload: dict = Body(...)):
    """
    Process incoming base64 video frame and optional audio energy.
    """
    session_id = payload.get("session_id", "default_session")
    image_b64 = payload.get("image", "")
    audio_energy = payload.get("audio_energy", 0.0)

    if not image_b64:
        raise HTTPException(status_code=400, detail="Base64 image frame required")

    return proctoring_service.process_frame(session_id, image_b64, audio_energy)

@router.get("/session/{session_id}/events")
def get_session_events(session_id: str):
    summary = risk_engine.get_session_summary(session_id)
    return {
        "session_id": session_id,
        "event_count": summary["event_count"],
        "events": summary["events"]
    }

@router.get("/session/{session_id}/risk")
def get_session_risk(session_id: str):
    summary = risk_engine.get_session_summary(session_id)
    return {
        "session_id": session_id,
        "overall_score": summary["overall_score"],
        "risk_level": summary["risk_level"],
        "event_count": summary["event_count"],
        "last_updated": time.strftime("%Y-%m-%d %H:%M:%S")
    }
