import time
from fastapi import APIRouter, HTTPException
from backend.schemas.session import ExamSessionStartRequest, ExamSessionEndRequest, ExamSessionResponse
from backend.scoring.risk_engine import risk_engine

router = APIRouter(prefix="/api/proctor/session", tags=["Proctoring Sessions"])

active_sessions = {}

@router.post("/start", response_model=ExamSessionResponse)
def start_exam_session(req: ExamSessionStartRequest):
    session_data = {
        "session_id": req.session_id,
        "student_id": req.student_id,
        "exam_id": req.exam_id,
        "student_name": req.student_name or "Student",
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "ended_at": None,
        "status": "ACTIVE"
    }
    active_sessions[req.session_id] = session_data
    summary = risk_engine.get_session_summary(req.session_id)
    return ExamSessionResponse(**session_data, overall_score=summary["overall_score"], risk_level=summary["risk_level"], event_count=summary["event_count"])

@router.post("/end", response_model=ExamSessionResponse)
def end_exam_session(req: ExamSessionEndRequest):
    if req.session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session_data = active_sessions[req.session_id]
    session_data["ended_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    session_data["status"] = "COMPLETED"
    
    summary = risk_engine.get_session_summary(req.session_id)
    return ExamSessionResponse(**session_data, overall_score=summary["overall_score"], risk_level=summary["risk_level"], event_count=summary["event_count"])

@router.get("/{session_id}", response_model=ExamSessionResponse)
def get_exam_session(session_id: str):
    if session_id not in active_sessions:
        # Default session placeholder if started implicitly
        active_sessions[session_id] = {
            "session_id": session_id,
            "student_id": "STUDENT_001",
            "exam_id": "EXAM_2026",
            "student_name": "Abhinav Mittal",
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "ended_at": None,
            "status": "ACTIVE"
        }
    session_data = active_sessions[session_id]
    summary = risk_engine.get_session_summary(session_id)
    return ExamSessionResponse(**session_data, overall_score=summary["overall_score"], risk_level=summary["risk_level"], event_count=summary["event_count"])
