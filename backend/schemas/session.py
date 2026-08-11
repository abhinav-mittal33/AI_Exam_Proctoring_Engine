from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

class ExamSessionStartRequest(BaseModel):
    session_id: str
    student_id: str
    exam_id: str
    student_name: Optional[str] = "Student"

class ExamSessionEndRequest(BaseModel):
    session_id: str

class ExamSessionResponse(BaseModel):
    session_id: str
    student_id: str
    exam_id: str
    student_name: str
    started_at: str
    ended_at: Optional[str] = None
    status: str = "ACTIVE"
    overall_score: float = 0.0
    risk_level: str = "NORMAL"
    event_count: int = 0
