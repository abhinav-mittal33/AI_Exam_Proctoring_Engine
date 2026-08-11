from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime

class ProctorEventCreate(BaseModel):
    session_id: str
    event_type: str = Field(..., description="Observable event type (e.g. FACE_MISSING, GAZE_AWAY, PHONE_DETECTED)")
    severity: str = Field("LOW", description="LOW, MODERATE, HIGH, CRITICAL")
    confidence: float = Field(1.0, ge=0.0, le=1.0)
    duration_ms: float = Field(0.0, ge=0.0)
    started_at: str
    ended_at: str
    metadata: Dict[str, Any] = {}

class ProctorEventResponse(BaseModel):
    event_id: str
    session_id: str
    event_type: str
    severity: str
    confidence: float
    duration_ms: float
    started_at: str
    ended_at: str
    metadata: Dict[str, Any]

class RiskSummaryResponse(BaseModel):
    session_id: str
    overall_score: float
    risk_level: str
    event_counts: Dict[str, int]
    last_updated: str
