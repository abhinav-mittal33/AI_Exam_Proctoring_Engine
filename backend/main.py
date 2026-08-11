from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import json
import asyncio
from typing import Dict, List

from backend.config.settings import settings
from backend.api import sessions, events, health
from backend.services.proctoring_service import proctoring_service
from backend.scoring.risk_engine import risk_engine

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    description="AI Exam Proctoring Engine with Multi-Rate Vision Pipeline, Temporal Event Debouncing, and Evidence Risk Scoring."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(sessions.router)
app.include_router(events.router)

# WebSocket Connection Manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, session_id: str, websocket: WebSocket):
        await websocket.accept()
        if session_id not in self.active_connections:
            self.active_connections[session_id] = []
        self.active_connections[session_id].append(websocket)

    def disconnect(self, session_id: str, websocket: WebSocket):
        if session_id in self.active_connections:
            if websocket in self.active_connections[session_id]:
                self.active_connections[session_id].remove(websocket)

    async def broadcast(self, session_id: str, message: dict):
        if session_id in self.active_connections:
            for connection in self.active_connections[session_id]:
                try:
                    await connection.send_json(message)
                except Exception:
                    pass

manager = ConnectionManager()

@app.websocket("/ws/proctor/{session_id}")
async def proctor_websocket_endpoint(websocket: WebSocket, session_id: str):
    await manager.connect(session_id, websocket)
    try:
        while True:
            data_text = await websocket.receive_text()
            data = json.loads(data_text)
            
            image_b64 = data.get("image", "")
            audio_energy = data.get("audio_energy", 0.0)

            if image_b64:
                result = proctoring_service.process_frame(session_id, image_b64, audio_energy)
                await manager.broadcast(session_id, result)
    except WebSocketDisconnect:
        manager.disconnect(session_id, websocket)
    except Exception as e:
        manager.disconnect(session_id, websocket)

@app.get("/")
def root():
    return {
        "app": settings.APP_NAME,
        "status": "running",
        "docs_url": "http://localhost:8001/docs"
    }
