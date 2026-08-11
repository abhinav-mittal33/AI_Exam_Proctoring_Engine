from fastapi import APIRouter
from backend.config.settings import settings

router = APIRouter(tags=["Health"])

@router.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": settings.APP_NAME,
        "version": settings.VERSION,
        "models_loaded": True,
        "detectors": {
            "face_presence": True,
            "head_pose": True,
            "gaze": True,
            "mouth": True,
            "object_detection": True,
            "audio": True
        }
    }
