import os
import yaml
from pydantic_settings import BaseSettings

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "configs", "proctor_config.yaml")

def load_yaml_config():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r") as f:
            return yaml.safe_load(f)
    return {}

_raw_config = load_yaml_config()

class Settings(BaseSettings):
    APP_NAME: str = _raw_config.get("app", {}).get("name", "AI Exam Proctoring Engine")
    VERSION: str = _raw_config.get("app", {}).get("version", "1.0.0")
    
    TARGET_FPS: int = _raw_config.get("pipeline", {}).get("target_fps", 30)
    FACE_INTERVAL: int = _raw_config.get("pipeline", {}).get("face_interval_frames", 1)
    POSE_GAZE_INTERVAL: int = _raw_config.get("pipeline", {}).get("pose_gaze_interval_frames", 3)
    OBJECT_INTERVAL: int = _raw_config.get("pipeline", {}).get("object_interval_frames", 6)
    
    FACE_MISSING_GRACE_PERIOD: float = _raw_config.get("face", {}).get("missing_grace_period_seconds", 3.0)
    MULTI_FACE_GRACE_PERIOD: float = _raw_config.get("face", {}).get("multi_face_grace_period_seconds", 2.0)
    
    YAW_THRESHOLD: float = _raw_config.get("head_pose", {}).get("yaw_threshold", 25.0)
    PITCH_THRESHOLD: float = _raw_config.get("head_pose", {}).get("pitch_threshold", 20.0)
    ROLL_THRESHOLD: float = _raw_config.get("head_pose", {}).get("roll_threshold", 25.0)
    
    GAZE_HORIZONTAL_THRESHOLD: float = _raw_config.get("gaze", {}).get("horizontal_threshold", 0.35)
    GAZE_MIN_DURATION: float = _raw_config.get("gaze", {}).get("minimum_duration_seconds", 2.0)
    
    MAR_THRESHOLD: float = _raw_config.get("mouth", {}).get("mar_threshold", 0.40)
    MOUTH_MIN_DURATION: float = _raw_config.get("mouth", {}).get("minimum_duration_seconds", 1.5)
    
    PROHIBITED_OBJECTS: list = _raw_config.get("objects", {}).get("prohibited_classes", ["cell phone", "book", "laptop"])
    OBJECT_CONFIDENCE: float = _raw_config.get("objects", {}).get("confidence_threshold", 0.45)
    
    RISK_WEIGHTS: dict = _raw_config.get("risk", {}).get("weights", {
        "FACE_MISSING": 25,
        "MULTIPLE_FACES": 35,
        "HEAD_TURNED_LEFT": 10,
        "HEAD_TURNED_RIGHT": 10,
        "HEAD_TURNED_UP": 10,
        "HEAD_TURNED_DOWN": 10,
        "GAZE_AWAY": 8,
        "MOUTH_MOVEMENT": 10,
        "POSSIBLE_SPEECH": 20,
        "PHONE_DETECTED": 45,
        "PROHIBITED_OBJECT_DETECTED": 35,
        "AUDIO_ACTIVITY": 10,
        "CAMERA_INTERRUPTED": 30,
        "IDENTITY_MISMATCH": 60
    })

settings = Settings()
