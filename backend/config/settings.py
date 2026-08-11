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
    VERSION: str = _raw_config.get("app", {}).get("version", "1.2.0")
    
    TARGET_FPS: int = _raw_config.get("pipeline", {}).get("target_fps", 30)
    FACE_INTERVAL: int = _raw_config.get("pipeline", {}).get("face_interval_frames", 1)
    POSE_GAZE_INTERVAL: int = _raw_config.get("pipeline", {}).get("pose_gaze_interval_frames", 1)
    OBJECT_INTERVAL: int = _raw_config.get("pipeline", {}).get("object_interval_frames", 2)
    
    FACE_DETECTION_CONFIDENCE: float = _raw_config.get("face", {}).get("detection_confidence", 0.50)
    FACE_PRESENCE_CONFIDENCE: float = _raw_config.get("face", {}).get("presence_confidence", 0.60)
    FACE_TRACKING_CONFIDENCE: float = _raw_config.get("face", {}).get("tracking_confidence", 0.60)
    FACE_MISSING_GRACE_PERIOD: float = _raw_config.get("face", {}).get("missing_grace_period_seconds", 2.5)
    MULTIPLE_FACE_MIN_DURATION: float = _raw_config.get("face", {}).get("multi_face_min_duration_seconds", 0.75)
    
    HEAD_YAW_OBSERVATION: float = _raw_config.get("head_pose", {}).get("yaw_observation_degrees", 20.0)
    HEAD_YAW_EVENT: float = _raw_config.get("head_pose", {}).get("yaw_event_degrees", 30.0)
    HEAD_PITCH_OBSERVATION: float = _raw_config.get("head_pose", {}).get("pitch_observation_degrees", 15.0)
    HEAD_PITCH_EVENT: float = _raw_config.get("head_pose", {}).get("pitch_event_degrees", 25.0)
    HEAD_POSE_EVENT_DURATION: float = _raw_config.get("head_pose", {}).get("min_duration_seconds", 2.5)
    
    GAZE_EVENT_DURATION: float = _raw_config.get("gaze", {}).get("min_duration_seconds", 3.0)
    
    MAR_THRESHOLD: float = _raw_config.get("mouth", {}).get("mar_threshold", 0.32)
    MOUTH_EVENT_DURATION: float = _raw_config.get("mouth", {}).get("min_duration_seconds", 1.5)
    
    YOLO_INITIAL_CONFIDENCE: float = _raw_config.get("objects", {}).get("yolo_initial_confidence", 0.55)
    OBJECT_MIN_PERSISTENCE: float = _raw_config.get("objects", {}).get("min_persistence_seconds", 1.5)
    ROLLING_WINDOW_FRAMES: int = _raw_config.get("objects", {}).get("rolling_window_frames", 10)
    PROHIBITED_OBJECTS: list = _raw_config.get("objects", {}).get("prohibited_classes", ["cell phone", "book", "laptop"])
    
    SEVERITY_CAPS: dict = _raw_config.get("evidence", {}).get("severity_caps", {
        "LOW": 25.0, "MODERATE": 50.0, "HIGH": 75.0, "CRITICAL": 100.0
    })
    CATEGORY_CAPS: dict = _raw_config.get("evidence", {}).get("category_caps", {
        "HEAD": 15.0, "GAZE": 20.0, "MOUTH": 10.0, "OBJECT": 40.0, "FACE": 40.0, "IDENTITY": 100.0, "CAMERA": 30.0
    })
    BASE_SEVERITIES: dict = _raw_config.get("evidence", {}).get("base_severities", {
        "LOW": 10.0, "MODERATE": 25.0, "HIGH": 45.0, "CRITICAL": 85.0
    })
    
    RISK_DECAY_HALF_LIFE: float = _raw_config.get("risk", {}).get("decay_half_life_minutes", 20.0)
    RISK_SMOOTHING_ALPHA: float = _raw_config.get("risk", {}).get("smoothing_alpha", 0.25)
    CORRELATION_DISCOUNT: float = _raw_config.get("risk", {}).get("correlation_discount", 0.50)

settings = Settings()
