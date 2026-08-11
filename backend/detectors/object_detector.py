import cv2
import numpy as np
from ultralytics import YOLO
from typing import Dict, List
from backend.config.settings import settings

class ObjectDetector:
    def __init__(self):
        self.model = YOLO('yolov8n.pt')
        self.target_classes = {
            "cell phone": "PHONE_DETECTED",
            "book": "PROHIBITED_OBJECT_DETECTED",
            "laptop": "PROHIBITED_OBJECT_DETECTED",
            "remote": "PROHIBITED_OBJECT_DETECTED",
            "keyboard": "PROHIBITED_OBJECT_DETECTED",
            "mouse": "PROHIBITED_OBJECT_DETECTED"
        }
        self.rolling_windows: Dict[str, List[bool]] = {}

    def detect_objects(self, frame: np.ndarray, session_id: str = "default_session") -> dict:
        """
        Detects prohibited objects using YOLOv8n with conf=0.55 operating threshold
        and 20-frame temporal consistency (N / 20).
        """
        if frame is None or frame.size == 0:
            return {"detected_objects": [], "has_prohibited": False, "count": 0, "persistence_ratio": 0.0}

        # Run inference with conf=0.55 conservative operating threshold
        results = self.model(frame, verbose=False, conf=settings.YOLO_INITIAL_CONFIDENCE)
        
        detected_objects = []
        is_detected_in_frame = False

        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for box in boxes:
                cls_id = int(box.cls[0].item())
                class_name = self.model.names.get(cls_id, "")
                
                if class_name in self.target_classes:
                    conf = float(box.conf[0].item())
                    xyxy = box.xyxy[0].tolist()
                    x1, y1, x2, y2 = [int(v) for v in xyxy]
                    is_detected_in_frame = True
                    
                    display_name = "notebook / book" if class_name == "book" else class_name
                    
                    detected_objects.append({
                        "object_name": display_name,
                        "raw_class": class_name,
                        "event_type": self.target_classes[class_name],
                        "confidence": round(conf, 2),
                        "bbox": [x1, y1, x2 - x1, y2 - y1]
                    })

        # Update 20-frame rolling window
        if session_id not in self.rolling_windows:
            self.rolling_windows[session_id] = []

        window = self.rolling_windows[session_id]
        window.append(is_detected_in_frame)
        if len(window) > settings.ROLLING_WINDOW_FRAMES:
            window.pop(0)

        positive_count = sum(1 for b in window if b)
        persistence_ratio = round(positive_count / max(1.0, float(len(window))), 2)

        for obj in detected_objects:
            obj["persistence_ratio"] = persistence_ratio

        # Require at least 40% persistence over 20-frame window to prevent false positives
        has_prohibited = len(detected_objects) > 0 and persistence_ratio >= 0.40

        return {
            "detected_objects": detected_objects,
            "has_prohibited": has_prohibited,
            "count": len(detected_objects),
            "persistence_ratio": persistence_ratio
        }

object_detector = ObjectDetector()
