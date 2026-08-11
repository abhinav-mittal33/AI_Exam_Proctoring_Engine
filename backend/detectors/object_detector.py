import cv2
import numpy as np
from ultralytics import YOLO

class ObjectDetector:
    def __init__(self):
        # Load YOLOv8n object detection model
        self.model = YOLO('yolov8n.pt')
        self.target_classes = {
            "cell phone": "PHONE_DETECTED",
            "book": "PROHIBITED_OBJECT_DETECTED",
            "laptop": "PROHIBITED_OBJECT_DETECTED",
            "remote": "PROHIBITED_OBJECT_DETECTED",
            "keyboard": "PROHIBITED_OBJECT_DETECTED",
            "mouse": "PROHIBITED_OBJECT_DETECTED"
        }

    def detect_objects(self, frame: np.ndarray) -> dict:
        """
        Detects prohibited objects using YOLOv8n.
        Returns list of detected objects with name, real confidence score, and bounding box.
        """
        if frame is None:
            return {"detected_objects": [], "has_prohibited": False, "count": 0}

        results = self.model(frame, verbose=False, conf=0.25)
        
        detected_objects = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for box in boxes:
                cls_id = int(box.cls[0].item())
                class_name = self.model.names.get(cls_id, "")
                
                if class_name in self.target_classes:
                    conf = float(box.conf[0].item())
                    xyxy = box.xyxy[0].tolist() # [x1, y1, x2, y2]
                    x1, y1, x2, y2 = [int(v) for v in xyxy]
                    
                    detected_objects.append({
                        "object_name": class_name,
                        "event_type": self.target_classes[class_name],
                        "confidence": round(conf, 2),
                        "bbox": [x1, y1, x2 - x1, y2 - y1]
                    })

        has_prohibited = len(detected_objects) > 0

        return {
            "detected_objects": detected_objects,
            "has_prohibited": has_prohibited,
            "count": len(detected_objects)
        }

object_detector = ObjectDetector()
