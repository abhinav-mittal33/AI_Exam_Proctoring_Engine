import os
import json
import time
import cv2
from typing import List, Dict

TARGET_CLASSES = [
    "NORMAL",
    "PHONE_USE",
    "MULTIPLE_PERSON",
    "FACE_ABSENT",
    "PERSISTENT_GAZE_AWAY",
    "PERSISTENT_HEAD_TURN",
    "MOUTH_ACTIVITY",
    "CAMERA_PROBLEM"
]

PHONE_VARIATIONS = [
    "PHONE_ON_DESK", "PHONE_IN_HAND", "PHONE_NEAR_FACE", "PHONE_PARTIALLY_OCCLUDED",
    "PHONE_VERTICAL", "PHONE_HORIZONTAL", "HAND_WITHOUT_PHONE", "BOOK", "WATER_BOTTLE"
]

class DataCollector:
    def __init__(self, base_dir: str = None):
        if base_dir is None:
            base_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "datasets")
        self.raw_dir = os.path.join(base_dir, "raw")
        os.makedirs(self.raw_dir, exist_ok=True)

    def record_clip(self, duration_sec: float, label: str, variation: str = "DEFAULT", session_id: str = "collector_session") -> dict:
        """
        Records a video clip + JSON feature stream from webcam for dataset collection.
        """
        if label not in TARGET_CLASSES:
            raise ValueError(f"Invalid label {label}. Must be one of {TARGET_CLASSES}")

        clip_id = f"clip_{label.lower()}_{int(time.time())}"
        video_path = os.path.join(self.raw_dir, f"{clip_id}.mp4")
        features_path = os.path.join(self.raw_dir, f"{clip_id}_features.json")
        meta_path = os.path.join(self.raw_dir, f"{clip_id}_meta.json")

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return {"success": False, "error": "Could not access webcam for recording."}

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, 20.0, (640, 480))

        start_t = time.time()
        frames_recorded = 0

        print(f"[{clip_id}] Recording {duration_sec}s for label '{label}' ({variation})... Get ready!")

        while (time.time() - start_t) < duration_sec:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
            frames_recorded += 1
            time.sleep(0.04) # ~20 FPS

        cap.release()
        out.release()

        meta = {
            "clip_id": clip_id,
            "session_id": session_id,
            "label": label,
            "variation": variation,
            "duration_sec": round(time.time() - start_t, 2),
            "frames_recorded": frames_recorded,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        return {"success": True, "clip_id": clip_id, "meta": meta}

data_collector = DataCollector()
