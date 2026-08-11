import os
import json
import random
from typing import Dict, List

class DatasetPipeline:
    def __init__(self, base_dir: str = None):
        if base_dir is None:
            base_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "datasets")
        self.base_dir = base_dir
        self.raw_dir = os.path.join(base_dir, "raw")
        self.processed_dir = os.path.join(base_dir, "processed")
        self.manifest_path = os.path.join(base_dir, "manifest.json")
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)

    def validate_dataset(self) -> dict:
        """
        Scans datasets/raw/ metadata files, checks class balance, and checks for data leakage.
        """
        meta_files = [f for f in os.listdir(self.raw_dir) if f.endswith("_meta.json")]
        
        class_counts = {
            "NORMAL": 0, "PHONE_USE": 0, "MULTIPLE_PERSON": 0, "FACE_ABSENT": 0,
            "PERSISTENT_GAZE_AWAY": 0, "PERSISTENT_HEAD_TURN": 0, "MOUTH_ACTIVITY": 0, "CAMERA_PROBLEM": 0
        }
        
        manifest_items = []
        sessions = set()

        for mf in meta_files:
            p = os.path.join(self.raw_dir, mf)
            with open(p, "r") as f:
                data = json.load(f)
                lbl = data.get("label", "UNCERTAIN")
                if lbl in class_counts:
                    class_counts[lbl] += 1
                sess_id = data.get("session_id", "sess_unknown")
                sessions.add(sess_id)
                manifest_items.append(data)

        total_clips = len(manifest_items)

        is_sufficient = total_clips >= 20 and min(class_counts.values()) >= 2
        status_str = "SUFFICIENT" if is_sufficient else "TRAINING BLOCKED — INSUFFICIENT LABELED DATA"

        report = {
            "status": status_str,
            "total_clips": total_clips,
            "total_sessions": len(sessions),
            "class_distribution": class_counts,
            "is_sufficient": is_sufficient
        }
        return report

    def partition_dataset_no_leakage(self, train_ratio: float = 0.7, val_ratio: float = 0.15, test_ratio: float = 0.15) -> dict:
        """
        Partitions dataset BY SESSION ID (not random frames) to prevent data leakage.
        """
        report = self.validate_dataset()
        if not report["is_sufficient"]:
            return {"status": "BLOCKED", "reason": "Insufficient labeled clips. Record dataset clips using data_collector."}

        meta_files = [f for f in os.listdir(self.raw_dir) if f.endswith("_meta.json")]
        sessions: Dict[str, List[dict]] = {}

        for mf in meta_files:
            p = os.path.join(self.raw_dir, mf)
            with open(p, "r") as f:
                data = json.load(f)
                s_id = data.get("session_id", mf)
                if s_id not in sessions:
                    sessions[s_id] = []
                sessions[s_id].append(data)

        sess_list = list(sessions.keys())
        random.seed(42)
        random.shuffle(sess_list)

        n_total = len(sess_list)
        n_train = max(1, int(n_total * train_ratio))
        n_val = max(1, int(n_total * val_ratio))

        train_sessions = sess_list[:n_train]
        val_sessions = sess_list[n_train:n_train + n_val]
        test_sessions = sess_list[n_train + n_val:]

        manifest = {
            "train": [item for s in train_sessions for item in sessions[s]],
            "validation": [item for s in val_sessions for item in sessions[s]],
            "test": [item for s in test_sessions for item in sessions[s]]
        }

        with open(self.manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        return {
            "status": "SUCCESS",
            "train_clips": len(manifest["train"]),
            "validation_clips": len(manifest["validation"]),
            "test_clips": len(manifest["test"])
        }

dataset_pipeline = DatasetPipeline()
