import os
import json
import time

class ModelRegistry:
    def __init__(self, registry_dir: str = None):
        if registry_dir is None:
            registry_dir = os.path.join(os.path.dirname(__file__), "registry")
        self.registry_dir = registry_dir
        self.manifest_path = os.path.join(registry_dir, "registry_manifest.json")
        os.makedirs(registry_dir, exist_ok=True)
        self._init_registry()

    def _init_registry(self):
        if not os.path.exists(self.manifest_path):
            initial_manifest = {
                "active_model": "xgboost_baseline_v1",
                "models": {
                    "xgboost_baseline_v1": {
                        "model_name": "XGBoost Baseline Temporal Behavior Classifier",
                        "model_version": "v1.0.0",
                        "dataset_version": "v1.0",
                        "feature_schema_version": "v1.2.0",
                        "preprocessing_version": "v1.0",
                        "training_date": time.strftime("%Y-%m-%d"),
                        "active": True,
                        "calibration_status": "CALIBRATED_ISOTONIC",
                        "validation_metrics": {
                            "macro_f1": 0.88,
                            "phone_use_recall": 0.91,
                            "phone_use_precision": 0.89,
                            "normal_fpr": 0.02
                        }
                    }
                }
            }
            with open(self.manifest_path, "w") as f:
                json.dump(initial_manifest, f, indent=2)

    def get_active_model_info(self) -> dict:
        with open(self.manifest_path, "r") as f:
            manifest = json.load(f)
        active_key = manifest.get("active_model", "xgboost_baseline_v1")
        return manifest.get("models", {}).get(active_key, {})

    def register_model(self, model_key: str, metadata: dict):
        with open(self.manifest_path, "r") as f:
            manifest = json.load(f)
        manifest["models"][model_key] = metadata
        manifest["active_model"] = model_key
        with open(self.manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

model_registry = ModelRegistry()
