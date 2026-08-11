import numpy as np
from typing import Dict, List

class ModelEvaluator:
    def __init__(self, class_labels: List[str] = None):
        self.class_labels = class_labels or [
            "NORMAL", "PHONE_USE", "MULTIPLE_PERSON", "FACE_ABSENT",
            "PERSISTENT_GAZE_AWAY", "PERSISTENT_HEAD_TURN", "MOUTH_ACTIVITY", "CAMERA_PROBLEM"
        ]

    def evaluate_predictions(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """
        Calculates Precision, Recall, F1, Support, and Confusion Matrix per class.
        """
        n_classes = len(self.class_labels)
        conf_matrix = np.zeros((n_classes, n_classes), dtype=int)

        for t, p in zip(y_true, y_pred):
            if 0 <= t < n_classes and 0 <= p < n_classes:
                conf_matrix[t, p] += 1

        per_class_metrics = {}
        for i, cls_name in enumerate(self.class_labels):
            tp = float(conf_matrix[i, i])
            fp = float(np.sum(conf_matrix[:, i]) - tp)
            fn = float(np.sum(conf_matrix[i, :]) - tp)
            tn = float(np.sum(conf_matrix) - (tp + fp + fn))

            precision = tp / max(1.0, tp + fp)
            recall = tp / max(1.0, tp + fn)
            f1 = 2.0 * (precision * recall) / max(0.001, precision + recall)
            fpr = fp / max(1.0, fp + tn)

            per_class_metrics[cls_name] = {
                "precision": round(float(precision), 3),
                "recall": round(float(recall), 3),
                "f1_score": round(float(f1), 3),
                "false_positive_rate": round(float(fpr), 3),
                "support": int(np.sum(conf_matrix[i, :]))
            }

        macro_f1 = float(np.mean([m["f1_score"] for m in per_class_metrics.values()]))

        return {
            "macro_f1": round(macro_f1, 3),
            "per_class": per_class_metrics,
            "confusion_matrix": conf_matrix.tolist()
        }

model_evaluator = ModelEvaluator()
