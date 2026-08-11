from typing import Dict, List

class VideoEvaluator:
    def __init__(self):
        self.dataset_classes = [
            "NORMAL", "GAZE", "HEAD", "FACE_MISSING", "MULTIPLE_FACES", "PHONE", "MOUTH", "MIXED"
        ]

    def evaluate_pipeline(self, test_results: List[dict]) -> dict:
        """
        Calculates Precision, Recall, F1, False Positive Rate (FPR), and False Negative Rate (FNR).
        """
        tp = sum(1 for r in test_results if r.get("is_correct") and r.get("is_positive"))
        fp = sum(1 for r in test_results if not r.get("is_correct") and not r.get("is_positive"))
        fn = sum(1 for r in test_results if not r.get("is_correct") and r.get("is_positive"))
        tn = sum(1 for r in test_results if r.get("is_correct") and not r.get("is_positive"))

        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2.0 * (precision * recall) / max(0.001, precision + recall)
        fpr = fp / max(1, fp + tn)
        fnr = fn / max(1, fn + tp)

        return {
            "total_frames_evaluated": len(test_results),
            "precision": round(float(precision), 3),
            "recall": round(float(recall), 3),
            "f1_score": round(float(f1), 3),
            "false_positive_rate": round(float(fpr), 3),
            "false_negative_rate": round(float(fnr), 3)
        }

video_evaluator = VideoEvaluator()
