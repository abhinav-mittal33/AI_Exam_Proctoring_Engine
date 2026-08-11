import numpy as np
from typing import Dict, List

class StudentBaselineRecorder:
    def __init__(self):
        self.yaw_samples: List[float] = []
        self.pitch_samples: List[float] = []
        self.gaze_samples: List[float] = []
        self.mar_samples: List[float] = []

    def record_sample(self, yaw: float, pitch: float, gaze_ratio: float, mar: float):
        self.yaw_samples.append(abs(yaw))
        self.pitch_samples.append(abs(pitch))
        self.gaze_samples.append(abs(gaze_ratio - 0.5))
        self.mar_samples.append(mar)

    def get_baseline_percentiles(self) -> dict:
        if len(self.yaw_samples) == 0:
            return {
                "samples": 0,
                "yaw_P50": 5.0, "yaw_P75": 12.0, "yaw_P90": 18.0, "yaw_P95": 22.0, "yaw_P99": 28.0,
                "pitch_P50": 4.0, "pitch_P75": 10.0, "pitch_P90": 15.0, "pitch_P95": 18.0, "pitch_P99": 24.0
            }

        return {
            "samples": len(self.yaw_samples),
            "yaw_P50": round(float(np.percentile(self.yaw_samples, 50)), 2),
            "yaw_P75": round(float(np.percentile(self.yaw_samples, 75)), 2),
            "yaw_P90": round(float(np.percentile(self.yaw_samples, 90)), 2),
            "yaw_P95": round(float(np.percentile(self.yaw_samples, 95)), 2),
            "yaw_P99": round(float(np.percentile(self.yaw_samples, 99)), 2),
            "pitch_P50": round(float(np.percentile(self.pitch_samples, 50)), 2),
            "pitch_P75": round(float(np.percentile(self.pitch_samples, 75)), 2),
            "pitch_P90": round(float(np.percentile(self.pitch_samples, 90)), 2),
            "pitch_P95": round(float(np.percentile(self.pitch_samples, 95)), 2),
            "pitch_P99": round(float(np.percentile(self.pitch_samples, 99)), 2)
        }

student_baseline_recorder = StudentBaselineRecorder()
