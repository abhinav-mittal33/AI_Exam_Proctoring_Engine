import cv2
import numpy as np

class FrameQualityGate:
    def __init__(self):
        self.min_brightness = 25.0
        self.max_brightness = 245.0
        self.min_laplacian_var = 12.0
        self.min_face_size_px = 60

    def evaluate_quality(self, frame: np.ndarray, bboxes: list = None) -> dict:
        """
        Evaluates frame image quality before passing to behavioral detectors.
        Returns state: GOOD, POOR_LIGHTING, TOO_DARK, BLURRY, STALE, FACE_UNCERTAIN, UNUSABLE
        """
        if frame is None or frame.size == 0:
            return {
                "quality_state": "UNUSABLE",
                "is_usable": False,
                "brightness": 0.0,
                "blur_var": 0.0,
                "reason": "Empty or unreadable video frame."
            }

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray))
        blur_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        # Check extreme darkness
        if brightness < self.min_brightness:
            return {
                "quality_state": "TOO_DARK",
                "is_usable": False,
                "brightness": round(brightness, 1),
                "blur_var": round(blur_var, 1),
                "reason": "Camera environment too dark for reliable detection."
            }

        # Check heavy blur
        if blur_var < self.min_laplacian_var:
            return {
                "quality_state": "BLURRY",
                "is_usable": False,
                "brightness": round(brightness, 1),
                "blur_var": round(blur_var, 1),
                "reason": "Frame severely blurred or out of focus."
            }

        # Check face size if bbox provided
        if bboxes and len(bboxes) > 0:
            w, h = bboxes[0][2], bboxes[0][3]
            if w < self.min_face_size_px or h < self.min_face_size_px:
                return {
                    "quality_state": "FACE_UNCERTAIN",
                    "is_usable": False,
                    "brightness": round(brightness, 1),
                    "blur_var": round(blur_var, 1),
                    "reason": "Face too far or too small in frame."
                }

        # Check poor lighting warning (usable but warning)
        quality_state = "POOR_LIGHTING" if brightness < 40.0 else "GOOD"

        return {
            "quality_state": quality_state,
            "is_usable": True,
            "brightness": round(brightness, 1),
            "blur_var": round(blur_var, 1),
            "reason": "Frame quality acceptable for detection."
        }

quality_gate = FrameQualityGate()
