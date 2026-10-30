import cv2
import numpy as np
from typing import Dict, Any, Tuple, Optional

class BarbellTracker:
    def __init__(self):
        self.prev_bar_pos = None
        self.smooth_alpha = 0.35 # EMA smoothing factor

    def track(self, frame: np.ndarray, landmarks: Optional[Dict[str, Tuple[float, float, float]]]) -> Tuple[Tuple[float, float], float]:
        """
        Returns (bar_x, bar_y) and confidence (0.0 to 1.0).
        Uses pose shoulder/wrist guiding combined with localized circle/contour detection.
        """
        if landmarks is None or "shoulder" not in landmarks:
            if self.prev_bar_pos is not None:
                return self.prev_bar_pos, 0.3
            return (0.0, 0.0), 0.0

        sh_x, sh_y, sh_vis = landmarks["shoulder"]
        
        # Default bar position estimate anchored near shoulder
        est_bar_x = sh_x
        est_bar_y = sh_y + 10.0 # Slightly below top of shoulder for low bar

        # Local ROI search around shoulder for circular barbell plate
        h, w, _ = frame.shape
        roi_size = int(min(w, h) * 0.18)
        x1 = max(0, int(est_bar_x - roi_size))
        y1 = max(0, int(est_bar_y - roi_size))
        x2 = min(w, int(est_bar_x + roi_size))
        y2 = min(h, int(est_bar_y + roi_size))

        detected_bar = None
        # A shoulder-anchored estimate is only a fallback, not a detected bar.
        # Keep its confidence below the processor's tracked threshold so the
        # report does not present this proxy as a measured bar path.
        conf = 0.3

        if x2 > x1 and y2 > y1:
            roi = frame[y1:y2, x1:x2]
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)

            # Circle detection
            circles = cv2.HoughCircles(
                blurred,
                cv2.HOUGH_GRADIENT,
                dp=1.2,
                minDist=30,
                param1=50,
                param2=30,
                minRadius=int(roi_size * 0.2),
                maxRadius=int(roi_size * 0.95)
            )

            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                # Pick circle closest to center of ROI
                roi_cx, roi_cy = (x2 - x1) / 2.0, (y2 - y1) / 2.0
                best_c = min(circles, key=lambda c: (c[0] - roi_cx)**2 + (c[1] - roi_cy)**2)
                detected_bar = (x1 + best_c[0], y1 + best_c[1])
                conf = 0.9

        if detected_bar is None:
            detected_bar = (est_bar_x, est_bar_y)

        # Smooth with EMA
        if self.prev_bar_pos is not None:
            sm_x = self.smooth_alpha * detected_bar[0] + (1 - self.smooth_alpha) * self.prev_bar_pos[0]
            sm_y = self.smooth_alpha * detected_bar[1] + (1 - self.smooth_alpha) * self.prev_bar_pos[1]
            final_pos = (sm_x, sm_y)
        else:
            final_pos = detected_bar

        self.prev_bar_pos = final_pos
        return final_pos, conf
