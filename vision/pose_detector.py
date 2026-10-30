import cv2
import numpy as np
import mediapipe as mp
from typing import Dict, Any, Tuple, Optional

class PoseDetector:
    def __init__(self, min_detection_confidence: float = 0.5, min_tracking_confidence: float = 0.5):
        self.mp_pose = mp.solutions.pose
        pose_options = {
            "static_image_mode": False,
            "smooth_landmarks": True,
            "min_detection_confidence": min_detection_confidence,
            "min_tracking_confidence": min_tracking_confidence,
        }
        try:
            # The heavy model offers higher accuracy, but some hosted Linux
            # environments ship its .tflite file without read permission.
            self.pose = self.mp_pose.Pose(model_complexity=2, **pose_options)
        except PermissionError:
            # Fall back to MediaPipe's full model so analysis can still run.
            self.pose = self.mp_pose.Pose(model_complexity=1, **pose_options)

    def process_frame(self, frame: np.ndarray) -> Tuple[Optional[Dict[str, Tuple[float, float, float]]], float]:
        """
        Processes an RGB frame and returns normalized + pixel coordinates for key landmarks,
        along with an overall confidence score.
        """
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)

        if not results.pose_landmarks:
            return None, 0.0

        landmarks = results.pose_landmarks.landmark
        
        # Pick dominant side (left vs right) based on average visibility confidence
        left_ids = [11, 13, 15, 23, 25, 27, 29, 31]
        right_ids = [12, 14, 16, 24, 26, 28, 30, 32]
        
        left_vis = np.mean([landmarks[i].visibility for i in left_ids])
        right_vis = np.mean([landmarks[i].visibility for i in right_ids])
        
        side = "left" if left_vis >= right_vis else "right"
        
        def get_pt(idx: int) -> Tuple[float, float, float]:
            lm = landmarks[idx]
            return (lm.x * w, lm.y * h, lm.visibility)

        idx_map = {
            "shoulder": 11 if side == "left" else 12,
            "elbow": 13 if side == "left" else 14,
            "wrist": 15 if side == "left" else 16,
            "hip": 23 if side == "left" else 24,
            "knee": 25 if side == "left" else 26,
            "ankle": 27 if side == "left" else 28,
            "heel": 29 if side == "left" else 30,
            "toe": 31 if side == "left" else 32,
            "ear": 7 if side == "left" else 8,
            "eye": 2 if side == "left" else 5,
            "nose": 0
        }

        lm_dict = {}
        for key, idx in idx_map.items():
            lm_dict[key] = get_pt(idx)

        # Also store opposite side landmarks for reference if needed
        opp_side = "right" if side == "left" else "left"
        lm_dict["opp_hip"] = get_pt(24 if opp_side == "right" else 23)
        lm_dict["opp_knee"] = get_pt(26 if opp_side == "right" else 25)

        avg_conf = float(np.mean([pt[2] for pt in lm_dict.values()]))
        return lm_dict, avg_conf

    def close(self):
        self.pose.close()
