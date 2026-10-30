import math
import numpy as np
from typing import Dict, Any, Tuple, Optional

class BiomechanicsCalculator:
    @staticmethod
    def calculate_angle(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> float:
        """
        Calculates angle at vertex b given points a, b, c in degrees.
        """
        ang = math.degrees(
            math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0])
        )
        ang = abs(ang)
        if ang > 180.0:
            ang = 360.0 - ang
        return ang

    @staticmethod
    def calculate_back_angle(shoulder: Tuple[float, float], hip: Tuple[float, float]) -> float:
        """
        Calculates back angle of torso (shoulder to hip) relative to horizontal ground plane.
        Horizontal floor = 0°, vertical upright = 90°.
        """
        dx = abs(shoulder[0] - hip[0])
        dy = abs(hip[1] - shoulder[1])
        angle = math.degrees(math.atan2(dy, dx + 1e-6))
        return angle

    @staticmethod
    def calculate_gaze_angle(ear: Tuple[float, float], eye: Tuple[float, float], nose: Tuple[float, float]) -> float:
        """
        Calculates eye gaze angle relative to horizontal plane.
        Negative values indicates looking downward at floor.
        Positive values indicates looking upward at ceiling.
        """
        # Head vector from ear to eye/nose. In image space, Y increases downwards.
        # When looking down, eye is lower than ear (eye[1] > ear[1]), so ear[1] - eye[1] is negative.
        ref_x = eye[0] - ear[0]
        ref_y = ear[1] - eye[1]
        angle = math.degrees(math.atan2(ref_y, abs(ref_x) + 1e-6))
        return angle

    @staticmethod
    def compute_frame_telemetry(landmarks: Optional[Dict[str, Tuple[float, float, float]]], bar_pos: Tuple[float, float]) -> Dict[str, Any]:
        """
        Calculates all biomechanical telemetry for a single frame.
        """
        if not landmarks or "hip" not in landmarks or "knee" not in landmarks or "ankle" not in landmarks:
            return {"valid": False}

        sh = landmarks["shoulder"][:2]
        hip = landmarks["hip"][:2]
        knee = landmarks["knee"][:2]
        ankle = landmarks["ankle"][:2]
        heel = landmarks["heel"][:2] if "heel" in landmarks else ankle
        toe = landmarks["toe"][:2] if "toe" in landmarks else (ankle[0] + 15, ankle[1])
        ear = landmarks["ear"][:2] if "ear" in landmarks else sh
        eye = landmarks["eye"][:2] if "eye" in landmarks else ear

        # 1. Joint angles
        knee_angle = BiomechanicsCalculator.calculate_angle(hip, knee, ankle)
        hip_angle = BiomechanicsCalculator.calculate_angle(sh, hip, knee)
        back_angle = BiomechanicsCalculator.calculate_back_angle(sh, hip)
        # MediaPipe Pose does not provide iris direction. Eye-to-ear geometry
        # describes head pose, so do not report it as measured eye gaze.
        gaze_angle = None

        # 2. Midfoot & Bar path
        midfoot_x = (heel[0] + toe[0]) / 2.0
        bar_x, bar_y = bar_pos
        bar_dev_px = bar_x - midfoot_x
        
        foot_len = max(abs(toe[0] - heel[0]), 20.0)
        norm_bar_dev = abs(bar_dev_px) / foot_len

        # 3. Depth (Hip crease vs Top of patella)
        # Top of patella is ~12% of thigh length above knee landmark center
        thigh_len = math.hypot(hip[0] - knee[0], hip[1] - knee[1])
        patella_top_y = knee[1] - (0.12 * thigh_len)
        hip_crease_y = hip[1] # Hip landmark is apex of crease
        
        # In image space Y increases downwards. So hip_crease_y > patella_top_y means hip is deeper than patella top!
        depth_delta_px = hip_crease_y - patella_top_y
        is_deep = depth_delta_px > 0

        # 4. Knee forward travel relative to toe
        # If facing right (toe_x > heel_x): knee is in front of toe if knee_x > toe_x
        # If facing left (toe_x < heel_x): knee is in front of toe if knee_x < toe_x
        if toe[0] > heel[0]:
            knee_forward_px = max(0.0, knee[0] - toe[0])
        else:
            knee_forward_px = max(0.0, toe[0] - knee[0])

        return {
            "valid": True,
            "knee_angle": knee_angle,
            "hip_angle": hip_angle,
            "back_angle": back_angle,
            "gaze_angle": gaze_angle,
            "midfoot_x": midfoot_x,
            "bar_x": bar_x,
            "bar_y": bar_y,
            "bar_dev_px": bar_dev_px,
            "norm_bar_dev": norm_bar_dev,
            "depth_delta_px": depth_delta_px,
            "is_deep": is_deep,
            "knee_forward_px": knee_forward_px,
            "hip_y": hip_crease_y,
            "shoulder_y": sh[1],
            "patella_y": patella_top_y
        }
