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
    def calculate_cervical_tilt(ear: Tuple[float, float], eye: Tuple[float, float]) -> float:
        """
        Calculates head / cervical tilt angle relative to the horizontal plane.
        Negative values indicate head tilted downward (chin tucked / looking down at floor).
        Positive values indicate head tilted upward toward ceiling.
        """
        # In image coords, Y increases downwards.
        # Downward gaze: eye is lower than ear (eye[1] > ear[1])
        dx = abs(eye[0] - ear[0]) + 1e-6
        dy = eye[1] - ear[1]
        # Invert so downward angle is negative relative to horizontal
        tilt_angle = -math.degrees(math.atan2(dy, dx))
        return tilt_angle

    @staticmethod
    def compute_frame_telemetry(landmarks: Optional[Dict[str, Tuple[float, float, float]]], bar_pos: Tuple[float, float]) -> Dict[str, Any]:
        """
        Calculates all biomechanical telemetry for a single frame.
        Includes pixel values, body-scaled normalized values, and estimated physical units (cm).
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

        # 1. Scale Estimation (Standard adult human foot length ~26.5 cm)
        foot_length_px = max(abs(toe[0] - heel[0]), 25.0)
        cm_per_px = 26.5 / foot_length_px

        # 2. Joint angles
        knee_angle = BiomechanicsCalculator.calculate_angle(hip, knee, ankle)
        hip_angle = BiomechanicsCalculator.calculate_angle(sh, hip, knee)
        back_angle = BiomechanicsCalculator.calculate_back_angle(sh, hip)
        cervical_tilt = BiomechanicsCalculator.calculate_cervical_tilt(ear, eye)

        # 3. Midfoot & Bar path
        midfoot_x = (heel[0] + toe[0]) / 2.0
        bar_x, bar_y = bar_pos
        bar_dev_px = bar_x - midfoot_x
        bar_dev_cm = bar_dev_px * cm_per_px
        norm_bar_dev = abs(bar_dev_px) / foot_length_px

        # 4. Depth (Hip crease vs Top of patella)
        # Top of patella is ~12% of thigh length above knee landmark center
        thigh_len_px = math.hypot(hip[0] - knee[0], hip[1] - knee[1])
        patella_top_y = knee[1] - (0.12 * thigh_len_px)
        hip_crease_y = hip[1]  # Hip landmark is apex of crease
        
        # In image space Y increases downwards. hip_crease_y > patella_top_y means hip is below patella top (below parallel)
        depth_delta_px = hip_crease_y - patella_top_y
        depth_delta_cm = depth_delta_px * cm_per_px
        depth_norm = depth_delta_px / max(thigh_len_px, 1.0)
        is_deep = depth_delta_px > 0

        # 5. Knee forward travel relative to toe
        # Facing right: toe[0] > heel[0] -> knee forward if knee[0] > toe[0]
        # Facing left: toe[0] < heel[0] -> knee forward if knee[0] < toe[0]
        if toe[0] >= heel[0]:
            knee_forward_px = max(0.0, knee[0] - toe[0])
        else:
            knee_forward_px = max(0.0, toe[0] - knee[0])
            
        knee_forward_norm = knee_forward_px / foot_length_px
        knee_forward_cm = knee_forward_px * cm_per_px

        return {
            "valid": True,
            "foot_length_px": foot_length_px,
            "cm_per_px": cm_per_px,
            "knee_angle": knee_angle,
            "hip_angle": hip_angle,
            "back_angle": back_angle,
            "cervical_tilt": cervical_tilt,
            "gaze_angle": None,  # Explicitly None because iris direction is not observable
            "midfoot_x": midfoot_x,
            "bar_x": bar_x,
            "bar_y": bar_y,
            "bar_dev_px": bar_dev_px,
            "bar_dev_cm": bar_dev_cm,
            "norm_bar_dev": norm_bar_dev,
            "depth_delta_px": depth_delta_px,
            "depth_delta_cm": depth_delta_cm,
            "depth_norm": depth_norm,
            "is_deep": is_deep,
            "knee_forward_px": knee_forward_px,
            "knee_forward_norm": knee_forward_norm,
            "knee_forward_cm": knee_forward_cm,
            "hip_y": hip_crease_y,
            "shoulder_y": sh[1],
            "knee_x": knee[0],
            "knee_y": knee[1],
            "toe_x": toe[0],
            "patella_y": patella_top_y
        }
