import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional

class VideoAnnotator:
    def __init__(self):
        # Color palette (BGR format)
        self.COLOR_GREEN = (0, 230, 115)     # Primary accent / Pass
        self.COLOR_RED = (80, 80, 255)       # Fail / Warning
        self.COLOR_CYAN = (255, 230, 0)      # Joint / Line accent
        self.COLOR_PURPLE = (235, 115, 180)  # Bar path
        self.COLOR_WHITE = (255, 255, 255)
        self.COLOR_DARK_BG = (20, 20, 25)

    def draw_dashed_line(self, img: np.ndarray, pt1: Tuple[int, int], pt2: Tuple[int, int], color: Tuple[int, int, int], thickness: int = 2, dash_len: int = 10):
        dist = np.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1])
        if dist == 0:
            return
        dashes = int(dist / dash_len)
        for i in range(dashes):
            start = (
                int(pt1[0] + (pt2[0] - pt1[0]) * (i / dashes)),
                int(pt1[1] + (pt2[1] - pt1[1]) * (i / dashes))
            )
            end = (
                int(pt1[0] + (pt2[0] - pt1[0]) * ((i + 0.5) / dashes)),
                int(pt1[1] + (pt2[1] - pt1[1]) * ((i + 0.5) / dashes))
            )
            cv2.line(img, start, end, color, thickness)

    def annotate_frame(
        self,
        frame: np.ndarray,
        landmarks: Dict[str, Tuple[float, float, float]],
        telemetry: Dict[str, Any],
        bar_history: List[Tuple[float, float]],
        current_rep: Optional[Dict[str, Any]],
        rep_phase: str
    ) -> np.ndarray:
        out_img = frame.copy()
        h, w, _ = out_img.shape

        if not telemetry.get("valid", False):
            return out_img

        # 1. Draw Skeleton Stickman
        sh = (int(landmarks["shoulder"][0]), int(landmarks["shoulder"][1]))
        hip = (int(landmarks["hip"][0]), int(landmarks["hip"][1]))
        knee = (int(landmarks["knee"][0]), int(landmarks["knee"][1]))
        ankle = (int(landmarks["ankle"][0]), int(landmarks["ankle"][1]))
        heel = (int(landmarks["heel"][0]), int(landmarks["heel"][1])) if "heel" in landmarks else ankle
        toe = (int(landmarks["toe"][0]), int(landmarks["toe"][1])) if "toe" in landmarks else ankle
        ear = (int(landmarks["ear"][0]), int(landmarks["ear"][1])) if "ear" in landmarks else sh
        eye = (int(landmarks["eye"][0]), int(landmarks["eye"][1])) if "eye" in landmarks else ear

        bones = [
            (sh, hip), (hip, knee), (knee, ankle), (ankle, heel), (heel, toe), (sh, ear), (ear, eye)
        ]
        if "elbow" in landmarks and "wrist" in landmarks:
            elbow = (int(landmarks["elbow"][0]), int(landmarks["elbow"][1]))
            wrist = (int(landmarks["wrist"][0]), int(landmarks["wrist"][1]))
            bones.extend([(sh, elbow), (elbow, wrist)])

        for b1, b2 in bones:
            cv2.line(out_img, b1, b2, self.COLOR_CYAN, 3, cv2.LINE_AA)

        # Draw joints
        for pt in [sh, hip, knee, ankle, heel, toe, ear, eye]:
            cv2.circle(out_img, pt, 6, self.COLOR_WHITE, -1, cv2.LINE_AA)
            cv2.circle(out_img, pt, 7, self.COLOR_CYAN, 2, cv2.LINE_AA)

        # 2. Midfoot Vertical Plumb Line
        midfoot_x = int(telemetry["midfoot_x"])
        self.draw_dashed_line(out_img, (midfoot_x, 0), (midfoot_x, h), (200, 200, 200), 2, 12)

        # 3. Bar Path Trajectory Ribbon
        if len(bar_history) > 1:
            for i in range(1, len(bar_history)):
                pt1 = (int(bar_history[i-1][0]), int(bar_history[i-1][1]))
                pt2 = (int(bar_history[i][0]), int(bar_history[i][1]))
                dev = abs(bar_history[i][0] - midfoot_x)
                color = self.COLOR_GREEN if dev <= 25 else self.COLOR_RED
                cv2.line(out_img, pt1, pt2, color, 3, cv2.LINE_AA)

        # Draw current Barbell Center
        bar_x, bar_y = int(telemetry["bar_x"]), int(telemetry["bar_y"])
        cv2.circle(out_img, (bar_x, bar_y), 10, self.COLOR_PURPLE, -1, cv2.LINE_AA)
        cv2.circle(out_img, (bar_x, bar_y), 12, self.COLOR_WHITE, 2, cv2.LINE_AA)

        # 4. Depth Reference Line at bottom position or active descent
        if rep_phase in ["BOTTOM", "DESCENT", "ASCENT"]:
            patella_y = int(telemetry["patella_y"])
            hip_y = int(telemetry["hip_y"])
            is_deep = telemetry["is_deep"]
            depth_color = self.COLOR_GREEN if is_deep else self.COLOR_RED

            cv2.line(out_img, (hip[0] - 40, patella_y), (hip[0] + 40, patella_y), (255, 255, 255), 2, cv2.LINE_AA)
            cv2.line(out_img, (hip[0] - 40, hip_y), (hip[0] + 40, hip_y), depth_color, 2, cv2.LINE_AA)

        # 5. Top Left HUD Badge
        hud_h = 110
        cv2.rectangle(out_img, (15, 15), (320, 15 + hud_h), self.COLOR_DARK_BG, -1)
        cv2.rectangle(out_img, (15, 15), (320, 15 + hud_h), (60, 60, 70), 2)

        rep_str = f"REP: {current_rep['rep_number']}" if current_rep else "REP: SETUP"
        cv2.putText(out_img, rep_str, (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.75, self.COLOR_WHITE, 2, cv2.LINE_AA)

        phase_color = self.COLOR_GREEN if rep_phase == "BOTTOM" else (self.COLOR_CYAN if "DESCENT" in rep_phase or "ASCENT" in rep_phase else self.COLOR_WHITE)
        cv2.putText(out_img, f"PHASE: {rep_phase}", (30, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.60, phase_color, 2, cv2.LINE_AA)

        knee_deg = telemetry.get("knee_angle", 0)
        back_deg = telemetry.get("back_angle", 0)
        cv2.putText(out_img, f"KNEE: {knee_deg:.0f}° | BACK: {back_deg:.0f}°", (30, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

        # Bottom depth badge
        if rep_phase == "BOTTOM":
            badge_text = "PASSED DEPTH" if telemetry["is_deep"] else "PARTIAL DEPTH"
            b_color = self.COLOR_GREEN if telemetry["is_deep"] else self.COLOR_RED
            cv2.rectangle(out_img, (w - 220, 15), (w - 15, 60), self.COLOR_DARK_BG, -1)
            cv2.rectangle(out_img, (w - 220, 15), (w - 15, 60), b_color, 2)
            cv2.putText(out_img, badge_text, (w - 205, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.65, b_color, 2, cv2.LINE_AA)

        return out_img
