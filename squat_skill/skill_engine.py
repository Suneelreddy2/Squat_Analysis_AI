import os
import yaml
from typing import Dict, Any, List

class SquatSkillEngine:
    def __init__(self, rules_path: str = None):
        if rules_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            rules_path = os.path.join(base_dir, "squat_rules.yaml")
            
        with open(rules_path, "r", encoding="utf-8") as f:
            self.schema = yaml.safe_load(f)
            
        self.rules = {r["id"]: r for r in self.schema.get("rules", [])}

    def evaluate_repetition(self, telemetry: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Evaluates a single repetition against the document-derived skill rules.
        """
        findings = []
        confidence = telemetry.get("pose_confidence", 1.0)
        low_confidence = confidence < 0.45

        # 1. Depth Evaluation
        depth_rule = self.rules["SQUAT_DEPTH"]
        if low_confidence:
            findings.append({
                "rule_id": depth_rule["id"],
                "name": depth_rule["name"],
                "status": "CANNOT_ASSESS",
                "citation": depth_rule["citation"],
                "observed": "Low tracking confidence",
                "explanation": "Landmark occlusion or low lighting prevented reliable hip crease / patella identification at bottom position.",
                "actionable_feedback": "Ensure full side-view framing with clear lighting and unobscured hips/knees."
            })
        else:
            depth_delta = telemetry.get("depth_delta_px", 0.0) # > 0 means hip crease is below patella top
            is_deep = telemetry.get("is_deep", depth_delta > 0)
            status = "MEETS_STANDARD" if is_deep else "DOES_NOT_MEET_STANDARD"
            obs_text = f"Hip crease dropped {abs(depth_delta):.1f}px below patella top" if is_deep else f"Hip crease remained {abs(depth_delta):.1f}px above patella top"
            findings.append({
                "rule_id": depth_rule["id"],
                "name": depth_rule["name"],
                "status": status,
                "citation": depth_rule["citation"],
                "observed": obs_text,
                "explanation": depth_rule["pass_message"] if is_deep else depth_rule["fail_message"],
                "actionable_feedback": "Maintain full depth!" if is_deep else "Commit to getting hip crease lower than the top of the patella. Shove knees outward to open hips."
            })

        # 2. Bar Path / Midfoot Alignment
        bar_rule = self.rules["BAR_PATH_MIDFOOT"]
        bar_dev = telemetry.get("max_bar_dev_px", 0.0)
        normalized_dev = telemetry.get("normalized_bar_dev", 0.05)
        # Threshold: deviation <= 0.10 of foot length / leg scale
        bar_pass = normalized_dev <= 0.10
        if low_confidence or not telemetry.get("bar_tracked", True):
            findings.append({
                "rule_id": bar_rule["id"],
                "name": bar_rule["name"],
                "status": "CANNOT_ASSESS",
                "citation": bar_rule["citation"],
                "observed": "Barbell tracking uncertain",
                "explanation": "Barbell plates were partially occluded or indistinguishable from background.",
                "actionable_feedback": "Position camera perpendicular to lifter with visible bar ends."
            })
        else:
            status = "MEETS_STANDARD" if bar_pass else "DOES_NOT_MEET_STANDARD"
            obs_text = f"Max bar path offset: {bar_dev:.1f}px ({normalized_dev*100:.1f}% deviation from midfoot line)"
            findings.append({
                "rule_id": bar_rule["id"],
                "name": bar_rule["name"],
                "status": status,
                "citation": bar_rule["citation"],
                "observed": obs_text,
                "explanation": bar_rule["pass_message"] if bar_pass else bar_rule["fail_message"],
                "actionable_feedback": "Keep bar balanced over midfoot!" if bar_pass else "Sit back and balance load over the middle of foot. Avoid leaning forward onto toes."
            })

        # 3. Back Angle & Rigidity
        back_rule = self.rules["BACK_ANGLE"]
        back_angle = telemetry.get("bottom_back_angle", 45.0)
        ascent_change = telemetry.get("ascent_back_angle_change", 5.0)
        # Low bar squat optimal back angle ~35-60 deg, and ascent change <= 15 deg
        back_pass = (30.0 <= back_angle <= 65.0) and (ascent_change <= 15.0)
        status = "MEETS_STANDARD" if back_pass else "DOES_NOT_MEET_STANDARD"
        obs_text = f"Bottom back angle: {back_angle:.1f}°, ascent inclination shift: {ascent_change:.1f}°"
        findings.append({
            "rule_id": back_rule["id"],
            "name": back_rule["name"],
            "status": status,
            "citation": back_rule["citation"],
            "observed": obs_text,
            "explanation": back_rule["pass_message"] if back_pass else back_rule["fail_message"],
            "actionable_feedback": "Great spinal extension!" if back_pass else "Keep chest lifted and lumbar spine extended. Do not allow hips to shoot up before chest."
        })

        # 4. Knee Position & Tracking
        knee_rule = self.rules["KNEE_TRACKING"]
        knee_fwd = telemetry.get("knee_forward_over_toe_px", 0.0)
        knee_pass = knee_fwd <= 30.0 # Controlled forward travel
        status = "MEETS_STANDARD" if knee_pass else "DOES_NOT_MEET_STANDARD"
        obs_text = f"Knee forward displacement over toe: {knee_fwd:.1f}px"
        findings.append({
            "rule_id": knee_rule["id"],
            "name": knee_rule["name"],
            "status": status,
            "citation": knee_rule["citation"],
            "observed": obs_text,
            "explanation": knee_rule["pass_message"] if knee_pass else knee_rule["fail_message"],
            "actionable_feedback": "Knees stayed properly anchored." if knee_pass else "Shove knees out early in descent and set them in place. Prevent late forward sliding at bottom."
        })

        # 5. Eye Gaze Direction
        gaze_rule = self.rules["EYE_GAZE"]
        gaze_angle = telemetry.get("gaze_angle_deg")
        if gaze_angle is None:
            findings.append({
                "rule_id": gaze_rule["id"],
                "name": gaze_rule["name"],
                "status": "CANNOT_ASSESS",
                "citation": gaze_rule["citation"],
                "observed": "Eye direction is not available from pose landmarks",
                "explanation": "The pose model provides head landmarks but not iris direction, so eye gaze cannot be measured reliably.",
                "actionable_feedback": "Check that your eyes remain fixed on the floor 4–5 feet ahead."
            })
        else:
            gaze_pass = gaze_angle <= 5.0
            status = "MEETS_STANDARD" if gaze_pass else "DOES_NOT_MEET_STANDARD"
            obs_text = f"Head gaze angle: {gaze_angle:.1f}° relative to horizon"
            findings.append({
                "rule_id": gaze_rule["id"],
                "name": gaze_rule["name"],
                "status": status,
                "citation": gaze_rule["citation"],
                "observed": obs_text,
                "explanation": gaze_rule["pass_message"] if gaze_pass else gaze_rule["fail_message"],
                "actionable_feedback": "Good eyes-down gaze!" if gaze_pass else "Fix your eyes on a spot on the floor 4-5 feet ahead. Do not look up at ceiling."
            })

        # 6. Hip Drive
        hip_rule = self.rules["HIP_DRIVE"]
        hip_drive_ok = telemetry.get("hip_drive_initiated")
        if hip_drive_ok is None:
            findings.append({
                "rule_id": hip_rule["id"],
                "name": hip_rule["name"],
                "status": "CANNOT_ASSESS",
                "citation": hip_rule["citation"],
                "observed": "Hip-drive initiation was not measured",
                "explanation": "The current video telemetry does not reliably distinguish hip-drive initiation from a coordinated ascent.",
                "actionable_feedback": "Drive your hips straight up out of the bottom position."
            })
        else:
            status = "MEETS_STANDARD" if hip_drive_ok else "DOES_NOT_MEET_STANDARD"
            findings.append({
                "rule_id": hip_rule["id"],
                "name": hip_rule["name"],
                "status": status,
                "citation": hip_rule["citation"],
                "observed": "Hips initiated vertical ascent" if hip_drive_ok else "Hips stalled or shifted horizontally on ascent",
                "explanation": hip_rule["pass_message"] if hip_drive_ok else hip_rule["fail_message"],
                "actionable_feedback": "Strong hip drive!" if hip_drive_ok else "Think of a chain pulling your hips straight up out of the bottom position."
            })

        # 7. Unassessable Stance & Grip Rule
        sg_rule = self.rules["STANCE_AND_GRIP"]
        findings.append({
            "rule_id": sg_rule["id"],
            "name": sg_rule["name"],
            "status": "CANNOT_ASSESS",
            "citation": sg_rule["citation"],
            "observed": "Side-view camera view active",
            "explanation": "Stance width (heels shoulder-width apart, 30° toe flare) and thumb-over-bar grip require front/rear camera perspective.",
            "actionable_feedback": "Verify stance width and hand grip manually prior to un-racking."
        })

        return findings
