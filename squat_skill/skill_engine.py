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
        Every finding includes supporting frame, timestamp, observed measurement,
        authoritative citations, and actionable feedback.
        """
        findings = []
        confidence = telemetry.get("pose_confidence", 1.0)
        low_confidence = confidence < 0.45

        depth_f = telemetry.get("depth_frame", 0)
        depth_t = telemetry.get("depth_timestamp", 0.0)

        # 1. Depth Evaluation (Hip crease vs Top of patella)
        depth_rule = self.rules["SQUAT_DEPTH"]
        if low_confidence:
            findings.append({
                "rule_id": depth_rule["id"],
                "name": depth_rule["name"],
                "status": "CANNOT_ASSESS",
                "frame": depth_f,
                "timestamp": depth_t,
                "citation": depth_rule["citation"],
                "observed": "Low tracking confidence",
                "explanation": "Landmark occlusion or low lighting prevented reliable hip crease / patella identification at bottom position.",
                "actionable_feedback": "Ensure full side-view framing with clear lighting and unobscured hips/knees."
            })
        else:
            depth_delta_px = telemetry.get("depth_delta_px", 0.0)  # > 0 means hip crease is below patella top
            depth_delta_cm = telemetry.get("depth_delta_cm", depth_delta_px * 0.16)
            is_deep = telemetry.get("is_deep", depth_delta_px > 0)
            status = "MEETS_STANDARD" if is_deep else "DOES_NOT_MEET_STANDARD"
            
            if is_deep:
                obs_text = f"Hip crease dropped {abs(depth_delta_px):.1f}px below patella top (~{abs(depth_delta_cm):.1f} cm below parallel)"
            else:
                obs_text = f"Hip crease remained {abs(depth_delta_px):.1f}px above patella top (~{abs(depth_delta_cm):.1f} cm above parallel)"

            findings.append({
                "rule_id": depth_rule["id"],
                "name": depth_rule["name"],
                "status": status,
                "frame": depth_f,
                "timestamp": depth_t,
                "citation": depth_rule["citation"],
                "observed": obs_text,
                "explanation": depth_rule["pass_message"] if is_deep else depth_rule["fail_message"],
                "actionable_feedback": "Maintain full depth!" if is_deep else "Commit to getting hip crease lower than the top of the patella. Shove knees outward to open hips."
            })

        # 2. Bar Path / Midfoot Alignment
        bar_rule = self.rules["BAR_PATH_MIDFOOT"]
        bar_dev_px = telemetry.get("max_bar_dev_px", 0.0)
        bar_dev_cm = telemetry.get("max_bar_dev_cm", bar_dev_px * 0.16)
        normalized_dev = telemetry.get("normalized_bar_dev", 0.05)
        bar_f = telemetry.get("bar_frame", depth_f)
        bar_t = telemetry.get("bar_timestamp", depth_t)

        # Scale-invariant threshold: deviation <= 0.12 (12%) of foot length from midfoot plumb line
        bar_pass = normalized_dev <= 0.12
        if low_confidence or not telemetry.get("bar_tracked", True):
            findings.append({
                "rule_id": bar_rule["id"],
                "name": bar_rule["name"],
                "status": "CANNOT_ASSESS",
                "frame": bar_f,
                "timestamp": bar_t,
                "citation": bar_rule["citation"],
                "observed": "Barbell tracking uncertain or occluded",
                "explanation": "Barbell plates were partially occluded or indistinguishable from background across repetition frames.",
                "actionable_feedback": "Position camera perpendicular to lifter with unobstructed bar ends."
            })
        else:
            status = "MEETS_STANDARD" if bar_pass else "DOES_NOT_MEET_STANDARD"
            obs_text = f"Max bar offset: {abs(bar_dev_px):.1f}px ({normalized_dev*100:.1f}% of foot length ~ {abs(bar_dev_cm):.1f} cm from midfoot)"
            findings.append({
                "rule_id": bar_rule["id"],
                "name": bar_rule["name"],
                "status": status,
                "frame": bar_f,
                "timestamp": bar_t,
                "citation": bar_rule["citation"],
                "observed": obs_text,
                "explanation": bar_rule["pass_message"] if bar_pass else bar_rule["fail_message"],
                "actionable_feedback": "Bar path stayed balanced over midfoot!" if bar_pass else "Sit back and balance load over the middle of foot. Avoid drifting forward onto toes."
            })

        # 3. Back Angle & Rigidity
        back_rule = self.rules["BACK_ANGLE"]
        back_angle = telemetry.get("bottom_back_angle", 45.0)
        ascent_change = telemetry.get("ascent_back_angle_change", 5.0)
        back_f = telemetry.get("back_frame", depth_f)
        back_t = telemetry.get("back_timestamp", depth_t)

        # Low bar squat optimal back angle ~35-65 deg, and ascent change <= 12 deg
        back_pass = (32.0 <= back_angle <= 65.0) and (ascent_change <= 12.0)
        status = "MEETS_STANDARD" if back_pass else "DOES_NOT_MEET_STANDARD"
        obs_text = f"Bottom back angle: {back_angle:.1f}°, ascent inclination shift: {ascent_change:.1f}°"
        findings.append({
            "rule_id": back_rule["id"],
            "name": back_rule["name"],
            "status": status,
            "frame": back_f,
            "timestamp": back_t,
            "citation": back_rule["citation"],
            "observed": obs_text,
            "explanation": back_rule["pass_message"] if back_pass else back_rule["fail_message"],
            "actionable_feedback": "Great spinal extension and torso rigidity!" if back_pass else "Keep chest lifted and lumbar spine extended. Do not allow hips to shoot up before chest ('good morning')."
        })

        # 4. Knee Position & Tracking (Normalized to foot length)
        knee_rule = self.rules["KNEE_TRACKING"]
        knee_fwd_px = telemetry.get("knee_forward_over_toe_px", 0.0)
        knee_fwd_cm = telemetry.get("knee_forward_over_toe_cm", knee_fwd_px * 0.16)
        knee_fwd_norm = telemetry.get("knee_forward_norm", 0.15)
        knee_f = telemetry.get("knee_frame", depth_f)
        knee_t = telemetry.get("knee_timestamp", depth_t)

        # In Starting Strength, knees naturally track slightly forward of toes (10-25% of foot length)
        # Excessive knee forward slide is when knees travel > 25% of foot length beyond toe
        knee_pass = knee_fwd_norm <= 0.25
        status = "MEETS_STANDARD" if knee_pass else "DOES_NOT_MEET_STANDARD"
        obs_text = f"Knee forward displacement over toe: {knee_fwd_px:.1f}px ({knee_fwd_norm*100:.1f}% of foot length ~ {knee_fwd_cm:.1f} cm)"
        findings.append({
            "rule_id": knee_rule["id"],
            "name": knee_rule["name"],
            "status": status,
            "frame": knee_f,
            "timestamp": knee_t,
            "citation": knee_rule["citation"],
            "observed": obs_text,
            "explanation": knee_rule["pass_message"] if knee_pass else knee_rule["fail_message"],
            "actionable_feedback": "Knees stayed properly anchored over toes." if knee_pass else "Shove knees out early in descent and set them in place. Prevent late forward sliding at bottom."
        })

        # 5. Eye Gaze & Cervical Spine Alignment
        gaze_rule = self.rules["EYE_GAZE"]
        cervical_tilt = telemetry.get("cervical_tilt_deg")
        gaze_f = telemetry.get("gaze_frame", depth_f)
        gaze_t = telemetry.get("gaze_timestamp", depth_t)

        # Pose landmarks track head position (ear-to-eye angle), not direct eye iris orientation
        if cervical_tilt is not None:
            # Downward head tilt (-45° to -5°) aligns cervical spine with floor gaze
            cervical_ok = cervical_tilt <= 5.0
            findings.append({
                "rule_id": gaze_rule["id"],
                "name": gaze_rule["name"],
                "status": "MEETS_STANDARD" if cervical_ok else "DOES_NOT_MEET_STANDARD",
                "frame": gaze_f,
                "timestamp": gaze_t,
                "citation": gaze_rule["citation"],
                "observed": f"Cervical spine / head tilt: {cervical_tilt:.1f}° relative to horizon (Iris gaze unmeasured)",
                "explanation": "Head tilt indicates neutral downward gaze. Note: Iris gaze is unmeasured by pose landmarks." if cervical_ok else "Head was tilted upward, compromising neutral cervical spine alignment.",
                "actionable_feedback": "Neutral cervical spine maintained." if cervical_ok else "Fix your eyes on a spot on the floor 4–5 feet ahead with chin tucked down."
            })
        else:
            findings.append({
                "rule_id": gaze_rule["id"],
                "name": gaze_rule["name"],
                "status": "CANNOT_ASSESS",
                "frame": gaze_f,
                "timestamp": gaze_t,
                "citation": gaze_rule["citation"],
                "observed": "Eye direction not available from 2D pose landmarks",
                "explanation": "The pose model provides head landmarks but not iris direction, so eye gaze cannot be measured directly.",
                "actionable_feedback": "Ensure your eyes remain fixed on the floor 4–5 feet ahead throughout the rep."
            })

        # 6. Hip Drive Initiation out of Bottom
        hip_rule = self.rules["HIP_DRIVE"]
        hip_drive_ok = telemetry.get("hip_drive_initiated")
        hip_f = telemetry.get("hip_drive_frame", depth_f)
        hip_t = telemetry.get("hip_drive_timestamp", depth_t)

        if hip_drive_ok is True:
            findings.append({
                "rule_id": hip_rule["id"],
                "name": hip_rule["name"],
                "status": "MEETS_STANDARD",
                "frame": hip_f,
                "timestamp": hip_t,
                "citation": hip_rule["citation"],
                "observed": "Hips drove upward immediately out of bottom turnaround with rigid torso",
                "explanation": hip_rule["pass_message"],
                "actionable_feedback": "Strong hip drive recruitment out of the bottom!"
            })
        elif hip_drive_ok is False:
            findings.append({
                "rule_id": hip_rule["id"],
                "name": hip_rule["name"],
                "status": "DOES_NOT_MEET_STANDARD",
                "frame": hip_f,
                "timestamp": hip_t,
                "citation": hip_rule["citation"],
                "observed": "Hips stalled or shifted forward upon starting ascent rather than driving straight up",
                "explanation": hip_rule["fail_message"],
                "actionable_feedback": "Think of a chain pulling your sacrum straight up out of the bottom position."
            })
        else:
            findings.append({
                "rule_id": hip_rule["id"],
                "name": hip_rule["name"],
                "status": "CANNOT_ASSESS",
                "frame": hip_f,
                "timestamp": hip_t,
                "citation": hip_rule["citation"],
                "observed": "Ascent acceleration signal below discrimination threshold",
                "explanation": "Video frame rate or landmark noise did not permit isolated measurement of hip drive initiation timing.",
                "actionable_feedback": "Drive your hips straight up out of the bottom position using the stretch reflex."
            })

        # 7. Unassessable Stance & Grip Rule (Front/Rear Perspective required)
        sg_rule = self.rules["STANCE_AND_GRIP"]
        setup_f = telemetry.get("setup_frame", 0)
        setup_t = telemetry.get("setup_timestamp", 0.0)
        findings.append({
            "rule_id": sg_rule["id"],
            "name": sg_rule["name"],
            "status": "CANNOT_ASSESS",
            "frame": setup_f,
            "timestamp": setup_t,
            "citation": sg_rule["citation"],
            "observed": "Side-view camera perspective active",
            "explanation": "Stance width (heels shoulder-width apart, 30° toe flare) and thumb-over-bar grip require front/rear camera perspective.",
            "actionable_feedback": "Verify stance width (shoulder-width heels, 30° toe flare) and grip manually prior to un-racking."
        })

        return findings
