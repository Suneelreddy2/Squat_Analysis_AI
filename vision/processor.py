import os
import shutil
import math
import cv2
import numpy as np
from typing import Dict, Any, List, Optional, Callable

from vision.pose_detector import PoseDetector
from vision.barbell_tracker import BarbellTracker
from vision.rep_detector import RepetitionDetector
from vision.biomechanics import BiomechanicsCalculator
from vision.video_annotator import VideoAnnotator
from vision.video_transcoder import transcode_to_h264
from squat_skill.skill_engine import SquatSkillEngine

class SquatVideoProcessor:
    def __init__(self, model_complexity: int = 1, rules_path: str = None):
        self.model_complexity = model_complexity
        self.skill_engine = SquatSkillEngine(rules_path=rules_path)

    def process_video(
        self,
        video_path: str,
        output_dir: str = "output",
        static_dir: str = "static",
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> Dict[str, Any]:
        """
        Processes a squat video end-to-end:
        - MediaPipe Pose landmark extraction
        - Barbell tracking & plumb line alignment
        - Repetition phase segmentation
        - Skill engine evaluation against Starting Strength standards with exact frame & timestamp evidence
        - Pure-Python H.264 MP4 transcoding via PyAV
        - Staging into static/ for high-performance HTTP 206 streaming playback
        """
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(static_dir, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open input video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            fps = 30.0

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        pose_detector = PoseDetector(model_complexity=self.model_complexity)
        bar_tracker = BarbellTracker()

        frames_telemetry = []
        raw_frames = []
        landmarks_history = []
        bar_history = []

        if progress_callback:
            progress_callback(0.05, "Extracting landmarks & tracking barbell…")

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            landmarks, conf = pose_detector.process_frame(frame)
            bar_pos, bar_conf = bar_tracker.track(frame, landmarks)

            frame_telemetry = BiomechanicsCalculator.compute_frame_telemetry(landmarks, bar_pos)
            frame_telemetry["frame_idx"] = frame_idx
            frame_telemetry["timestamp"] = frame_idx / fps
            frame_telemetry["pose_confidence"] = conf
            frame_telemetry["bar_tracked"] = bar_conf > 0.4

            frames_telemetry.append(frame_telemetry)
            raw_frames.append(frame)
            landmarks_history.append(landmarks)
            bar_history.append(bar_pos)

            frame_idx += 1
            if progress_callback and frame_idx % 15 == 0 and total_frames > 0:
                prog = 0.05 + 0.50 * (frame_idx / total_frames)
                progress_callback(prog, f"Analyzing frame {frame_idx}/{total_frames}…")

        cap.release()
        pose_detector.close()

        if not raw_frames:
            raise ValueError(f"Input video contains no readable frames: {video_path}")

        if progress_callback:
            progress_callback(0.58, "Detecting repetition phases & bottom positions…")

        # Segment repetitions
        rep_detector = RepetitionDetector(fps=fps)
        repetitions = rep_detector.detect_repetitions(frames_telemetry)

        # Evaluate each repetition against SquatSkill rules
        evaluated_reps = []
        for rep in repetitions:
            start_f = rep["start_frame"]
            bottom_f = rep["bottom_frame"]
            end_f = rep["end_frame"]

            bottom_telemetry = frames_telemetry[bottom_f] if bottom_f < len(frames_telemetry) else {}
            cm_per_px = bottom_telemetry.get("cm_per_px", 0.16)
            foot_len = bottom_telemetry.get("foot_length_px", 160.0)

            # Compute rep aggregate telemetry
            rep_frames = frames_telemetry[start_f:end_f+1]
            valid_rep_frames = [f for f in rep_frames if f.get("valid", False)]
            tracked_rep_frames = [f for f in valid_rep_frames if f.get("bar_tracked", False)]

            # Peak barbell deviation frame
            if tracked_rep_frames:
                peak_bar_frame_data = max(tracked_rep_frames, key=lambda f: abs(f.get("bar_dev_px", 0)))
                max_bar_dev = peak_bar_frame_data.get("bar_dev_px", 0.0)
                max_norm_bar_dev = peak_bar_frame_data.get("norm_bar_dev", 0.0)
                max_bar_f = peak_bar_frame_data.get("frame_idx", bottom_f)
            else:
                max_bar_dev = 0.0
                max_norm_bar_dev = 0.0
                max_bar_f = bottom_f

            # Peak knee forward travel frame
            if valid_rep_frames:
                peak_knee_frame_data = max(valid_rep_frames, key=lambda f: f.get("knee_forward_px", 0.0))
                max_knee_fwd_px = peak_knee_frame_data.get("knee_forward_px", 0.0)
                max_knee_fwd_norm = peak_knee_frame_data.get("knee_forward_norm", max_knee_fwd_px / max(foot_len, 1.0))
                max_knee_f = peak_knee_frame_data.get("frame_idx", bottom_f)
            else:
                max_knee_fwd_px = 0.0
                max_knee_fwd_norm = 0.0
                max_knee_f = bottom_f

            avg_conf = float(np.mean([f.get("pose_confidence", 1.0) for f in valid_rep_frames])) if valid_rep_frames else 0.0

            # Ascent back angle stability check
            ascent_frames = frames_telemetry[bottom_f:end_f+1]
            ascent_back_angles = [f.get("back_angle", 45) for f in ascent_frames if f.get("valid", False)]
            min_ascent_back = min(ascent_back_angles, default=45.0)
            bottom_back = bottom_telemetry.get("back_angle", 45.0)
            ascent_change = abs(bottom_back - min_ascent_back)

            # Hip drive heuristic: evaluate vertical hip ascension immediately out of the hole
            # Hips should drive upwards without premature spinal extension or forward stall
            init_ascent_window = frames_telemetry[bottom_f: min(bottom_f + max(int(fps * 0.4), 4), end_f + 1)]
            valid_init_ascent = [f for f in init_ascent_window if f.get("valid", False)]
            hip_drive_status = None
            if len(valid_init_ascent) >= 3 and avg_conf >= 0.5:
                # In image coords, hip rising means hip_y decreases:
                bottom_hip_y = bottom_telemetry.get("hip_y", 0)
                ascent_hip_ys = [f.get("hip_y", bottom_hip_y) for f in valid_init_ascent]
                min_ascent_hip_y = min(ascent_hip_ys)
                hip_rise_px = bottom_hip_y - min_ascent_hip_y
                # If hips move upward promptly with stable torso, hip drive initiated
                if hip_rise_px >= 8.0 and ascent_change <= 12.0:
                    hip_drive_status = True
                elif hip_rise_px < 3.0 and ascent_change > 15.0:
                    hip_drive_status = False

            rep_summary_telemetry = {
                "pose_confidence": avg_conf,
                "fps": fps,
                "cm_per_px": cm_per_px,
                "foot_length_px": foot_len,
                # Depth
                "depth_delta_px": bottom_telemetry.get("depth_delta_px", 0.0),
                "depth_delta_cm": bottom_telemetry.get("depth_delta_cm", 0.0),
                "depth_frame": bottom_f,
                "depth_timestamp": bottom_f / fps,
                "is_deep": bottom_telemetry.get("is_deep", False),
                # Bar Path
                "max_bar_dev_px": max_bar_dev,
                "max_bar_dev_cm": max_bar_dev * cm_per_px,
                "normalized_bar_dev": max_norm_bar_dev,
                "bar_frame": max_bar_f,
                "bar_timestamp": max_bar_f / fps,
                "bar_tracked": bool(tracked_rep_frames) and (len(tracked_rep_frames) >= 0.70 * len(valid_rep_frames)),
                # Back Angle
                "bottom_back_angle": bottom_back,
                "ascent_back_angle_change": ascent_change,
                "back_frame": bottom_f,
                "back_timestamp": bottom_f / fps,
                # Knee Tracking
                "knee_forward_over_toe_px": max_knee_fwd_px,
                "knee_forward_over_toe_cm": max_knee_fwd_px * cm_per_px,
                "knee_forward_norm": max_knee_fwd_norm,
                "knee_frame": max_knee_f,
                "knee_timestamp": max_knee_f / fps,
                # Cervical / Eye Gaze
                "cervical_tilt_deg": bottom_telemetry.get("cervical_tilt", -12.0),
                "gaze_angle_deg": None,
                "gaze_frame": bottom_f,
                "gaze_timestamp": bottom_f / fps,
                # Hip Drive
                "hip_drive_initiated": hip_drive_status,
                "hip_drive_frame": bottom_f,
                "hip_drive_timestamp": bottom_f / fps,
                # Stance and grip
                "setup_frame": start_f,
                "setup_timestamp": start_f / fps,
            }

            findings = self.skill_engine.evaluate_repetition(rep_summary_telemetry)

            rep_eval = {
                **rep,
                "bottom_timestamp": bottom_f / fps,
                "telemetry": rep_summary_telemetry,
                "findings": findings
            }
            evaluated_reps.append(rep_eval)

        if progress_callback:
            progress_callback(0.72, "Rendering annotated video with biomechanical HUD…")

        # ── Write annotated video (raw mp4v container first) ───────────────────
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        raw_annotated_path = os.path.join(output_dir, f"raw_ann_{base_name}.mp4")
        final_web_name = f"web_annotated_{base_name}.mp4"
        final_web_path = os.path.join(output_dir, final_web_name)

        fourcc_mp4v = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(raw_annotated_path, fourcc_mp4v, fps, (width, height))
        if not out_writer.isOpened():
            out_writer.release()
            raise RuntimeError(f"Could not create annotated video writer: {raw_annotated_path}")

        annotator = VideoAnnotator()

        for idx, frame in enumerate(raw_frames):
            lm = landmarks_history[idx]
            tel = frames_telemetry[idx]

            cur_rep = None
            cur_phase = "SETUP / STANDING"
            for rep in evaluated_reps:
                if rep["start_frame"] <= idx <= rep["end_frame"]:
                    cur_rep = rep
                    if abs(idx - rep["bottom_frame"]) <= 2:
                        cur_phase = "BOTTOM"
                    elif idx < rep["bottom_frame"]:
                        cur_phase = "DESCENT"
                    else:
                        cur_phase = "ASCENT"
                    break

            annotated_frame = annotator.annotate_frame(
                frame=frame,
                landmarks=lm,
                telemetry=tel,
                bar_history=bar_history[:idx+1],
                current_rep=cur_rep,
                rep_phase=cur_phase
            )
            out_writer.write(annotated_frame)

        out_writer.release()

        if progress_callback:
            progress_callback(0.92, "Transcoding to browser-native H.264 video…")

        # ── Transcode to browser-standard H.264 (libx264, yuv420p) ────────────
        transcode_ok = transcode_to_h264(raw_annotated_path, final_web_path)
        if transcode_ok and os.path.isfile(final_web_path) and os.path.getsize(final_web_path) > 0:
            active_annotated_path = final_web_path
            try:
                os.remove(raw_annotated_path)
            except OSError:
                pass
        else:
            active_annotated_path = raw_annotated_path

        # ── Stage into static/ for instant HTTP 206 partial streaming ──────────
        static_annotated_name = f"ann_{base_name}.mp4"
        static_raw_name = f"raw_{base_name}.mp4"
        static_annotated_path = os.path.join(static_dir, static_annotated_name)
        static_raw_path = os.path.join(static_dir, static_raw_name)

        try:
            shutil.copy2(active_annotated_path, static_annotated_path)
        except Exception:
            static_annotated_path = active_annotated_path

        try:
            shutil.copy2(video_path, static_raw_path)
        except Exception:
            static_raw_path = video_path

        if progress_callback:
            progress_callback(1.0, "Analysis complete!")

        return {
            "video_path": video_path,
            "annotated_video_path": active_annotated_path,
            "static_annotated_url": f"app/static/{static_annotated_name}",
            "static_raw_url": f"app/static/{static_raw_name}",
            "fps": fps,
            "total_frames": frame_idx,
            "duration_sec": frame_idx / fps,
            "repetitions": evaluated_reps,
            "frames_telemetry": frames_telemetry
        }
