"""
Live Rep Counter — stateful processor for real-time squat detection via webcam.

Used with streamlit-webrtc's VideoProcessorBase pattern.
Each incoming frame is processed server-side (MediaPipe Pose + rep logic),
annotated with overlays, and sent back to the browser.
"""
import math
import time
import threading
import os
import subprocess
import numpy as np
import cv2
import mediapipe as mp
from collections import deque

from vision.biomechanics import BiomechanicsCalculator
from vision.video_annotator import VideoAnnotator
from vision.video_transcoder import transcode_to_h264

# Landmark indices — mediapipe 0.10.x keeps solutions under mp.solutions
try:
    _mp_pose = mp.solutions.pose
except AttributeError:
    # Fallback for environments where solutions is not directly available
    from mediapipe.python.solutions import pose as _mp_pose_mod
    import types
    _mp_pose = types.SimpleNamespace(
        Pose=_mp_pose_mod.Pose,
        PoseLandmark=_mp_pose_mod.PoseLandmark,
    )

class LiveRepCounter:
    """Thread-safe, stateful real-time rep counter."""

    def __init__(self, smoothing_window: int = 9):
        self.lock = threading.Lock()
        self._record_lock = threading.Lock()
        self._recording = False
        self._record_path = None
        self._record_writer = None
        self._record_frame_count = 0
        self._record_pending_frames = []
        self._record_last_timestamp = None

        # MediaPipe Pose (lightweight, low-latency)
        self.pose = _mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.55,
            min_tracking_confidence=0.55,
        )
        self.annotator   = VideoAnnotator()
        self.bio         = BiomechanicsCalculator()

        # Hip Y ring buffer for smoothing + peak detection
        self._hip_buf    = deque(maxlen=smoothing_window)
        self._smooth_buf = deque(maxlen=smoothing_window * 3)  # extended for peak look-around

        # Rep counting state machine
        self._phase      = "STANDING"      # STANDING | DESCENT | BOTTOM | ASCENT
        self._rep_count  = 0
        self._y_baseline = None            # top hip Y (standing)
        self._y_bottom   = None            # deepest hip Y seen this rep
        self._depth_ok   = False           # did this rep pass depth?
        self._last_rep_t = time.time()

        # Rolling bar history for trajectory ribbon
        self._bar_hist   = deque(maxlen=60)

        # Current frame metrics (read by Streamlit UI)
        self._metrics    = {
            "rep_count": 0,
            "phase": "STANDING",
            "knee_angle": 0.0,
            "back_angle": 0.0,
            "depth_ok": False,
            "pose_conf": 0.0,
        }

    # ─── Public API (called from Streamlit UI thread) ────────────────────────
    def get_metrics(self) -> dict:
        with self.lock:
            return dict(self._metrics)

    def reset(self):
        with self.lock:
            self._rep_count  = 0
            self._phase      = "STANDING"
            self._y_baseline = None
            self._y_bottom   = None
            self._depth_ok   = False
            self._hip_buf.clear()
            self._smooth_buf.clear()
            self._bar_hist.clear()
            self._metrics.update({
                "rep_count": 0,
                "phase": "STANDING",
                "knee_angle": 0.0,
                "back_angle": 0.0,
                "depth_ok": False,
                "pose_conf": 0.0,
            })

    def start_recording(self, output_dir: str = "output") -> bool:
        """Begin saving raw camera frames; open the writer on the next frame."""
        os.makedirs(output_dir, exist_ok=True)
        with self._record_lock:
            if self._recording:
                return False
            self._record_path = os.path.join(
                output_dir, f"live_recording_{int(time.time() * 1000)}.mp4"
            )
            self._record_writer = None
            self._record_frame_count = 0
            self._record_pending_frames = []
            self._record_last_timestamp = None
            self._recording = True
            return True

    def stop_recording(self):
        """Stop recording and return a browser-playable recording when possible."""
        with self._record_lock:
            self._recording = False
            if self._record_writer is None and self._record_pending_frames:
                self._open_record_writer(*self._record_pending_frames[0][0].shape[1::-1], fps=30.0)
                self._flush_pending_record_frames()
            if self._record_writer is not None:
                self._record_writer.release()
                self._record_writer = None
            path = self._record_path if self._record_frame_count else None
            self._record_path = None
            self._record_frame_count = 0
            self._record_pending_frames = []
            self._record_last_timestamp = None

        if path:
            return self._make_browser_playable(path)
        return None

    @staticmethod
    def _make_browser_playable(source_path: str) -> str:
        """Convert MP4V webcam output to H.264 for HTML5 video players."""
        output_path = os.path.join(
            os.path.dirname(source_path),
            f"browser_{os.path.basename(source_path)}",
        )
        if transcode_to_h264(source_path, output_path) and os.path.isfile(output_path) and os.path.getsize(output_path) > 0:
            return output_path
        return source_path

    def _open_record_writer(self, width: int, height: int, fps: float):
        size = (width, height)
        self._record_writer = cv2.VideoWriter(
            self._record_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size
        )
        if not self._record_writer.isOpened():
            self._record_writer.release()
            self._record_writer = None
            self._recording = False
            self._record_pending_frames.clear()
            return False
        return True

    def _flush_pending_record_frames(self):
        if self._record_writer is None:
            return
        for pending_frame, _ in self._record_pending_frames:
            self._record_writer.write(pending_frame)
            self._record_frame_count += 1
        self._record_pending_frames.clear()

    def _write_recording_frame(self, frame: np.ndarray, timestamp: float = None):
        with self._record_lock:
            if not self._recording:
                return
            if timestamp is None or not math.isfinite(timestamp):
                timestamp = time.monotonic()
            if self._record_last_timestamp is not None and timestamp <= self._record_last_timestamp:
                timestamp = self._record_last_timestamp + (1.0 / 30.0)
            self._record_last_timestamp = timestamp

            if self._record_writer is None:
                self._record_pending_frames.append((frame.copy(), timestamp))
                first_timestamp = self._record_pending_frames[0][1]
                elapsed = timestamp - first_timestamp
                if len(self._record_pending_frames) >= 12 and elapsed >= 0.5:
                    measured_fps = (len(self._record_pending_frames) - 1) / elapsed
                    measured_fps = min(60.0, max(5.0, measured_fps))
                    height, width = frame.shape[:2]
                    if self._open_record_writer(width, height, measured_fps):
                        self._flush_pending_record_frames()
                return

            self._record_writer.write(frame)
            self._record_frame_count += 1

    # ─── Frame processing (called from WebRTC worker thread) ─────────────────
    def process_frame(self, bgr: np.ndarray, timestamp: float = None) -> np.ndarray:
        h, w = bgr.shape[:2]
        rgb  = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        res  = self.pose.process(rgb)

        if not res.pose_landmarks:
            with self.lock:
                self._metrics["pose_conf"] = 0.0
            out = self._draw_hud(bgr, {})
            self._write_recording_frame(bgr, timestamp)
            return out

        lm_raw = res.pose_landmarks.landmark

        # Pick dominant side
        left_ids  = [11, 13, 15, 23, 25, 27, 29, 31]
        right_ids = [12, 14, 16, 24, 26, 28, 30, 32]
        l_vis = np.mean([lm_raw[i].visibility for i in left_ids])
        r_vis = np.mean([lm_raw[i].visibility for i in right_ids])
        side  = "left" if l_vis >= r_vis else "right"

        def px(idx):
            lm = lm_raw[idx]
            return (lm.x * w, lm.y * h, lm.visibility)

        S = 0 if side == "left" else 1   # offset: left=0, right=1
        landmarks = {
            "shoulder": px(11 + S),
            "elbow":    px(13 + S),
            "wrist":    px(15 + S),
            "hip":      px(23 + S),
            "knee":     px(25 + S),
            "ankle":    px(27 + S),
            "heel":     px(29 + S),
            "toe":      px(31 + S),
            "ear":      px(7  + (1 if side == "left" else -1)),  # 7 left, 8 right
            "eye":      px(2  + (3 if side == "right" else 0)),  # 2 left, 5 right
        }

        conf = float(np.mean([lm[2] for lm in landmarks.values()]))

        # Barbell position: approximate as near-shoulder (low-bar style)
        bar_pos = (landmarks["shoulder"][0], landmarks["shoulder"][1] + 10)
        self._bar_hist.append(bar_pos)

        # Biomechanics
        tel = BiomechanicsCalculator.compute_frame_telemetry(landmarks, bar_pos)

        hip_y = tel.get("hip_y", 0.0)
        self._hip_buf.append(hip_y)
        smooth_y = float(np.mean(self._hip_buf))
        self._smooth_buf.append(smooth_y)

        # Update rep state machine
        phase = self._update_rep_state(smooth_y, tel, conf)

        # Annotate
        out = self.annotator.annotate_frame(
            frame       = bgr,
            landmarks   = landmarks,
            telemetry   = tel,
            bar_history = list(self._bar_hist),
            current_rep = {"rep_number": self._rep_count} if self._rep_count > 0 else None,
            rep_phase   = phase,
        )

        # Overlay large rep counter
        out = self._draw_hud(out, tel, phase, conf)
        self._write_recording_frame(bgr, timestamp)
        return out

    # ─── State machine ────────────────────────────────────────────────────────
    def _update_rep_state(self, smooth_y: float, tel: dict, pose_conf: float) -> str:
        """
        Hip Y increases downward in image space.
        STANDING  → high hip (small Y)
        BOTTOM    → low hip  (large Y)
        """
        is_deep = tel.get("is_deep", False)

        # Calibrate baseline from first few frames
        if self._y_baseline is None or self._phase == "STANDING":
            if len(self._smooth_buf) >= 5:
                self._y_baseline = float(np.min(list(self._smooth_buf)[-10:]))

        if self._y_baseline is None:
            return self._phase

        depth_threshold = self._y_baseline + 40   # must travel at least 40px
        if self._phase == "STANDING":
            if smooth_y > depth_threshold:
                self._phase   = "DESCENT"
                self._y_bottom = smooth_y
                self._depth_ok = False

        elif self._phase == "DESCENT":
            if smooth_y > (self._y_bottom or smooth_y):
                self._y_bottom = smooth_y
            # Detect turnaround (hip starts coming back up)
            if len(self._smooth_buf) >= 3:
                recent = list(self._smooth_buf)[-3:]
                if recent[-1] < recent[-2] - 1.5:
                    self._depth_ok = is_deep
                    self._phase    = "BOTTOM"

        elif self._phase == "BOTTOM":
            # Hold a distinct turnaround phase for at least one processed frame,
            # then move into ascent once upward motion continues.
            if len(self._smooth_buf) >= 3:
                recent = list(self._smooth_buf)[-3:]
                if recent[-1] < recent[-2] - 1.5:
                    self._phase = "ASCENT"

        elif self._phase == "ASCENT":
            # Rep complete when back at ~standing height
            if smooth_y < self._y_baseline + 30:
                self._rep_count += 1
                self._last_rep_t = time.time()
                self._phase      = "STANDING"
                self._y_bottom   = None
                # Refresh baseline
                self._y_baseline = smooth_y

        with self.lock:
            self._metrics.update({
                "rep_count":  self._rep_count,
                "phase":      self._phase,
                "knee_angle": tel.get("knee_angle", 0.0),
                "back_angle": tel.get("back_angle", 0.0),
                "depth_ok":   self._depth_ok,
                "pose_conf":  pose_conf,
            })
        return self._phase

    # ─── HUD overlay ─────────────────────────────────────────────────────────
    def _draw_hud(self, img: np.ndarray, tel: dict, phase: str = "STANDING", conf: float = 0.0) -> np.ndarray:
        h, w = img.shape[:2]
        overlay = img.copy()

        # Top-right rep badge
        badge_w, badge_h = 180, 90
        x1, y1 = w - badge_w - 12, 12
        cv2.rectangle(overlay, (x1, y1), (x1 + badge_w, y1 + badge_h), (15, 20, 35), -1)
        cv2.rectangle(overlay, (x1, y1), (x1 + badge_w, y1 + badge_h), (80, 80, 200), 2)
        cv2.putText(overlay, f"{self._rep_count}", (x1 + 20, y1 + 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.4, (150, 130, 255), 4, cv2.LINE_AA)
        cv2.putText(overlay, "REPS", (x1 + 108, y1 + 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 255), 2, cv2.LINE_AA)

        # Phase badge (bottom-left)
        phase_color = {
            "BOTTOM":   (0, 230, 115),
            "DESCENT":  (0, 200, 255),
            "ASCENT":   (0, 200, 255),
            "STANDING": (160, 160, 160),
        }.get(phase, (160, 160, 160))
        cv2.rectangle(overlay, (12, h - 52), (12 + 200, h - 14), (15, 20, 35), -1)
        cv2.putText(overlay, f"PHASE: {phase}", (20, h - 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, phase_color, 2, cv2.LINE_AA)

        # Blend
        return cv2.addWeighted(overlay, 0.88, img, 0.12, 0)
