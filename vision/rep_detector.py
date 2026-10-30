import numpy as np
from typing import List, Dict, Any

class RepetitionDetector:
    def __init__(self, fps: float = 30.0):
        self.fps = fps

    def detect_repetitions(self, frames_telemetry: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Processes frame-by-frame telemetry to detect squat repetitions.
        Returns a list of repetition dictionary metadata.
        """
        if not frames_telemetry:
            return []

        # Process each contiguous pose-valid segment independently. Compressing
        # valid samples across an occlusion makes frames on either side of a
        # tracking gap appear adjacent and can invent a turnaround.
        if any(not data.get("valid", False) or "hip_y" not in data for data in frames_telemetry):
            segments = []
            segment = []
            segment_start = 0
            for frame_idx, data in enumerate(frames_telemetry):
                if data.get("valid", False) and "hip_y" in data:
                    if not segment:
                        segment_start = frame_idx
                    segment.append(data)
                elif segment:
                    segments.append((segment_start, segment))
                    segment = []
            if segment:
                segments.append((segment_start, segment))

            repetitions = []
            for offset, segment in segments:
                if len(segment) < 10:
                    continue
                for rep in self.detect_repetitions(segment):
                    rep["start_frame"] += offset
                    rep["bottom_frame"] += offset
                    rep["end_frame"] += offset
                    rep["rep_number"] = len(repetitions) + 1
                    repetitions.append(rep)
            return repetitions

        hip_y_series = []
        knee_angle_series = []
        valid_indices = []
        for i, data in enumerate(frames_telemetry):
            if data.get("valid", False) and "hip_y" in data:
                hip_y_series.append(data["hip_y"])
                knee_angle_series.append(data.get("knee_angle", 170.0))
                valid_indices.append(i)

        if len(hip_y_series) < 10:
            return []

        # Pad with edge values so boundaries do not artificially drop towards 0
        padded_y = np.pad(hip_y_series, (3, 3), mode='edge')
        smoothed_y = np.convolve(padded_y, np.ones(7)/7, mode='valid')

        y_min = float(np.min(smoothed_y))
        y_max = float(np.max(smoothed_y))
        y_range = y_max - y_min

        # Threshold for meaningful squat movement: at least 30px vertical change
        if y_range < 30.0:
            return []

        # Find local peaks of hip Y (bottom turnaround point of squat)
        # In image coordinates, Y increases downward, so bottom position = local maximum of hip Y
        # Additionally, squat bottom requires significant knee flexion (knee_angle < 125°)
        bottom_candidates = []
        for i in range(2, len(smoothed_y) - 2):
            val = smoothed_y[i]
            is_local_max = (
                val >= smoothed_y[i-1] and val >= smoothed_y[i-2] and
                val >= smoothed_y[i+1] and val >= smoothed_y[i+2]
            )
            has_descent = (val - y_min) > 0.40 * y_range
            is_knee_flexed = knee_angle_series[i] < 125.0

            if is_local_max and has_descent and is_knee_flexed:
                bottom_candidates.append(i)

        if not bottom_candidates:
            # Fallback: check absolute max if it satisfies flexion criteria
            max_idx = int(np.argmax(smoothed_y))
            if (smoothed_y[max_idx] - y_min) > 0.40 * y_range and knee_angle_series[max_idx] < 125.0:
                bottom_candidates = [max_idx]

        # Merge close bottom candidates (within 0.75 seconds)
        min_dist = max(int(self.fps * 0.75), 10)
        merged_bottoms = []
        for b in bottom_candidates:
            if not merged_bottoms or (b - merged_bottoms[-1]) > min_dist:
                merged_bottoms.append(b)
            else:
                if smoothed_y[b] > smoothed_y[merged_bottoms[-1]]:
                    merged_bottoms[-1] = b

        repetitions = []
        for idx, bottom_idx in enumerate(merged_bottoms):
            # Find start of repetition (descent onset from standing)
            prev_bound = merged_bottoms[idx - 1] if idx > 0 else 0
            start_rel_idx = prev_bound
            for k in range(bottom_idx - 1, prev_bound - 1, -1):
                is_standing_height = (smoothed_y[k] - y_min) <= 0.15 * y_range
                is_leg_extended = knee_angle_series[k] >= 155.0
                if is_standing_height or is_leg_extended:
                    start_rel_idx = k
                    break
                if k > prev_bound and smoothed_y[k] <= smoothed_y[k-1] and smoothed_y[k] <= smoothed_y[k+1]:
                    start_rel_idx = k
                    break

            # Find end of repetition (ascent completion back to standing)
            next_bound = merged_bottoms[idx + 1] if idx < len(merged_bottoms) - 1 else len(smoothed_y) - 1
            end_rel_idx = next_bound
            for k in range(bottom_idx + 1, next_bound + 1):
                is_standing_height = (smoothed_y[k] - y_min) <= 0.15 * y_range
                is_leg_extended = knee_angle_series[k] >= 155.0
                if is_standing_height or is_leg_extended:
                    end_rel_idx = k
                    break
                if k < next_bound and smoothed_y[k] <= smoothed_y[k-1] and smoothed_y[k] <= smoothed_y[k+1]:
                    end_rel_idx = k
                    break

            start_frame = valid_indices[start_rel_idx]
            bottom_frame = valid_indices[bottom_idx]
            end_frame = valid_indices[end_rel_idx]

            duration = (end_frame - start_frame) / max(self.fps, 1.0)
            
            repetitions.append({
                "rep_number": len(repetitions) + 1,
                "start_frame": start_frame,
                "bottom_frame": bottom_frame,
                "end_frame": end_frame,
                "duration_sec": duration,
                "descent_duration_sec": (bottom_frame - start_frame) / max(self.fps, 1.0),
                "ascent_duration_sec": (end_frame - bottom_frame) / max(self.fps, 1.0)
            })

        return repetitions
