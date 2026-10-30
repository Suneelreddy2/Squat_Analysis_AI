import cv2
import numpy as np
from vision.pose_detector import PoseDetector
from vision.rep_detector import RepetitionDetector
from vision.biomechanics import BiomechanicsCalculator

cap = cv2.VideoCapture("samples/common_sample.mp4")
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
detector = PoseDetector()

telemetry = []
idx = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    lm, conf = detector.process_frame(frame)
    if lm:
        t = BiomechanicsCalculator.compute_frame_telemetry(lm, (0.0, 0.0))
        t["frame_idx"] = idx
        t["pose_confidence"] = conf
        telemetry.append(t)
    else:
        telemetry.append({"valid": False, "frame_idx": idx, "pose_confidence": 0.0})
    idx += 1

cap.release()
detector.close()

rep_det = RepetitionDetector(fps=fps)
reps = rep_det.detect_repetitions(telemetry)
print("Detected reps:", len(reps))
for r in reps:
    print("Rep:", r)
