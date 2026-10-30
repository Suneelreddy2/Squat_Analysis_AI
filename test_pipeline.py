
import time
from vision.processor import SquatVideoProcessor

def progress_bar(pct, msg):
    print(f"[{pct*100:5.1f}%] {msg}")

print("Initializing SquatVideoProcessor...")
t0 = time.time()
processor = SquatVideoProcessor()

print("Processing video: samples/common_sample.mp4...")
result = processor.process_video("samples/common_sample.mp4", output_dir="output", progress_callback=progress_bar)
t1 = time.time()

print("\nProcessing completed in", round(t1 - t0, 2), "seconds!")
print("Duration:", result["duration_sec"], "s")
print("Reps detected:", len(result["repetitions"]))
for rep in result["repetitions"]:
    print(f"\n--- REP {rep['rep_number']} ---")
    print(f"Bottom frame: {rep['bottom_frame']} (t={rep['bottom_timestamp']:.2f}s)")
    for f in rep["findings"]:
        print(f"  [{f['status']}] {f['name']}: {f['observed']}")

print("\nAnnotated Video Output:", result["annotated_video_path"])

