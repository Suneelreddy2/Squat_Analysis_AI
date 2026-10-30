import time
from vision.processor import SquatVideoProcessor

def progress_bar(pct, msg):
    print(f"[{pct*100:5.1f}%] {msg}")

print("Initializing SquatVideoProcessor (Fast Mode)...")
t0 = time.time()
processor = SquatVideoProcessor(model_complexity=1)

print("Processing video: samples/common_sample.mp4...")
result = processor.process_video("samples/common_sample.mp4", output_dir="output", progress_callback=progress_bar)
t1 = time.time()

print("\nProcessing completed in", round(t1 - t0, 2), "seconds!")
print("Duration:", round(result["duration_sec"], 2), "s")
print("Reps detected:", len(result["repetitions"]))
for rep in result["repetitions"]:
    print(f"\n================ REP {rep['rep_number']} ================")
    print(f"Descent: {rep['descent_duration_sec']:.2f}s | Ascent: {rep['ascent_duration_sec']:.2f}s")
    print(f"Bottom frame: #{rep['bottom_frame']} (t = {rep['bottom_timestamp']:.2f}s)")
    print("Findings:")
    for f in rep["findings"]:
        frame_info = f"Frame #{f.get('frame', 'N/A')} (t = {f.get('timestamp', 0.0):.2f}s)"
        print(f"  • [{f['status']}] {f['name']}")
        print(f"    Evidence: {frame_info} | {f['observed']}")
        print(f"    Citation: {f['citation']['ref_pages']} ({f['citation']['pdf_pages']}) · {f['citation'].get('figure', '')}")
        print(f"    Feedback: {f['actionable_feedback']}")

print("\nAnnotated Video Output:", result["annotated_video_path"])
print("Static Stream URL:", result.get("static_annotated_url"))
