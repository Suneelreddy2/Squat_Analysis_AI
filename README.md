# 🏋️ AI Barbell Squat Analysis & Biomechanics Studio

A pure Python Computer Vision and Biomechanics assessment system grounded in Starting Strength barbell squat standards (*Document_for_skill.pdf*).

## 🌟 Key Features

- **🎥 Dual-Synced Video Studio**: Synchronized side-by-side player (Raw Input vs AI Annotated) featuring lockstep scrubbing, slow-motion playback (`0.25x`, `0.5x`, `0.75x`, `1.0x`), frame-by-frame stepping (`◀` / `▶`), and keyboard hotkeys.
- **🔍 Rep-by-Rep Audit**: Granular evaluation of each repetition against 7 Starting Strength criteria, featuring visual pass badges, observed measurements, book page citations, and actionable coaching cues.
- **📊 Continuous Kinematic Telemetry**: Interactive trajectory plots for knee angle, ground-relative torso angle, and horizontal bar drift from the midfoot plumb line, with one-click `.CSV` data export.
- **📷 Live Camera Counter**: Real-time webcam rep counting and biomechanical HUD metrics via WebRTC using MediaPipe Pose.
- **📖 Document-Grounded Skill Engine**: 7 codified rules mapped directly to Starting Strength squat criteria with authoritative page citations and explicit side-view assessability flags.
- **✨ High-Tech Sports-Science UI**: Glassmorphic dark luxury dashboard styled with custom CSS tokens, modern typography (*Outfit*, *Plus Jakarta Sans*, *JetBrains Mono*), and responsive controls.

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- `ffmpeg` (installed on system)

### Installation

```bash
# Clone the repository
git clone https://github.com/CHANDU32455/HiringTests.git
cd HiringTests

# Create & activate virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running the App

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

For live camera access, open the app on the same computer using `localhost`. Browsers require a secure page for webcam access: remote or LAN access must use HTTPS. In the live counter, start the camera to begin recording, stop the camera to save the clip, then select **Analyze recording** to open the full report with the raw video, annotated video, rep audit, and telemetry.

### Hosted webcam setup

Remote WebRTC connections may fail with STUN alone when a network blocks direct peer-to-peer traffic. For Streamlit Community Cloud, configure Cloudflare Realtime TURN credentials in the app's **Manage app → Settings → Secrets** panel:

```toml
CLOUDFLARE_TURN_KEY_ID = "your-turn-key-id"
CLOUDFLARE_TURN_KEY_API_TOKEN = "your-turn-api-token"
```

Create these credentials in Cloudflare Realtime. Keep them in Streamlit Secrets; do not commit them to the repository. The app passes them to `streamlit-webrtc`, which obtains short-lived ICE server credentials. Without TURN credentials, the app falls back to Google STUN, which may not work on restrictive networks.

## 📁 Project Structure

```
├── app.py                      # Main Streamlit application with custom player component
├── styles.css                  # High-tech glassmorphism design system & Streamlit overrides
├── squat_skill/
│   ├── squat_rules.yaml        # 7 rule specifications with Starting Strength PDF citations
│   └── skill_engine.py         # Rule evaluation engine
├── vision/
│   ├── pose_detector.py        # MediaPipe Pose landmark extraction (dominant side auto-detect)
│   ├── barbell_tracker.py      # Bar path tracking (shoulder anchor + Hough circles + EMA)
│   ├── rep_detector.py         # Repetition phase & turnaround segmentation
│   ├── biomechanics.py         # Joint angles, midfoot plumb, and patella-crease depth delta
│   ├── video_annotator.py      # Biomechanical skeletal and trajectory overlay drawing
│   ├── processor.py            # End-to-end video processing pipeline & H.264 transcoding
│   └── live_counter.py         # Real-time state machine for webcam rep counting
├── samples/
│   └── common_sample.mp4       # Pre-loaded benchmark squat video
└── requirements.txt            # Package dependencies
```
