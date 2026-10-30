import os
import sys
import tempfile
import hashlib
import subprocess
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
try:
    import av
    from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
    HAS_WEBRTC = True
except ImportError:
    HAS_WEBRTC = False
    webrtc_streamer = None
    VideoProcessorBase = object
    RTCConfiguration = None

# Ensure local imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vision.processor import SquatVideoProcessor
from squat_skill.skill_engine import SquatSkillEngine
from vision.live_counter import LiveRepCounter

# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Barbell Squat Analysis & Biomechanics Studio",
    page_icon="🏋️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Inject Custom CSS Design System ───────────────────────────────────────────
css_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "styles.css")
if os.path.exists(css_path):
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# ── WebRTC Live Camera Transformer ───────────────────────────────────────────
if HAS_WEBRTC:
    class _SquatTransformer(VideoProcessorBase):
        """Processes webcam frames server-side: MediaPipe pose + real-time rep counter."""
        def __init__(self):
            self.counter = LiveRepCounter()

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            bgr = frame.to_ndarray(format="bgr24")
            out = self.counter.process_frame(bgr, timestamp=frame.time)
            return av.VideoFrame.from_ndarray(out, format="bgr24")
else:
    _SquatTransformer = None


def _render_live_camera():
    """Live interactive webcam studio panel."""
    st.markdown(f"""
    <div class="card" style="padding:16px 20px;margin-bottom:14px;">
      <div style="display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap;">
        <div>
          <h3 style="margin:0;color:#fff;font-size:1.25rem;">Live squat session</h3>
          <p style="margin:3px 0 0;color:#cbd5e1;font-size:.88rem;">Side profile · full body in frame · camera at hip height</p>
        </div>
        <span class="citation-pill">Live camera analysis</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    if not HAS_WEBRTC:
        st.warning("streamlit-webrtc and av are required for live camera recording. Install them with pip install streamlit-webrtc av." )
        return None

    c_stream, c_setup = st.columns([2, 1], gap="large")

    with c_stream:
        def configured_secret(name):
            value = os.getenv(name)
            try:
                value = st.secrets.get(name, value)
            except Exception:
                pass
            return str(value).strip() if value else ""

        cloudflare_key_id = configured_secret("CLOUDFLARE_TURN_KEY_ID")
        cloudflare_api_token = configured_secret("CLOUDFLARE_TURN_KEY_API_TOKEN")
        turn_url = configured_secret("WEBRTC_TURN_URL")
        turn_username = configured_secret("WEBRTC_TURN_USERNAME")
        turn_credential = configured_secret("WEBRTC_TURN_CREDENTIAL")
        has_turn = bool(cloudflare_key_id and cloudflare_api_token) or bool(
            turn_url and turn_username and turn_credential
        )

        if cloudflare_key_id and cloudflare_api_token:
            # streamlit-webrtc reads these server-side and obtains short-lived
            # Cloudflare TURN credentials when rtc_configuration is omitted.
            os.environ["CLOUDFLARE_TURN_KEY_ID"] = cloudflare_key_id
            os.environ["CLOUDFLARE_TURN_KEY_API_TOKEN"] = cloudflare_api_token
            rtc_config = None
        else:
            ice_servers = [{"urls": ["stun:stun.l.google.com:19302"]}]
            if turn_url and turn_username and turn_credential:
                ice_servers.append({
                    "urls": [turn_url],
                    "username": turn_username,
                    "credential": turn_credential,
                })
            rtc_config = RTCConfiguration({"iceServers": ice_servers})

        ctx = webrtc_streamer(
            key="squat-live",
            video_processor_factory=_SquatTransformer,
            rtc_configuration=rtc_config,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

    with c_setup:
        with st.container(border=True):
            st.markdown("#### Session setup")
            st.markdown(
                "1. Place the camera at hip height.\n"
                "2. Stand sideways to the camera.\n"
                "3. Keep your head, hips, knees, and feet visible."
            )
            st.info("Use http://localhost:8501 on this computer. Camera access requires HTTPS when opening the app through a LAN or remote address.")
            if not has_turn:
                st.warning(
                    "Remote camera connections may time out with STUN alone. "
                    "Configure Cloudflare TURN credentials in Streamlit Secrets for hosted use."
                )

    counter = getattr(ctx.video_processor, "counter", None) if ctx else None
    if counter is not None:
        st.session_state["live_counter"] = counter
    else:
        counter = st.session_state.get("live_counter")
    camera_playing = bool(ctx and ctx.state.playing)
    recording_active = st.session_state.get("live_recording_active", False)
    if counter and camera_playing and not recording_active:
        st.session_state.pop("live_recording_path", None)
        st.session_state.pop("analysis_result", None)
        st.session_state.pop("active_video", None)
        counter.reset()
        st.session_state["live_recording_active"] = counter.start_recording("output")
        recording_active = st.session_state["live_recording_active"]
    elif counter and not camera_playing and recording_active:
        saved_path = counter.stop_recording()
        st.session_state["live_recording_active"] = False
        recording_active = False
        if saved_path:
            st.session_state["live_recording_path"] = saved_path

    phase_map = {
        "BOTTOM": "🟢 AT BOTTOM", "DESCENT": "🔽 DESCENT",
        "ASCENT": "🔼 ASCENT", "STANDING": "⏸ STANDING",
    }

    with st.container(border=True):
        st.markdown("#### Live telemetry")
        k1, k2, k3, k4, k5 = st.columns(5)
        slot_reps, slot_phase, slot_knee, slot_back, slot_depth = [
            col.empty() for col in (k1, k2, k3, k4, k5)
        ]

        def render_live_metrics():
            metrics = counter.get_metrics() if counter else {}
            slot_reps.metric("Reps", str(metrics.get("rep_count", 0)))
            slot_phase.metric("Phase", phase_map.get(metrics.get("phase"), "⏸ STANDING"))
            slot_knee.metric("Knee angle", f"{metrics.get('knee_angle', 0):.0f}°")
            slot_back.metric("Torso angle", f"{metrics.get('back_angle', 0):.0f}°")
            slot_depth.metric("Last rep depth", "✅ Passed" if metrics.get("depth_ok") else "—")

        if hasattr(st, "fragment"):
            @st.fragment(run_every=1)
            def refresh_live_metrics():
                render_live_metrics()
            refresh_live_metrics()
        else:
            render_live_metrics()

        controls = st.columns([2.2, 1.5])
        with controls[0]:
            if recording_active:
                st.success("Recording while camera is on · stop the camera to finish")
            elif st.session_state.get("live_recording_path"):
                st.success("Recording saved · ready to analyze")
            else:
                st.caption("Start the camera to begin recording; stop it to save the clip.")
        with controls[1]:
            recorded_path = st.session_state.get("live_recording_path")
            if recorded_path and os.path.isfile(recorded_path):
                st.caption(f"Saved · {os.path.basename(recorded_path)}")
                if st.button("Analyze recording", key="live_analyze_recording", use_container_width=True):
                    return recorded_path
    return None


# ── Top Navigation Header & Brand Bar ─────────────────────────────────────────
st.markdown("""
<div style="display:flex;justify-content:space-between;align-items:center;padding:14px 24px;
            background:linear-gradient(135deg,#0f172a,#1e293b);
            border:1px solid rgba(255,255,255,0.12);border-radius:16px;margin-bottom:20px;
            box-shadow:0 8px 32px rgba(0,0,0,0.5);">
  <div style="display:flex;align-items:center;gap:16px;">
    <div style="background:linear-gradient(135deg,#6366f1,#8b5cf6);width:46px;height:46px;
                border-radius:12px;display:flex;align-items:center;justify-content:center;
                font-size:1.6rem;box-shadow:0 0 20px rgba(99,102,241,0.5);">
      🏋️
    </div>
    <div>
      <div style="font-family:'Outfit',sans-serif;font-size:1.5rem;font-weight:800;letter-spacing:-0.02em;color:#ffffff;">
        AESTHETIQ BIOMECHANICS
      </div>
      <div style="color:#cbd5e1;font-size:0.84rem;font-weight:500;">
        Pure Python Computer Vision · MediaPipe Pose v0.10 · Starting Strength Standards
      </div>
    </div>
  </div>
  <div style="display:flex;align-items:center;gap:12px;">
    <span class="status-badge status-badge-pass">● Vision Pipeline Online</span>
    <span class="citation-pill">📖 7 Rules Grounded</span>
  </div>
</div>
""", unsafe_allow_html=True)

# Mode Selector
nav_col1, nav_col2 = st.columns([3, 1])
with nav_col1:
    app_mode = st.radio(
        "Application Mode",
        options=["📂 Video Analysis Studio", "📷 Live Camera Counter"],
        horizontal=True,
        label_visibility="collapsed",
        key="app_mode_switch"
    )

selected_video_path = None
run_analysis = False

sample_video_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "samples", "common_sample.mp4")

# Interactive Control Toolbar for Video Mode
if app_mode == "📂 Video Analysis Studio":
    st.markdown("""
    <div style="background:rgba(18,27,48,0.95);border:1px solid rgba(255,255,255,0.12);
                border-radius:16px;padding:18px 24px;margin-bottom:20px;">
    """, unsafe_allow_html=True)

    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1.2, 2.6, 1.2])

    with ctrl_col1:
        video_source_type = st.selectbox(
            "Video Ingestion Source:",
            options=["🎥 YouTube Video / Benchmark Sample", "📤 Upload Custom Video (MP4/MOV)"],
            key="video_source_dropdown"
        )

    if video_source_type == "🎥 YouTube Video / Benchmark Sample":
        with ctrl_col2:
            yt_url = st.text_input(
                "YouTube Video Link:",
                value="https://youtube.com/shorts/TRvg083BrXY",
                help="Paste any YouTube squat video or short URL",
                key="yt_url_input"
            )
        default_url = "https://youtube.com/shorts/TRvg083BrXY"
        is_default = (yt_url.strip() == default_url or "TRvg083BrXY" in yt_url)

        if is_default and os.path.exists(sample_video_path):
            selected_video_path = sample_video_path
        else:
            if (
                st.session_state.get("custom_yt_url") == yt_url.strip()
                and "custom_yt_path" in st.session_state
                and os.path.exists(st.session_state["custom_yt_path"])
            ):
                selected_video_path = st.session_state["custom_yt_path"]

        with ctrl_col3:
            st.write("")
            if not is_default and selected_video_path is None:
                if st.button("⬇️ Download Video", use_container_width=True):
                    with st.spinner("Downloading with yt-dlp..."):
                        os.makedirs("samples", exist_ok=True)
                        url_key = hashlib.sha256(yt_url.strip().encode("utf-8")).hexdigest()[:12]
                        custom_path = os.path.join("samples", f"custom_yt_{url_key}.mp4")
                        try:
                            cmd = [sys.executable, "-m", "yt_dlp", "-f", "mp4", "-o", custom_path, yt_url]
                            res = subprocess.run(cmd, capture_output=True, text=True)
                            if res.returncode == 0 and os.path.exists(custom_path):
                                st.session_state["custom_yt_path"] = custom_path
                                st.session_state["custom_yt_url"] = yt_url.strip()
                                st.success("Video downloaded successfully!")
                                st.rerun()
                            else:
                                st.error(f"Failed to download: {res.stderr[:200] if res.stderr else 'Unknown error'}")
                        except Exception as e:
                            st.error(f"Download error: {e}")
            else:
                run_analysis = st.button("🚀 Run AI Assessment", type="primary", use_container_width=True)

    else:
        with ctrl_col2:
            uploaded = st.file_uploader(
                "Upload Side-View Squat Video (MP4 / MOV / AVI)",
                type=["mp4", "mov", "avi"],
                key="file_uploader_control"
            )
            if uploaded:
                video_bytes = uploaded.getvalue()
                upload_key = hashlib.sha256(video_bytes).hexdigest()
                if st.session_state.get("uploaded_video_key") != upload_key:
                    suffix = os.path.splitext(uploaded.name)[1].lower() or ".mp4"
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tfile:
                        tfile.write(video_bytes)
                    old_path = st.session_state.get("uploaded_video_path")
                    st.session_state["uploaded_video_path"] = tfile.name
                    st.session_state["uploaded_video_key"] = upload_key
                    if old_path and os.path.isfile(old_path):
                        try:
                            os.remove(old_path)
                        except OSError:
                            pass
                selected_video_path = st.session_state["uploaded_video_path"]
        with ctrl_col3:
            st.write("")
            run_analysis = st.button("🚀 Run AI Assessment", type="primary", use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

if app_mode == "📷 Live Camera Counter":
    submitted_recording = _render_live_camera()
    selected_video_path = st.session_state.get("live_recording_path")
    run_analysis = bool(submitted_recording)

# ── Processing & Main Dashboard View ──────────────────────────────────────────
if selected_video_path and app_mode in ("📂 Video Analysis Studio", "📷 Live Camera Counter"):
    need_run = (
        run_analysis
        or "analysis_result" not in st.session_state
        or st.session_state.get("active_video") != selected_video_path
    )

    if run_analysis and need_run:
        prog = st.progress(0, text="Initialising Vision & Biomechanics Pipeline…")
        stat = st.empty()

        def cb(pct, msg):
            prog.progress(int(pct * 100), text=msg)
            stat.caption(f"⚙️ {msg}")

        try:
            processor = SquatVideoProcessor()
            result = processor.process_video(
                selected_video_path, output_dir="output", progress_callback=cb
            )
            st.session_state["analysis_result"] = result
            st.session_state["active_video"] = selected_video_path
            prog.empty()
            stat.empty()
            st.toast("🎉 Biomechanical assessment complete!", icon="✅")
        except Exception as exc:
            prog.empty()
            stat.empty()
            st.error(f"Vision Processing Error: {exc}")

    result = st.session_state.get("analysis_result")

    if result:
        reps = result.get("repetitions", [])

        # Executive KPI Row
        all_findings = [f for rep in reps for f in rep.get("findings", [])]
        passes = sum(
            sum(1 for f in r["findings"] if f["status"] == "MEETS_STANDARD")
            for r in reps
        )
        failed_checks = sum(f["status"] == "DOES_NOT_MEET_STANDARD" for f in all_findings)
        unassessed_checks = sum(f["status"] == "CANNOT_ASSESS" for f in all_findings)
        total_ev = passes + failed_checks
        pass_rate = (passes / total_ev * 100) if total_ev else 0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("⏱ Total Duration", f"{result['duration_sec']:.1f} s")
        c2.metric("🔁 Detected Reps", f"{len(reps)} Reps")
        c3.metric("✅ Assessable Checks Passed", f"{pass_rate:.0f}%", help=f"{passes} passed · {failed_checks} need attention · {unassessed_checks} could not be assessed")
        c4.metric("🎬 Video Output", "H.264 MP4 Synced")

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

        # Tab Navigation
        video_tab_label = "🎞️ Raw + analyzed video" if app_mode == "📷 Live Camera Counter" else "🎥 Video Analysis Studio"
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            video_tab_label,
            "📋 Rep-by-Rep Audit",
            "📊 Biomechanics Telemetry",
            "📖 Skill Rule Inspector",
            "📷 Live Session",
        ])

        # ── TAB 1: Clean Dual Video Analysis Studio ───────────────────────────
        with tab1:
            ann_path = result.get("annotated_video_path")
            
            # Action & metadata toolbar
            t_col1, t_col2, t_col3 = st.columns([3, 1, 1])
            with t_col1:
                st.markdown(f"""
                <div style="display:flex;gap:12px;align-items:center;margin-bottom:12px;">
                  <span class="observed-pill">Video: {os.path.basename(selected_video_path)}</span>
                  <span class="observed-pill">FPS: {result.get('fps', 30.0):.1f}</span>
                  <span class="observed-pill">Total Frames: {result.get('total_frames', 0)}</span>
                </div>
                """, unsafe_allow_html=True)
            
            with t_col2:
                if os.path.exists(selected_video_path):
                    with open(selected_video_path, "rb") as raw_file:
                        st.download_button(
                            label="Download raw",
                            data=raw_file.read(),
                            file_name=os.path.basename(selected_video_path),
                            mime="video/mp4",
                            use_container_width=True,
                        )
            with t_col3:
                if ann_path and os.path.exists(ann_path):
                    with open(ann_path, "rb") as vf:
                        st.download_button(
                            label="📥 Download Annotated Video",
                            data=vf.read(),
                            file_name=f"annotated_{os.path.basename(selected_video_path)}",
                            mime="video/mp4",
                            use_container_width=True
                        )

            # ── Synchronized dual HTML5 video player ──────────────────────────
            import base64

            def _video_b64(path: str) -> str:
                with open(path, "rb") as f:
                    return base64.b64encode(f.read()).decode()

            raw_b64  = _video_b64(selected_video_path)
            ann_b64  = _video_b64(ann_path) if ann_path and os.path.exists(ann_path) else None

            dual_player_html = f"""
<style>
.dual-player-wrap {{
  display: flex;
  gap: 16px;
  align-items: flex-start;
  flex-wrap: wrap;
}}
.vid-card {{
  flex: 1 1 0;
  min-width: 260px;
  background: rgba(15,23,42,0.95);
  border: 1px solid rgba(255,255,255,0.12);
  border-radius: 14px;
  overflow: hidden;
}}
.vid-card-header {{
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 10px 14px;
  background: rgba(30,41,59,0.9);
  border-bottom: 1px solid rgba(255,255,255,0.08);
  font-family: 'Inter', sans-serif;
  font-size: 0.82rem;
  font-weight: 600;
  color: #cbd5e1;
}}
.vid-badge-raw  {{ background:rgba(100,116,139,0.3); color:#94a3b8; padding:2px 8px; border-radius:6px; font-size:0.72rem; font-weight:700; }}
.vid-badge-ai   {{ background:rgba(34,197,94,0.2);  color:#4ade80; padding:2px 8px; border-radius:6px; font-size:0.72rem; font-weight:700; }}
.vid-card video {{
  width: 100%;
  max-height: 340px;
  object-fit: contain;
  display: block;
  background: #000;
}}
.sync-toolbar {{
  display: flex;
  gap: 10px;
  align-items: center;
  padding: 10px 0 4px 0;
  flex-wrap: wrap;
}}
.sync-btn {{
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 7px 18px;
  border-radius: 8px;
  border: none;
  font-family: 'Inter', sans-serif;
  font-size: 0.84rem;
  font-weight: 700;
  cursor: pointer;
  transition: all 0.18s;
}}
.sync-btn-play  {{ background: linear-gradient(135deg,#6366f1,#8b5cf6); color:#fff; }}
.sync-btn-pause {{ background: rgba(51,65,85,0.9); color:#cbd5e1; border:1px solid rgba(255,255,255,0.15); }}
.sync-btn-reset {{ background: rgba(51,65,85,0.9); color:#cbd5e1; border:1px solid rgba(255,255,255,0.15); }}
.sync-btn:hover  {{ transform: translateY(-1px); filter: brightness(1.1); }}
.sync-status {{
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.78rem;
  color: #64748b;
  margin-left: auto;
}}
</style>

<div class="sync-toolbar">
  <button type="button" class="sync-btn sync-btn-play" id="playBoth">▶ Play Both</button>
  <button type="button" class="sync-btn sync-btn-pause" id="pauseBoth">⏸ Pause Both</button>
  <button type="button" class="sync-btn sync-btn-reset" id="resetBoth">⟳ Reset</button>
  <span class="sync-status" id="syncStatus">Ready — click ▶ Play Both to start</span>
</div>

<div class="dual-player-wrap">
  <div class="vid-card">
    <div class="vid-card-header">
      📹 Camera 01 · Original Input Footage
      <span class="vid-badge-raw">RAW INPUT</span>
    </div>
    <video id="vidRaw" preload="auto" controls>
      <source src="data:video/mp4;base64,{raw_b64}" type="video/mp4">
    </video>
  </div>
  <div class="vid-card">
    <div class="vid-card-header">
      🤖 Camera 02 · AI Annotated Biomechanics
      <span class="vid-badge-ai">POSE + SKELETON + HUD</span>
    </div>
    {'<video id="vidAnn" preload="auto" controls><source src="data:video/mp4;base64,' + ann_b64 + '" type="video/mp4"></video>' if ann_b64 else '<div style="padding:40px;text-align:center;color:#64748b;">⚠️ Annotated video not generated yet.</div>'}
  </div>
</div>

<script>
(function() {{
  var raw = document.getElementById('vidRaw');
  var ann = document.getElementById('vidAnn');
  var status = document.getElementById('syncStatus');
  function fmt(t) {{
    var m = Math.floor(t/60), s = (t%60).toFixed(2);
    return m + ':' + (s < 10 ? '0' : '') + s;
  }}

  function align(source, target) {{
    if (target && Number.isFinite(source.currentTime) &&
        Math.abs(source.currentTime - target.currentTime) > 0.15) {{
      target.currentTime = source.currentTime;
    }}
  }}

  document.getElementById('playBoth').addEventListener('click', async function() {{
    if (!raw) return;
    if (ann) align(raw, ann);
    try {{
      await Promise.all([raw.play(), ann ? ann.play() : Promise.resolve()]);
      status.textContent = 'Playing — both videos synced';
    }} catch (error) {{
      status.textContent = 'Playback could not start. Use the video controls or check browser playback permissions.';
    }}
  }});

  document.getElementById('pauseBoth').addEventListener('click', function() {{
    if (raw) raw.pause();
    if (ann) ann.pause();
    status.textContent = 'Paused at ' + fmt(raw ? raw.currentTime : 0);
  }});

  document.getElementById('resetBoth').addEventListener('click', function() {{
    if (raw) {{ raw.pause(); raw.currentTime = 0; }}
    if (ann) {{ ann.pause(); ann.currentTime = 0; }}
    status.textContent = 'Reset — click ▶ Play Both to start';
  }});

  if (raw && ann) {{
    raw.addEventListener('seeked', function() {{ align(raw, ann); }});
    ann.addEventListener('seeked', function() {{ align(ann, raw); }});
  }}

  // Keep the annotated player aligned to the source while playing.
  if (raw && ann) {{
    setInterval(function() {{
      if (!raw.paused && Math.abs(raw.currentTime - ann.currentTime) > 0.25) {{
        ann.currentTime = raw.currentTime;
      }}
      if (!raw.paused) {{
        status.textContent = '▶ ' + fmt(raw.currentTime) + ' / ' + fmt(raw.duration || 0);
      }}
    }}, 2000);
  }}
}})();
</script>
"""
            if hasattr(st, "iframe"):
                st.iframe(dual_player_html, height=560)
            else:
                # Keep compatibility with the minimum Streamlit version in
                # requirements.txt, which predates st.iframe.
                components.html(dual_player_html, height=560, scrolling=False)

            # Rep quick-jump reference strip
            if reps:
                st.markdown("<h4 style='color:#ffffff;margin-top:14px;'>⏱ Quick Rep Turnaround Reference</h4>", unsafe_allow_html=True)
                rep_cols = st.columns(len(reps))
                for idx, r in enumerate(reps):
                    with rep_cols[idx]:
                        st.markdown(f"""
                        <div class="card" style="padding:14px;text-align:center;">
                          <div style="font-weight:700;color:#38bdf8;font-size:0.95rem;">Repetition {r['rep_number']}</div>
                          <div style="font-family:'JetBrains Mono';font-size:0.88rem;color:#cbd5e1;margin-top:4px;">
                            Bottom @ <strong>{r['bottom_timestamp']:.2f}s</strong>
                          </div>
                          <div style="font-size:0.78rem;color:#94a3b8;margin-top:2px;">
                            Duration: {r['duration_sec']:.1f}s
                          </div>
                        </div>
                        """, unsafe_allow_html=True)

        # ── TAB 2: Rep-by-Rep Audit ───────────────────────────────────────────
        with tab2:
            if not reps:
                st.warning("⚠️ No completed squat repetitions were detected in this video segment.")
            else:
                rep_labels = [f"Repetition {r['rep_number']}" for r in reps]
                sel_rep = st.selectbox("Select Repetition to Inspect:", rep_labels, key="rep_audit_select")
                rep_idx = rep_labels.index(sel_rep)
                rep_data = reps[rep_idx]

                rep_passes = sum(1 for f in rep_data.get("findings", []) if f["status"] == "MEETS_STANDARD")
                rep_failed = sum(1 for f in rep_data.get("findings", []) if f["status"] == "DOES_NOT_MEET_STANDARD")
                rep_total = rep_passes + rep_failed
                rep_unassessed = sum(1 for f in rep_data.get("findings", []) if f["status"] == "CANNOT_ASSESS")
                rep_pct = (rep_passes / rep_total * 100) if rep_total else 0

                # Rep Kinematic Glance Strip
                st.markdown(f"""
                <div class="card" style="margin-bottom:16px;background:rgba(20,30,54,0.85);display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px;">
                  <div>
                    <h3 style="margin:0;color:#ffffff;font-size:1.35rem;">🔍 Audit Results — Repetition {rep_data['rep_number']}</h3>
                    <p style="color:#cbd5e1;font-size:0.88rem;margin:4px 0 0;">
                      Bottom Inflection: <strong>t = {rep_data['bottom_timestamp']:.2f} s</strong> (Frame #{rep_data['bottom_frame']})
                    </p>
                  </div>
                  <div style="display:flex;gap:16px;align-items:center;">
                    <div style="text-align:right;">
                      <div style="font-size:0.75rem;color:#94a3b8;text-transform:uppercase;font-weight:700;">Descent / Ascent Split</div>
                      <div style="font-family:'JetBrains Mono';font-size:0.95rem;color:#38bdf8;">
                        ↓ {rep_data['descent_duration_sec']:.1f}s &nbsp;|&nbsp; ↑ {rep_data['ascent_duration_sec']:.1f}s
                      </div>
                    </div>
                    <span class="status-badge {'status-badge-pass' if rep_pct >= 80 else ('status-badge-fail' if rep_pct < 50 else 'status-badge-cannot')}">
                      {rep_passes}/{rep_total} Assessable Checks Passed ({rep_pct:.0f}%)
                    </span>
                    <div style="font-size:0.75rem;color:#94a3b8;">{rep_unassessed} checks could not be assessed</div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

                # Render structured finding cards
                for f in rep_data.get("findings", []):
                    st_val = f["status"]
                    if st_val == "MEETS_STANDARD":
                        card_class = "audit-card audit-card-pass"
                        badge_html = '<span class="status-badge status-badge-pass">✅ MEETS STANDARD</span>'
                    elif st_val == "DOES_NOT_MEET_STANDARD":
                        card_class = "audit-card audit-card-fail"
                        badge_html = '<span class="status-badge status-badge-fail">❌ DOES NOT MEET STANDARD</span>'
                    else:
                        card_class = "audit-card audit-card-cannot"
                        badge_html = '<span class="status-badge status-badge-cannot">⚪ CANNOT ASSESS</span>'

                    st.markdown(f"""
                    <div class="{card_class}">
                      <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:10px;">
                        <div>
                          <h4 style="margin:0 0 6px 0;font-size:1.15rem;color:#ffffff;">{f['name']}</h4>
                          <span class="observed-pill">📐 {f['observed']}</span>
                        </div>
                        {badge_html}
                      </div>
                      <p style="margin:8px 0;color:#cbd5e1;font-size:0.94rem;line-height:1.6;">
                        {f['explanation']}
                      </p>
                      <div style="margin-top:12px;padding:10px 14px;background:rgba(6,182,212,0.12);border-left:4px solid #06b6d4;border-radius:4px;">
                        <span style="color:#38bdf8;font-size:0.88rem;font-weight:700;">💡 Actionable Coaching Cue:</span>
                        <span style="color:#f8fafc;font-size:0.88rem;"> {f['actionable_feedback']}</span>
                      </div>
                      <div style="margin-top:12px;">
                        <span class="citation-pill">
                          📖 Ref: {f['citation']['ref_pages']} ({f['citation']['pdf_pages']}) · {f['citation'].get('figure', '')} · {f['citation']['section']}
                        </span>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

        # ── TAB 3: Biomechanics Telemetry ─────────────────────────────────────
        with tab3:
            st.markdown("<h3 style='color:#ffffff;'>📊 Continuous Kinematic Trajectory</h3>", unsafe_allow_html=True)
            tel = result.get("frames_telemetry", [])
            if tel:
                valid_tel = [tf for tf in tel if tf.get("valid", False)]
                df = pd.DataFrame([
                    {
                        "Time (s)": tf.get("timestamp", 0),
                        "Knee Flexion Angle (°)": tf.get("knee_angle", 0),
                        "Torso Back Angle (°)": tf.get("back_angle", 0),
                        "Bar Horizontal Drift (px)": tf.get("bar_dev_px", 0),
                    }
                    for tf in valid_tel
                ])

                # Telemetry KPI Strip
                if not df.empty:
                    min_knee = df["Knee Flexion Angle (°)"].min()
                    min_back = df["Torso Back Angle (°)"].min()
                    max_drift = df["Bar Horizontal Drift (px)"].abs().max()

                    t1, t2, t3 = st.columns(3)
                    t1.metric("🦵 Max Knee Flexion", f"{min_knee:.1f}°")
                    t2.metric("📐 Min Torso Angle", f"{min_back:.1f}°", help="Target low-bar back angle: 40°–50° at bottom")
                    t3.metric("⚖️ Max Barbell Drift", f"{max_drift:.1f} px", help="Target: 0 px deviation from midfoot line")

                st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

                col_chart1, col_chart2 = st.columns(2)
                with col_chart1:
                    st.markdown("""
                    <div class="card" style="padding:16px;">
                      <h4 style="font-size:1.0rem;margin-bottom:4px;color:#38bdf8;">📈 Joint & Torso Angles over Time</h4>
                      <p style="font-size:0.84rem;color:#cbd5e1;margin:0 0 10px 0;">Knee flexion vs Torso angle (ground-relative). Notice inflection points at bottom.</p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.line_chart(df.set_index("Time (s)")[["Knee Flexion Angle (°)", "Torso Back Angle (°)"]])

                with col_chart2:
                    st.markdown("""
                    <div class="card" style="padding:16px;">
                      <h4 style="font-size:1.0rem;margin-bottom:4px;color:#38bdf8;">⚖️ Barbell Balance vs Midfoot Plumb Line</h4>
                      <p style="font-size:0.84rem;color:#cbd5e1;margin:0 0 10px 0;">Horizontal displacement (px) from center of foot arch. Ideal path is 0.</p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.line_chart(df.set_index("Time (s)")[["Bar Horizontal Drift (px)"]])

                # Export CSV button
                csv_bytes = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Export Kinematic Telemetry (.CSV)",
                    data=csv_bytes,
                    file_name="squat_telemetry_analysis.csv",
                    mime="text/csv",
                    use_container_width=False
                )

        # ── TAB 4: Skill Rule Inspector ───────────────────────────────────────
        with tab4:
            st.markdown("""
            <div class="card" style="margin-bottom:16px;">
              <h3 style="margin:0 0 6px 0;color:#ffffff;">
                📖 Document-Derived Starting Strength Skill Base
              </h3>
              <p style="color:#cbd5e1;font-size:0.92rem;margin:0;">
                7 criteria extracted from <em>Document_for_skill.pdf</em> (Starting Strength standards). Each rule maps to authoritative page citations and figures.
              </p>
            </div>
            """, unsafe_allow_html=True)

            engine = SquatSkillEngine()
            for rule_id, rule in engine.rules.items():
                is_side = rule.get("assessable_from_side_view", True)
                icon = "✅" if is_side else "⚪"
                side_badge = '<span class="status-badge status-badge-pass">Side-View Assessable</span>' if is_side else '<span class="status-badge status-badge-cannot">Requires Front / Rear View</span>'

                with st.expander(f"{icon} {rule['name']}  ·  {rule['category']}"):
                    st.markdown(f"""
                    <div style="margin-bottom:12px;">
                      {side_badge}
                      <span class="citation-pill" style="margin-left:8px;">
                        📖 Ref: {rule['citation']['ref_pages']} ({rule['citation']['pdf_pages']}) · {rule['citation'].get('figure', '')} · {rule['citation']['section']}
                      </span>
                    </div>
                    <p style="color:#f8fafc;font-size:0.94rem;line-height:1.6;"><strong>Description:</strong> {rule['description']}</p>
                    """, unsafe_allow_html=True)

                    col_p, col_f = st.columns(2)
                    with col_p:
                        st.success(f"**Standard Criteria:** {rule['pass_message']}")
                    with col_f:
                        st.error(f"**Fault Criteria:** {rule['fail_message']}")

        # ── TAB 5: Live Camera Counter ────────────────────────────────────────
        with tab5:
            if app_mode == "📂 Video Analysis Studio":
                _render_live_camera()
            else:
                st.info("Recording controls are available in the live camera panel above.")

        if app_mode == "📷 Live Camera Counter":
            with st.container(border=True):
                st.markdown("### Live session outcome")
                st.caption("Analysis is run on the saved camera recording. Unassessable checks are excluded from the pass percentage.")
                outcome_cols = st.columns(4)
                outcome_cols[0].metric("Completed reps", str(len(reps)))
                outcome_cols[1].metric("Checks passed", str(passes))
                outcome_cols[2].metric("Need attention", str(failed_checks))
                outcome_cols[3].metric("Could not assess", str(unassessed_checks))

                rep_rows = []
                for rep in reps:
                    rep_findings = rep.get("findings", [])
                    depth = next((f["status"] for f in rep_findings if f.get("rule_id") == "SQUAT_DEPTH"), "CANNOT_ASSESS")
                    rep_rows.append({
                        "Rep": rep["rep_number"],
                        "Duration": f"{rep.get('duration_sec', 0):.2f} s",
                        "Depth": depth.replace("_", " ").title(),
                        "Passed": sum(f["status"] == "MEETS_STANDARD" for f in rep_findings),
                        "Needs attention": sum(f["status"] == "DOES_NOT_MEET_STANDARD" for f in rep_findings),
                        "Unassessed": sum(f["status"] == "CANNOT_ASSESS" for f in rep_findings),
                    })
                if rep_rows:
                    st.dataframe(pd.DataFrame(rep_rows), hide_index=True, use_container_width=True)
                else:
                    st.info("No complete repetitions were detected in this recording. The raw and analyzed videos are available above for review.")

elif app_mode == "📷 Live Camera Counter":
    pass

else:
    # ── Landing Hero Showcase (when no video is analyzed yet) ───────────────────
    st.markdown("""
    <div class="card" style="padding:42px 32px;text-align:center;margin-top:10px;background:linear-gradient(145deg,#0f172a,#1e293b);">
      <div style="display:inline-block;background:linear-gradient(135deg,#6366f1,#06b6d4);padding:14px;border-radius:20px;box-shadow:0 0 30px rgba(99,102,241,0.5);margin-bottom:18px;">
        <span style="font-size:2.8rem;">🏋️</span>
      </div>
      <h2 style="font-size:2.2rem;margin-bottom:10px;color:#ffffff;">
        AI Barbell Squat Biomechanics & Skill Evaluator
      </h2>
      <p style="color:#cbd5e1;font-size:1.05rem;max-width:720px;margin:0 auto 24px;line-height:1.6;">
        Computer vision motion analysis grounded in Starting Strength low-bar squat standards.
        Instant video playback, anatomical depth tracking, midfoot balance lines, and document citations.
      </p>
    </div>
    """, unsafe_allow_html=True)

    # 4 Feature Showcase Cards
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        st.markdown("""
        <div class="card" style="height:100%;padding:20px;">
          <div style="font-size:1.8rem;margin-bottom:10px;">🎥</div>
          <h4 style="font-size:1.1rem;color:#38bdf8;margin-bottom:6px;">Dual Video Studio</h4>
          <p style="color:#cbd5e1;font-size:0.88rem;line-height:1.5;">
            Side-by-side native video player comparing raw movement against AI skeletal and barbell overlays.
          </p>
        </div>
        """, unsafe_allow_html=True)

    with f2:
        st.markdown("""
        <div class="card" style="height:100%;padding:20px;">
          <div style="font-size:1.8rem;margin-bottom:10px;">📐</div>
          <h4 style="font-size:1.1rem;color:#38bdf8;margin-bottom:6px;">Patella-Crease Depth</h4>
          <p style="color:#cbd5e1;font-size:0.88rem;line-height:1.5;">
            Detects hip crease apex vs patella top. Validates below-parallel standards (Fig 2-1 & 2-10).
          </p>
        </div>
        """, unsafe_allow_html=True)

    with f3:
        st.markdown("""
        <div class="card" style="height:100%;padding:20px;">
          <div style="font-size:1.8rem;margin-bottom:10px;">⚖️</div>
          <h4 style="font-size:1.1rem;color:#38bdf8;margin-bottom:6px;">Midfoot Plumb Line</h4>
          <p style="color:#cbd5e1;font-size:0.88rem;line-height:1.5;">
            Barbell trajectory ribbon color-coded to horizontal deviation from the midfoot balance point.
          </p>
        </div>
        """, unsafe_allow_html=True)

    with f4:
        st.markdown("""
        <div class="card" style="height:100%;padding:20px;">
          <div style="font-size:1.8rem;margin-bottom:10px;">📖</div>
          <h4 style="font-size:1.1rem;color:#38bdf8;margin-bottom:6px;">Document Citations</h4>
          <p style="color:#cbd5e1;font-size:0.88rem;line-height:1.5;">
            Every finding cites Starting Strength page numbers, figure diagrams, and coaching cues.
          </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align:center;margin-top:24px;">
      <span style="color:#cbd5e1;font-size:0.92rem;">
        Ready to test? Select <strong>🎥 YouTube Video / Benchmark Sample</strong> above and click <strong>🚀 Run AI Assessment</strong>.
      </span>
    </div>
    """, unsafe_allow_html=True)
