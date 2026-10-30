import os
import subprocess
import logging

logger = logging.getLogger(__name__)

def transcode_to_h264(input_path: str, output_path: str) -> bool:
    """
    Transcodes any video to browser-standard H.264 (yuv420p) MP4.
    Tries PyAV (pure Python with bundled libx264) first, then imageio-ffmpeg,
    and finally system ffmpeg binaries.
    Returns True if transcoding succeeded, False otherwise.
    """
    if not os.path.isfile(input_path) or os.path.getsize(input_path) == 0:
        logger.warning(f"Input file for transcoding is missing or empty: {input_path}")
        return False

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # 1. Try PyAV (fast, pure Python, bundled libx264)
    try:
        import av
        in_container = av.open(input_path)
        if not in_container.streams.video:
            in_container.close()
            raise ValueError("No video stream found in input")

        in_stream = in_container.streams.video[0]
        width = in_stream.codec_context.width
        height = in_stream.codec_context.height
        # H.264 requires dimensions to be even numbers
        width = width - (width % 2)
        height = height - (height % 2)

        rate = in_stream.average_rate
        if rate is None or float(rate) <= 0:
            rate = 30

        out_container = av.open(output_path, mode="w")
        out_stream = out_container.add_stream("libx264", rate=rate)
        out_stream.width = width
        out_stream.height = height
        out_stream.pix_fmt = "yuv420p"
        out_stream.options = {"preset": "ultrafast", "crf": "23"}

        for frame in in_container.decode(in_stream):
            # If frame size doesn't match even dimensions, reformat
            if frame.width != width or frame.height != height:
                frame = frame.reformat(width=width, height=height, format="yuv420p")
            for packet in out_stream.encode(frame):
                out_container.mux(packet)

        for packet in out_stream.encode():
            out_container.mux(packet)

        in_container.close()
        out_container.close()

        if os.path.isfile(output_path) and os.path.getsize(output_path) > 0:
            return True
    except Exception as e:
        logger.warning(f"PyAV transcoding failed, attempting fallback: {e}")
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
            except OSError:
                pass

    # 2. Fallback to imageio-ffmpeg or system ffmpeg
    ffmpeg_executables = []
    try:
        import imageio_ffmpeg
        ffmpeg_executables.append(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception:
        pass
    ffmpeg_executables.extend(["ffmpeg", "ffmpeg.exe"])

    for exe in ffmpeg_executables:
        try:
            cmd = [
                exe, "-y",
                "-i", input_path,
                "-vcodec", "libx264",
                "-preset", "ultrafast",
                "-crf", "23",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                output_path
            ]
            res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=180)
            if res.returncode == 0 and os.path.isfile(output_path) and os.path.getsize(output_path) > 0:
                return True
        except Exception:
            continue

    return False

