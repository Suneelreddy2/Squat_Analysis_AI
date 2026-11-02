import os
import shutil
import subprocess
import logging
from typing import List

logger = logging.getLogger(__name__)


def _valid_output(path: str) -> bool:
    return os.path.isfile(path) and os.path.getsize(path) > 1024


def _ffmpeg_candidates() -> List[str]:
    candidates = []
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe:
            candidates.append(exe)
    except Exception as exc:
        logger.debug("imageio-ffmpeg is unavailable: %s", exc)

    for exe in ("ffmpeg", "ffmpeg.exe"):
        resolved = shutil.which(exe)
        if resolved:
            candidates.append(resolved)

    # Preserve order while removing duplicates.
    return list(dict.fromkeys(candidates))


def transcode_to_h264(input_path: str, output_path: str) -> bool:
    """Create a browser-friendly H.264/yuv420p MP4. Returns False on failure."""
    if not os.path.isfile(input_path) or os.path.getsize(input_path) == 0:
        logger.warning("Input video is missing or empty: %s", input_path)
        return False

    output_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(output_dir, exist_ok=True)

    # FFmpeg first: it handles timestamps, audio, and MP4 fast-start reliably.
    for exe in _ffmpeg_candidates():
        try:
            if os.path.exists(output_path):
                os.remove(output_path)

            cmd = [
                exe, "-y", "-i", input_path,
                "-map", "0:v:0", "-map", "0:a?",
                "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "128k",
                "-movflags", "+faststart",
                "-f", "mp4", output_path,
            ]
            result = subprocess.run(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, timeout=300
            )
            if result.returncode == 0 and _valid_output(output_path):
                logger.info("H.264 transcoding succeeded with %s", exe)
                return True

            logger.warning(
                "FFmpeg transcoding failed with %s: %s",
                exe, (result.stderr or "")[-1500:]
            )
        except Exception as exc:
            logger.warning("FFmpeg attempt failed (%s): %s", exe, exc)

    # PyAV fallback for environments where a usable FFmpeg executable is absent.
    try:
        import av

        if os.path.exists(output_path):
            os.remove(output_path)

        with av.open(input_path) as input_container:
            if not input_container.streams.video:
                raise ValueError("No video stream found")

            input_stream = input_container.streams.video[0]
            rate = input_stream.average_rate
            if rate is None or float(rate) <= 0:
                rate = 30

            width = input_stream.codec_context.width
            height = input_stream.codec_context.height
            width -= width % 2
            height -= height % 2
            if width <= 0 or height <= 0:
                raise ValueError("Invalid input video dimensions")

            with av.open(
                output_path, mode="w", format="mp4",
                options={"movflags": "+faststart"}
            ) as output_container:
                output_stream = output_container.add_stream("libx264", rate=rate)
                output_stream.width = width
                output_stream.height = height
                output_stream.pix_fmt = "yuv420p"
                output_stream.options = {"preset": "ultrafast", "crf": "23"}

                for frame in input_container.decode(input_stream):
                    converted = frame.reformat(
                        width=width, height=height, format="yuv420p"
                    )
                    for packet in output_stream.encode(converted):
                        output_container.mux(packet)

                for packet in output_stream.encode():
                    output_container.mux(packet)

        if _valid_output(output_path):
            logger.info("H.264 transcoding succeeded with PyAV")
            return True

    except Exception as exc:
        logger.exception("PyAV transcoding failed: %s", exc)

    try:
        if os.path.exists(output_path):
            os.remove(output_path)
    except OSError:
        pass
