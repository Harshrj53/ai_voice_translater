"""
modules/extract.py
~~~~~~~~~~~~~~~~~~
Handles video segment extraction and audio extraction from the source video.

Responsibilities:
  - Cut a precise time-bounded clip from the full video (e.g., 0:15 – 0:30)
  - Extract the audio track as a 16 kHz mono WAV (optimal for Whisper)
  - Return file paths for downstream modules

Dependencies: ffmpeg (system binary), ffmpeg-python (pip)
"""

import os
import subprocess
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _run_ffmpeg(cmd: list[str], step_name: str) -> None:
    """Run an FFmpeg command and raise a clear error on failure."""
    logger.info(f"[extract] Running FFmpeg step: {step_name}")
    logger.debug(f"[extract] Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"FFmpeg failed during '{step_name}':\n{result.stderr}"
        )


def extract_segment(
    video_path: str,
    start_sec: float,
    end_sec: float,
    output_dir: str,
    segment_name: str = "segment",
) -> dict[str, str]:
    """
    Cut a time-bounded clip from the source video and extract its audio.

    Args:
        video_path  : Absolute path to the source video (any format ffmpeg supports)
        start_sec   : Start time in seconds (e.g. 15.0 for 0:15)
        end_sec     : End time in seconds (e.g. 30.0 for 0:30)
        output_dir  : Directory where output files will be written
        segment_name: Base name prefix for output files

    Returns:
        dict with keys:
            "video"  -> path to the extracted video clip (MP4)
            "audio"  -> path to the extracted audio (16 kHz, mono WAV)
            "ref_audio" -> path to a short reference clip for voice cloning (WAV)
    """
    video_path = str(Path(video_path).resolve())
    output_dir = str(Path(output_dir).resolve())
    os.makedirs(output_dir, exist_ok=True)

    duration = end_sec - start_sec
    if duration <= 0:
        raise ValueError(f"end_sec ({end_sec}) must be greater than start_sec ({start_sec})")

    clip_video_path = os.path.join(output_dir, f"{segment_name}_clip.mp4")
    clip_audio_path = os.path.join(output_dir, f"{segment_name}_audio.wav")
    ref_audio_path  = os.path.join(output_dir, f"{segment_name}_ref.wav")

    # ── 1. Cut video segment (re-encode to ensure frame-perfect cut) ──────────
    _run_ffmpeg(
        [
            "ffmpeg", "-y",
            "-ss", str(start_sec),
            "-i", video_path,
            "-t", str(duration),
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "18",          # visually lossless quality
            "-c:a", "aac",
            "-ar", "44100",
            clip_video_path,
        ],
        "cut video segment",
    )
    logger.info(f"[extract] Segment video saved → {clip_video_path}")

    # ── 2. Extract audio as 16 kHz mono WAV (Whisper-optimal) ─────────────────
    _run_ffmpeg(
        [
            "ffmpeg", "-y",
            "-i", clip_video_path,
            "-vn",                  # no video
            "-acodec", "pcm_s16le", # 16-bit PCM
            "-ar", "16000",         # 16 kHz sample rate
            "-ac", "1",             # mono channel
            clip_audio_path,
        ],
        "extract 16kHz mono WAV",
    )
    logger.info(f"[extract] Audio WAV saved → {clip_audio_path}")

    # ── 3. Extract a reference audio clip for voice cloning ───────────────────
    # We grab the first 5s of the segment as the speaker reference for XTTS
    ref_duration = min(5.0, duration)
    _run_ffmpeg(
        [
            "ffmpeg", "-y",
            "-i", clip_audio_path,
            "-t", str(ref_duration),
            "-acodec", "pcm_s16le",
            "-ar", "22050",         # XTTS expects 22.05 kHz
            "-ac", "1",
            ref_audio_path,
        ],
        "extract reference audio for voice cloning",
    )
    logger.info(f"[extract] Reference audio saved → {ref_audio_path}")

    return {
        "video":     clip_video_path,
        "audio":     clip_audio_path,
        "ref_audio": ref_audio_path,
    }


def get_video_info(video_path: str) -> dict[str, float | int | str]:
    """
    Probe a video file and return basic metadata.

    Returns:
        dict with keys: duration, width, height, fps, audio_sample_rate
    """
    import json

    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_streams",
        "-show_format",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")

    data = json.loads(result.stdout)
    info: dict[str, float | int | str] = {}

    for stream in data.get("streams", []):
        if stream.get("codec_type") == "video":
            info["width"]  = int(stream.get("width", 0))
            info["height"] = int(stream.get("height", 0))
            fps_str = stream.get("r_frame_rate", "25/1")
            try:
                num, den = fps_str.split("/")
                info["fps"] = round(int(num) / int(den), 4)
            except (ValueError, ZeroDivisionError):
                info["fps"] = 25.0

        if stream.get("codec_type") == "audio":
            info["audio_sample_rate"] = int(stream.get("sample_rate", 16000))

    fmt = data.get("format", {})
    info["duration"] = float(fmt.get("duration", 0))

    return info
