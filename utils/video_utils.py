"""
utils/video_utils.py
~~~~~~~~~~~~~~~~~~~~~
FFmpeg-based video utility functions for the dubbing pipeline.

Provides:
  - Replacing audio track in a video file
  - Extracting video frames as images
  - Getting video metadata
  - Converting frame sequences back to video

Dependencies: ffmpeg (system binary), subprocess
"""

import os
import json
import logging
import subprocess
import glob
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _ffmpeg(cmd: list[str], step_name: str = "ffmpeg") -> str:
    """Run an ffmpeg command and return stdout. Raises on failure."""
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"FFmpeg failed during '{step_name}':\n{result.stderr}"
        )
    return result.stdout


def replace_audio(
    video_path: str,
    audio_path: str,
    output_path: str,
    audio_sample_rate: int = 44100,
) -> str:
    """
    Replace the audio track of a video with a new audio file.

    The video track is copied without re-encoding (fast). The audio is
    re-encoded to AAC for broad compatibility.

    Args:
        video_path        : Input video file (MP4)
        audio_path        : New audio file (WAV/MP3)
        output_path       : Output MP4 with replaced audio
        audio_sample_rate : Sample rate for output audio encoding

    Returns:
        Path to output video
    """
    os.makedirs(Path(output_path).parent, exist_ok=True)
    logger.info(f"[video_utils] Replacing audio: {audio_path} → {video_path}")

    _ffmpeg(
        [
            "ffmpeg", "-y",
            "-i",  video_path,
            "-i",  audio_path,
            "-map", "0:v:0",    # video from first input
            "-map", "1:a:0",    # audio from second input
            "-c:v", "copy",     # copy video stream (no re-encode)
            "-c:a", "aac",      # encode audio as AAC
            "-ar", str(audio_sample_rate),
            "-shortest",        # trim to shorter stream
            output_path,
        ],
        step_name="replace audio",
    )
    logger.info(f"[video_utils] Video with new audio → {output_path}")
    return output_path


def get_video_duration(video_path: str) -> float:
    """
    Return the duration of a video file in seconds.

    Args:
        video_path : Path to video file

    Returns:
        Duration as float (seconds)
    """
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "json",
            video_path,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")

    data = json.loads(result.stdout)
    return float(data["format"]["duration"])


def get_video_fps(video_path: str) -> float:
    """
    Return the frame rate of a video file.

    Returns:
        Frames per second as float
    """
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate",
            "-of", "json",
            video_path,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")

    data = json.loads(result.stdout)
    fps_str = data["streams"][0]["r_frame_rate"]
    num, den = fps_str.split("/")
    return round(int(num) / int(den), 4)


def extract_frames(
    video_path: str,
    output_dir: str,
    fps: Optional[float] = None,
    quality: int = 2,           # ffmpeg -q:v quality (1=best, 31=worst)
) -> list[str]:
    """
    Extract all (or sampled) frames from a video file as PNG images.

    Args:
        video_path : Input video
        output_dir : Directory to write frame PNGs
        fps        : If None, extract every frame. If specified, sample at this rate.
        quality    : JPEG/PNG quality (lower = better quality)

    Returns:
        Sorted list of extracted frame file paths
    """
    os.makedirs(output_dir, exist_ok=True)
    pattern = os.path.join(output_dir, "frame_%06d.png")

    cmd = ["ffmpeg", "-y", "-i", video_path]
    if fps is not None:
        cmd += ["-vf", f"fps={fps}"]
    cmd += ["-q:v", str(quality), pattern]

    _ffmpeg(cmd, step_name="extract frames")
    frames = sorted(glob.glob(os.path.join(output_dir, "*.png")))
    logger.info(f"[video_utils] Extracted {len(frames)} frames → {output_dir}")
    return frames


def frames_to_video(
    frames_dir: str,
    audio_path: str,
    output_path: str,
    fps: float = 25.0,
    crf: int = 17,
) -> str:
    """
    Re-encode a directory of sequentially-named PNG frames into a video.

    Args:
        frames_dir  : Directory containing frame_000001.png, etc.
        audio_path  : Audio track to include (WAV/MP3)
        output_path : Output MP4 path
        fps         : Output frame rate
        crf         : H.264 quality (lower = better; 17 ≈ visually lossless)

    Returns:
        Path to output video
    """
    os.makedirs(Path(output_path).parent, exist_ok=True)
    pattern = os.path.join(frames_dir, "frame_%06d.png")

    _ffmpeg(
        [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i",  pattern,
            "-i",  audio_path,
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf",  str(crf),
            "-c:a", "aac",
            "-shortest",
            output_path,
        ],
        step_name="frames to video",
    )
    logger.info(f"[video_utils] Video re-encoded → {output_path}")
    return output_path


def trim_video(
    video_path: str,
    start_sec: float,
    end_sec: float,
    output_path: str,
) -> str:
    """
    Trim a video to [start_sec, end_sec] using stream copy (no re-encode).
    Slightly less frame-accurate than re-encode but much faster.

    Returns:
        Path to trimmed video
    """
    _ffmpeg(
        [
            "ffmpeg", "-y",
            "-ss", str(start_sec),
            "-i",  video_path,
            "-t",  str(end_sec - start_sec),
            "-c", "copy",
            output_path,
        ],
        step_name="trim video",
    )
    logger.info(f"[video_utils] Trimmed video → {output_path}")
    return output_path


def add_subtitles(
    video_path: str,
    srt_path: str,
    output_path: str,
    font_size: int = 24,
    font_color: str = "white",
) -> str:
    """
    Burn hard subtitles into the video using the subtitles filter.
    Useful for adding Hindi text captions to the final output.

    Args:
        video_path : Input video
        srt_path   : Path to .srt subtitle file
        output_path: Output video with burned-in subtitles
        font_size  : Subtitle font size
        font_color : Subtitle font color

    Returns:
        Path to output video
    """
    subtitle_filter = (
        f"subtitles={srt_path}:force_style="
        f"'FontSize={font_size},PrimaryColour=&H{_color_to_ass(font_color)}'"
    )
    _ffmpeg(
        [
            "ffmpeg", "-y",
            "-i",  video_path,
            "-vf", subtitle_filter,
            "-c:a", "copy",
            output_path,
        ],
        step_name="add subtitles",
    )
    logger.info(f"[video_utils] Subtitled video → {output_path}")
    return output_path


def _color_to_ass(color: str) -> str:
    """Convert color name to ASS hex format (AABBGGRR)."""
    color_map = {
        "white":  "00FFFFFF",
        "yellow": "0000FFFF",
        "cyan":   "00FFFF00",
    }
    return color_map.get(color.lower(), "00FFFFFF")


def generate_srt(
    segments: list,  # TranslatedSegment objects
    output_path: str,
    use_hindi: bool = True,
) -> str:
    """
    Generate an SRT subtitle file from translated segments.

    Args:
        segments    : List of TranslatedSegment objects
        output_path : Path to write the .srt file
        use_hindi   : If True, write Hindi text; otherwise write English

    Returns:
        Path to SRT file
    """
    def _ts(sec: float) -> str:
        h = int(sec // 3600)
        m = int((sec % 3600) // 60)
        s = int(sec % 60)
        ms = int((sec - int(sec)) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    lines: list[str] = []
    for i, seg in enumerate(segments, start=1):
        text = seg.hindi if use_hindi else seg.english
        lines.append(str(i))
        lines.append(f"{_ts(seg.start)} --> {_ts(seg.end)}")
        lines.append(text)
        lines.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info(f"[video_utils] SRT file written → {output_path}")
    return output_path
