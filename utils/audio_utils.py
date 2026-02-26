"""
utils/audio_utils.py
~~~~~~~~~~~~~~~~~~~~~
Audio processing utilities for the Hindi dubbing pipeline.

Provides:
  - Silence-based audio splitting (for scalable long-video processing)
  - Audio time-stretching (for duration matching)
  - Multi-chunk audio merging
  - Audio mixing utilities

Dependencies: librosa, pydub, soundfile, numpy
"""

import os
import logging
import glob
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ─── Silence-Based Splitting ─────────────────────────────────────────────────

def split_audio_on_silence(
    audio_path: str,
    output_dir: str,
    silence_thresh_db: int = -40,
    min_silence_len_ms: int = 500,
    keep_silence_ms: int = 300,
    min_chunk_duration_sec: float = 1.0,
) -> list[dict]:
    """
    Split an audio file into chunks at silence boundaries.

    This is the core scalability feature: for 500h+ of video, we split
    audio into utterance-level chunks, process them in parallel workers,
    and merge results. Each chunk is short enough to fit in GPU VRAM.

    Args:
        audio_path             : Path to input WAV file
        output_dir             : Directory to write chunk WAV files
        silence_thresh_db      : dBFS threshold for silence detection
        min_silence_len_ms     : Minimum silence duration to split on (ms)
        keep_silence_ms        : Silence buffer kept at chunk boundaries
        min_chunk_duration_sec : Discard chunks shorter than this (noise)

    Returns:
        List of dicts: {"index", "start_ms", "end_ms", "duration_ms", "path"}
    """
    try:
        from pydub import AudioSegment
        from pydub.silence import split_on_silence
    except ImportError:
        raise ImportError("pydub is required. Run: pip install pydub")

    os.makedirs(output_dir, exist_ok=True)
    audio_path = str(Path(audio_path).resolve())

    logger.info(f"[audio_utils] Loading audio: {audio_path}")
    audio = AudioSegment.from_file(audio_path)

    logger.info(
        f"[audio_utils] Splitting on silence "
        f"(thresh={silence_thresh_db}dBFS, min_len={min_silence_len_ms}ms)"
    )
    chunks = split_on_silence(
        audio,
        min_silence_len=min_silence_len_ms,
        silence_thresh=silence_thresh_db,
        keep_silence=keep_silence_ms,
        seek_step=10,
    )

    if not chunks:
        logger.warning("[audio_utils] No silence detected — returning full audio as one chunk")
        out_path = os.path.join(output_dir, "chunk_0000.wav")
        audio.export(out_path, format="wav",
                     parameters=["-ar", "16000", "-ac", "1"])
        return [{
            "index":       0,
            "start_ms":    0,
            "end_ms":      len(audio),
            "duration_ms": len(audio),
            "path":        out_path,
        }]

    results: list[dict] = []
    cursor_ms = 0

    for i, chunk in enumerate(chunks):
        dur_ms = len(chunk)
        dur_sec = dur_ms / 1000.0

        if dur_sec < min_chunk_duration_sec:
            cursor_ms += dur_ms
            continue  # skip very short noise chunks

        out_path = os.path.join(output_dir, f"chunk_{i:04d}.wav")
        chunk.export(out_path, format="wav",
                     parameters=["-ar", "16000", "-ac", "1"])

        results.append({
            "index":       i,
            "start_ms":    cursor_ms,
            "end_ms":      cursor_ms + dur_ms,
            "duration_ms": dur_ms,
            "path":        out_path,
        })
        cursor_ms += dur_ms

    logger.info(f"[audio_utils] Split into {len(results)} chunks")
    return results


# ─── Audio Duration Stretching ────────────────────────────────────────────────

def stretch_audio_to_duration(
    audio_path: str,
    target_duration_sec: float,
    output_path: str,
    preserve_pitch: bool = True,
) -> str:
    """
    Time-stretch audio to a target duration.

    Uses librosa phase-vocoder (pitch-preserving) by default.
    Falls back to simple speed change (via ffmpeg atempo) for large ratios.

    Args:
        audio_path          : Input WAV path
        target_duration_sec : Desired duration in seconds
        output_path         : Output WAV path
        preserve_pitch      : If True, use pitch-preserving stretch (librosa)
                              If False, use simple speed change (ffmpeg)

    Returns:
        Path to stretched audio file
    """
    try:
        import librosa
        import soundfile as sf
        import numpy as np
    except ImportError:
        raise ImportError(
            "librosa and soundfile are required. "
            "Run: pip install librosa soundfile"
        )

    y, sr = librosa.load(audio_path, sr=None)
    current_dur = len(y) / sr

    if abs(current_dur - target_duration_sec) < 0.05:
        logger.info("[audio_utils] Duration match within 50ms, no stretching")
        import shutil
        shutil.copy(audio_path, output_path)
        return output_path

    rate = current_dur / target_duration_sec
    rate_clamped = float(np.clip(rate, 0.4, 2.5))

    if rate_clamped != rate:
        logger.warning(
            f"[audio_utils] Stretch rate {rate:.3f} clamped to {rate_clamped:.3f} "
            "to prevent distortion. Duration will be approximate."
        )

    logger.info(
        f"[audio_utils] Stretching audio: {current_dur:.2f}s → {target_duration_sec:.2f}s "
        f"(rate={rate_clamped:.3f}, preserve_pitch={preserve_pitch})"
    )

    if preserve_pitch:
        y_out = librosa.effects.time_stretch(y, rate=rate_clamped)
    else:
        # Simple speed change via resampling — shifts pitch but is lossless
        new_sr = int(sr * rate_clamped)
        y_resampled = librosa.resample(y, orig_sr=sr, target_sr=new_sr)
        y_out = y_resampled

    sf.write(output_path, y_out, sr)
    logger.info(f"[audio_utils] Stretched audio → {output_path}")
    return output_path


# ─── Audio Merging ────────────────────────────────────────────────────────────

def merge_audio_chunks(
    chunk_paths: list[str],
    output_path: str,
    gap_between_chunks_ms: int = 0,
) -> str:
    """
    Concatenate a list of audio chunks into a single WAV file.

    Args:
        chunk_paths            : Ordered list of WAV file paths
        output_path            : Output concatenated WAV path
        gap_between_chunks_ms  : Silence gap to insert between chunks (ms)

    Returns:
        Path to merged audio file
    """
    try:
        from pydub import AudioSegment
    except ImportError:
        raise ImportError("pydub is required. Run: pip install pydub")

    if not chunk_paths:
        raise ValueError("[audio_utils] No chunks to merge")

    logger.info(f"[audio_utils] Merging {len(chunk_paths)} audio chunks ...")
    combined = AudioSegment.empty()
    silence_gap = AudioSegment.silent(duration=gap_between_chunks_ms) if gap_between_chunks_ms > 0 else None

    for i, path in enumerate(chunk_paths):
        chunk = AudioSegment.from_file(path)
        combined += chunk
        if silence_gap and i < len(chunk_paths) - 1:
            combined += silence_gap

    combined.export(output_path, format="wav",
                    parameters=["-ar", "16000", "-ac", "1"])
    logger.info(f"[audio_utils] Merged audio → {output_path} ({len(combined)/1000:.2f}s)")
    return output_path


# ─── Audio Mixing (Original + Dubbed) ────────────────────────────────────────

def mix_audio_tracks(
    background_path: str,
    foreground_path: str,
    output_path: str,
    bg_volume_db: float = -20.0,  # duck background by 20dB
) -> str:
    """
    Mix a background audio track with a foreground dubbed audio.
    Useful for preserving ambient sound while overlaying Hindi speech.

    Args:
        background_path : Original video audio (ducked)
        foreground_path : Dubbed Hindi audio (full volume)
        output_path     : Output mixed WAV
        bg_volume_db    : Background volume adjustment in dB (negative = quieter)

    Returns:
        Path to mixed audio
    """
    try:
        from pydub import AudioSegment
    except ImportError:
        raise ImportError("pydub is required. Run: pip install pydub")

    bg = AudioSegment.from_file(background_path) + bg_volume_db
    fg = AudioSegment.from_file(foreground_path)

    # Ensure same length
    if len(bg) > len(fg):
        bg = bg[:len(fg)]
    elif len(fg) > len(bg):
        silence = AudioSegment.silent(duration=len(fg) - len(bg))
        bg = bg + silence

    mixed = bg.overlay(fg)
    mixed.export(output_path, format="wav")
    logger.info(f"[audio_utils] Mixed audio → {output_path}")
    return output_path


def get_audio_duration(audio_path: str) -> float:
    """Return duration of an audio file in seconds."""
    try:
        import librosa
        y, sr = librosa.load(audio_path, sr=None)
        return len(y) / sr
    except Exception:
        from pydub import AudioSegment
        audio = AudioSegment.from_file(audio_path)
        return len(audio) / 1000.0
