"""
modules/transcribe.py
~~~~~~~~~~~~~~~~~~~~~
Transcribes English audio using OpenAI Whisper.

Features:
  - Segment-level and word-level timestamps
  - Silence-aware batching for long audio files (scalable to 500h+)
  - Configurable model size (base → large-v3 on GPU)

Dependencies: openai-whisper, pydub (for silence splitting)
"""

import os
import logging
import json
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ─── Data Structures ─────────────────────────────────────────────────────────

class Segment:
    """A transcribed segment with timing and text."""

    def __init__(self, start: float, end: float, text: str,
                 words: Optional[list] = None):
        self.start = start      # seconds
        self.end   = end        # seconds
        self.text  = text.strip()
        self.words = words or []  # list of {"word", "start", "end"}

    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> dict:
        return {
            "start": self.start,
            "end":   self.end,
            "text":  self.text,
            "words": self.words,
        }

    def __repr__(self) -> str:
        return f"Segment({self.start:.2f}s → {self.end:.2f}s): {self.text!r}"


class TranscriptionResult:
    """Container for all transcription output."""

    def __init__(self, full_text: str, segments: list[Segment],
                 language: str = "en"):
        self.full_text = full_text
        self.segments  = segments
        self.language  = language

    def to_dict(self) -> dict:
        return {
            "language":  self.language,
            "full_text": self.full_text,
            "segments":  [s.to_dict() for s in self.segments],
        }

    def save_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        logger.info(f"[transcribe] Transcript saved → {path}")


# ─── Core Transcription ───────────────────────────────────────────────────────

def transcribe_audio(
    audio_path: str,
    model_size: str = "base",
    language: str = "en",
    output_dir: Optional[str] = None,
    use_word_timestamps: bool = True,
) -> TranscriptionResult:
    """
    Transcribe audio using Whisper.

    Args:
        audio_path          : Path to WAV/MP3 audio file
        model_size          : Whisper model size: tiny/base/small/medium/large-v3
                              (base is default — fast on CPU, good accuracy)
        language            : Source language code (e.g. "en")
        output_dir          : If provided, saves transcript.json to this directory
        use_word_timestamps : Enable word-level timestamps (requires Whisper ≥ 20230918)

    Returns:
        TranscriptionResult with full text and timed segments
    """
    try:
        import whisper
    except ImportError:
        raise ImportError(
            "openai-whisper is not installed. Run: pip install openai-whisper"
        )

    audio_path = str(Path(audio_path).resolve())
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    logger.info(f"[transcribe] Loading Whisper model: {model_size}")
    model = whisper.load_model(model_size)

    logger.info(f"[transcribe] Transcribing: {audio_path}")
    result = model.transcribe(
        audio_path,
        language=language,
        word_timestamps=use_word_timestamps,
        verbose=False,
    )

    # ── Parse segments ────────────────────────────────────────────────────────
    segments: list[Segment] = []
    for raw_seg in result.get("segments", []):
        words: list[dict] = []
        if use_word_timestamps and "words" in raw_seg:
            for w in raw_seg["words"]:
                words.append({
                    "word":  w.get("word", "").strip(),
                    "start": w.get("start", 0.0),
                    "end":   w.get("end", 0.0),
                })
        segments.append(Segment(
            start=raw_seg["start"],
            end=raw_seg["end"],
            text=raw_seg["text"],
            words=words,
        ))

    full_text = result.get("text", "").strip()
    detected_language = result.get("language", language)

    logger.info(
        f"[transcribe] Done — {len(segments)} segments, "
        f"language detected: {detected_language}"
    )
    logger.info(f"[transcribe] Transcript: {full_text}")

    transcript = TranscriptionResult(full_text, segments, detected_language)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        json_path = os.path.join(output_dir, "transcript.json")
        transcript.save_json(json_path)

    return transcript


# ─── Scalable Batching Logic ─────────────────────────────────────────────────

def transcribe_long_audio(
    audio_path: str,
    model_size: str = "base",
    language: str = "en",
    silence_thresh_db: int = -40,
    min_silence_len_ms: int = 500,
    output_dir: Optional[str] = None,
) -> TranscriptionResult:
    """
    Transcribe a long audio file by first splitting it on silence boundaries.
    This is the scalable path for processing 500h+ of video overnight.

    Strategy:
      1. Detect silence gaps (≥ 500ms, ≤ -40 dBFS)
      2. Split audio into chunks at those boundaries
      3. Transcribe each chunk independently
      4. Merge results with corrected timestamps

    Args:
        silence_thresh_db   : RMS silence threshold in dBFS (default -40)
        min_silence_len_ms  : Minimum silence duration to split on (ms)

    Returns:
        Merged TranscriptionResult with global timestamps
    """
    try:
        from pydub import AudioSegment as PydubAudio
        from pydub.silence import split_on_silence
    except ImportError:
        raise ImportError(
            "pydub is not installed. Run: pip install pydub"
        )

    logger.info(f"[transcribe] Loading audio for silence-based chunking: {audio_path}")
    audio = PydubAudio.from_wav(audio_path)
    total_duration_ms = len(audio)

    chunks = split_on_silence(
        audio,
        min_silence_len=min_silence_len_ms,
        silence_thresh=silence_thresh_db,
        keep_silence=300,   # keep 300ms of silence at boundaries for naturalness
    )

    if not chunks:
        logger.warning("[transcribe] No silence boundaries found — transcribing as single chunk")
        return transcribe_audio(audio_path, model_size, language, output_dir)

    logger.info(f"[transcribe] Split into {len(chunks)} chunks for batch processing")

    # Write chunks to temp dir and transcribe each
    import tempfile
    merged_segments: list[Segment] = []
    merged_text_parts: list[str]   = []
    time_offset_ms = 0

    with tempfile.TemporaryDirectory() as tmpdir:
        for idx, chunk in enumerate(chunks):
            chunk_path = os.path.join(tmpdir, f"chunk_{idx:04d}.wav")
            chunk.export(chunk_path, format="wav",
                         parameters=["-ar", "16000", "-ac", "1"])

            chunk_result = transcribe_audio(
                chunk_path, model_size, language,
                use_word_timestamps=True
            )

            # Offset timestamps so they are relative to the original audio
            offset_sec = time_offset_ms / 1000.0
            for seg in chunk_result.segments:
                seg.start += offset_sec
                seg.end   += offset_sec
                for w in seg.words:
                    w["start"] += offset_sec
                    w["end"]   += offset_sec
                merged_segments.append(seg)

            if chunk_result.full_text:
                merged_text_parts.append(chunk_result.full_text)

            time_offset_ms += len(chunk) + min_silence_len_ms  # approximate

    full_text = " ".join(merged_text_parts)
    result = TranscriptionResult(full_text, merged_segments, language)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        result.save_json(os.path.join(output_dir, "transcript_batched.json"))

    logger.info(f"[transcribe] Batch transcription complete — {len(merged_segments)} total segments")
    return result
