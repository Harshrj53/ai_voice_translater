"""
Hindi Video Dubbing Pipeline — Modules Package
================================================
Each module is independently importable and testable.

Sub-modules:
  extract    — FFmpeg-based video/audio segment extraction
  transcribe — Whisper English transcription with word timestamps
  translate  — MarianMT (+ Google Translate fallback) Hindi translation
  tts        — Coqui XTTS v2 Hindi voice cloning + duration stretching
  lipsync    — Wav2Lip / VideoReTalking lip-sync
  enhance    — GFPGAN / CodeFormer face restoration
"""

__all__ = [
    "extract",
    "transcribe",
    "translate",
    "tts",
    "lipsync",
    "enhance",
]
