"""
config.py
~~~~~~~~~
Centralized configuration constants for the Hindi Video Dubbing Pipeline.

All default values and model identifiers live here so they are easy to
audit and change in one place.
"""

# ─── Whisper ──────────────────────────────────────────────────────────────────

WHISPER_DEFAULT_MODEL   = "base"
WHISPER_LANGUAGES       = "en"         # source language
WHISPER_WORD_TIMESTAMPS = True         # enable word-level alignment

# ─── MarianMT / Translation ───────────────────────────────────────────────────

MARIAN_MODEL_NAME = "Helsinki-NLP/opus-mt-en-hi"
TRANSLATION_BATCH_SIZE = 8             # segments per batch

# ─── Coqui XTTS v2 ───────────────────────────────────────────────────────────

XTTS_MODEL_NAME    = "tts_models/multilingual/multi-dataset/xtts_v2"
XTTS_LANGUAGE      = "hi"             # Hindi
XTTS_REF_CLIP_SEC  = 5.0             # length of reference clip for voice cloning
XTTS_REF_SAMPLE_RATE = 22050          # XTTS expects 22.05 kHz reference audio

# ─── Audio Time-Stretching ────────────────────────────────────────────────────

STRETCH_MIN_RATE   = 0.4              # slowest stretch ratio (below this = too slow)
STRETCH_MAX_RATE   = 2.5              # fastest stretch ratio (above this = too fast)
STRETCH_TOLERANCE  = 0.05            # seconds tolerance before stretching is applied

# ─── Wav2Lip ─────────────────────────────────────────────────────────────────

WAV2LIP_REPO_URL   = "https://github.com/Rudrabha/Wav2Lip.git"
WAV2LIP_GDRIVE_ID  = "1HCNBBsKKV8Ij2QMqzn5BLTJmJPH1VZDO"
WAV2LIP_MODEL_FILE = "wav2lip_gan.pth"

# ─── VideoReTalking ───────────────────────────────────────────────────────────

VIDEORETALKING_REPO_URL = "https://github.com/OpenTalker/video-retalking.git"

# ─── GFPGAN ───────────────────────────────────────────────────────────────────

GFPGAN_MODEL_URL  = (
    "https://github.com/TencentARC/GFPGAN/releases/download/"
    "v1.3.0/GFPGANv1.4.pth"
)
GFPGAN_MODEL_FILE = "GFPGANv1.4.pth"

# ─── FFmpeg / Extraction ─────────────────────────────────────────────────────

EXTRACT_VIDEO_CRF  = 18              # visually lossless quality
EXTRACT_AUDIO_RATE = 16000           # 16 kHz mono — Whisper optimal
ENCODE_OUTPUT_CRF  = 17              # slight improvement vs. extraction
ENCODE_PRESET      = "fast"

# ─── Silence Splitting ───────────────────────────────────────────────────────

SILENCE_THRESH_DB     = -40          # dBFS below which is considered silence
MIN_SILENCE_LEN_MS    = 500          # min silence gap to split on (ms)
KEEP_SILENCE_MS       = 300          # buffer kept at chunk boundaries (ms)
MIN_CHUNK_DURATION_S  = 1.0          # discard noise chunks shorter than this
