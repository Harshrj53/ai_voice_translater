# 🎬 AI Hindi Video Dubbing Pipeline

A modular, open-source Python pipeline that takes any video in any language, extracts a 15-second segment, transcribes and translates it to Hindi, synthesizes Hindi speech, and produces a dubbed output video — all at **zero cost** using free/open-source models.

---

## 🚀 Features

| Step | Tool Used | Notes |
|------|-----------|-------|
| 🎞️ Video Extraction | FFmpeg | Frame-perfect cut with `dynaudnorm` audio normalization |
| 🎤 Transcription | OpenAI Whisper (`base`) | Auto-detects source language (Kannada, Hindi, Tamil, English…) |
| 🌐 Translation | MarianMT (`Helsinki-NLP/opus-mt-en-hi`) | English → Hindi via HuggingFace Transformers |
| 🔊 Text-to-Speech | Coqui XTTS v2 *(or gTTS fallback)* | Voice cloning with 3-10s reference clip |
| 🗣️ Lip Sync | Wav2Lip GAN | Requires GPU; skippable with `--skip-lipsync` |
| ✨ Enhancement | GFPGAN | Face restoration; skippable with `--enhance-model none` |

---

## 📁 Project Structure

```
ai_voice_translater/
├── dub_video.py           # Main pipeline entrypoint
├── config.py              # Centralized constants and model names
├── setup.py               # pip-installable package config
├── requirement.txt        # All Python dependencies
│
├── modules/
│   ├── extract.py         # FFmpeg video/audio extraction
│   ├── transcribe.py      # Whisper transcription & translation
│   ├── translate.py       # MarianMT English → Hindi translation
│   ├── tts.py             # Coqui XTTS v2 / gTTS speech synthesis
│   ├── lipsync.py         # Wav2Lip lip-sync integration
│   └── enhance.py         # GFPGAN face restoration
│
└── utils/
    ├── audio_utils.py     # Audio helpers (merge, stretch)
    └── video_utils.py     # Video helpers (replace audio, merge)
```

---

## ⚙️ Setup

### 1. Clone the repository

```bash
git clone https://github.com/Harshrj53/ai_voice_translater.git
cd ai_voice_translater
```

### 2. Create a virtual environment

```bash
python3 -m venv ml_env
source ml_env/bin/activate      # macOS / Linux
# ml_env\Scripts\activate       # Windows
```

### 3. Install dependencies

```bash
pip install -r requirement.txt
```

> **Python 3.13 Note:** Coqui TTS (`TTS`) and `gfpgan`/`basicsr` do not yet have Python 3.13 wheels. On Python 3.13, the pipeline automatically falls back to **gTTS** for speech synthesis and **skips** face enhancement. For full features, use Python 3.9–3.11 or [Google Colab](#-google-colab-recommended-for-gpu).

### 4. Install FFmpeg (system dependency)

```bash
# macOS
brew install ffmpeg

# Ubuntu / Debian
sudo apt install ffmpeg
```

---

## ▶️ Running the Pipeline

### Basic usage (audio dub, no lip-sync):

```bash
ml_env/bin/python3 dub_video.py \
  --input your_video.mp4 \
  --start 180 --end 195 \
  --output output_hindi.mp4 \
  --skip-lipsync \
  --enhance-model none
```

### Full pipeline (with lip-sync — requires GPU):

```bash
python3 dub_video.py \
  --input your_video.mp4 \
  --start 180 --end 195 \
  --output output_hindi_hq.mp4 \
  --gpu
```

### All CLI options:

| Flag | Default | Description |
|------|---------|-------------|
| `--input` | *(required)* | Path to input video |
| `--start` | `0` | Segment start time (seconds) |
| `--end` | `15` | Segment end time (seconds) |
| `--output` | `output_hindi.mp4` | Output video path |
| `--whisper-model` | `base` | Whisper model size: `tiny/base/small/medium/large-v3` |
| `--skip-lipsync` | `False` | Skip Wav2Lip (use when no GPU available) |
| `--enhance-model` | `gfpgan` | Face enhancement model (`none` to skip) |
| `--gpu` | `False` | Enable CUDA GPU for Whisper + XTTS + Wav2Lip |
| `--long-audio` | `False` | Use silence-based chunking for videos > 30 min |

---

## 🧠 Architecture

```
Input Video
    │
    ▼
┌─────────────┐
│  STEP 1     │  FFmpeg — cuts precise segment, extracts 16kHz mono WAV
│  Extract    │  dynaudnorm ensures consistent volume for Whisper
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  STEP 2     │  Whisper (base/small/large-v3)
│  Transcribe │  Auto-detects language → translates to English internally
└──────┬──────┘  (task="translate" supports Kannada, Tamil, Hindi, etc.)
       │
       ▼
┌─────────────┐
│  STEP 3     │  MarianMT (Helsinki-NLP/opus-mt-en-hi)
│  Translate  │  English → Hindi, context-aware, runs fully offline
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  STEP 4     │  Coqui XTTS v2 (primary) / gTTS (fallback)
│  TTS        │  Voice cloning from 5s reference clip, time-stretched to
└──────┬──────┘  match original segment duration for natural pacing
       │
       ▼
┌─────────────┐
│  STEP 5     │  Wav2Lip GAN
│  Lip Sync   │  Replaces speaker lip movements to match Hindi audio
└──────┬──────┘  (GPU strongly recommended; skippable on CPU)
       │
       ▼
┌─────────────┐
│  STEP 6     │  GFPGAN
│  Enhance    │  Restores face quality in lip-sync region
└──────┬──────┘  (requires Python ≤3.11; skippable on Python 3.13)
       │
       ▼
 Output Hindi MP4
```

---

## ☁️ Google Colab (Recommended for GPU)

Open [`colab_notebook.ipynb`](colab_notebook.ipynb) in Google Colab for a free T4 GPU environment with full support for:
- Coqui XTTS v2 voice cloning
- Wav2Lip lip-sync
- GFPGAN face enhancement

The Colab notebook includes all setup steps and can process the video end-to-end in ~2–3 minutes.

---

## 🔬 Testing

The pipeline was tested on a **Kannada-language training video** (source: Google Drive). Key findings:

- Whisper correctly auto-detected Kannada (`kn`) and translated speech to English
- MarianMT successfully translated 7 segments to Hindi in < 1 second
- gTTS generated Hindi audio (16.9s → stretched to 15s, rate=1.13x)
- Full pipeline ran in **~19 seconds** on MacBook (CPU, no GPU)

### Run on the test video:

```bash
# Download test video
ml_env/bin/python3 -m gdown "1uDzLVEow_gAJsXnNjbSoskzVLZ4d7opW" -O test_video.mp4

# Process 15-second segment (seconds 180–195 — where speech starts)
ml_env/bin/python3 dub_video.py \
  --input test_video.mp4 \
  --start 180 --end 195 \
  --output output_hindi.mp4 \
  --skip-lipsync --enhance-model none
```

---

## 📦 Scaling to Full-Length Videos

For processing full-length videos (30 min+), use the `--long-audio` flag which enables silence-based chunking:

```bash
python3 dub_video.py \
  --input full_lecture.mp4 \
  --output full_lecture_hindi.mp4 \
  --long-audio \
  --whisper-model small \
  --gpu
```

This uses `pydub` to split audio on silence boundaries, processes each chunk in parallel, and merges the results with corrected timestamps — suitable for overnight batch processing.

---

## 📋 Requirements

- Python 3.9–3.11 (for full feature set) or Python 3.13 (with TTS/lipsync/enhance auto-skipped)
- FFmpeg (system install)
- ~4 GB disk space for model weights (downloaded automatically on first run)
- GPU optional but strongly recommended for Wav2Lip and XTTS

---

## 📄 License

MIT License — free to use, modify, and distribute.
