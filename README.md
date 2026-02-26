# 🎬 Hindi Video Dubbing Pipeline
### Supernan AI Intern Challenge — "The Golden 15 Seconds"

A modular, zero-cost Python pipeline that takes an English training video and produces a high-quality Hindi-dubbed version with **voice cloning**, **accurate translation**, and **high-fidelity lip sync**.

---

## 📐 Architecture

```
Input Video (MP4)
       │
       ▼
┌─────────────────────────────────────────────────────────────────┐
│  modules/extract.py                                             │
│  FFmpeg  →  15s clip (MP4)  +  16kHz mono WAV  +  ref clip    │
└─────────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────┐
│  modules/transcribe.py                                          │
│  OpenAI Whisper (base → large-v3)                               │
│  Word-level timestamps  |  Silence-based batching (scalable)   │
└─────────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────┐
│  modules/translate.py                                           │
│  Helsinki-NLP MarianMT en→hi  (offline, free)                  │
│  Segment-aware (not word-by-word)  |  Google Translate fallback│
└─────────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────┐
│  modules/tts.py                                                 │
│  Coqui XTTS v2  (zero-shot voice cloning, Hindi support)        │
│  Duration matching via librosa time-stretch                     │
│  gTTS fallback (no voice cloning)                               │
└─────────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────┐
│  modules/lipsync.py                                             │
│  Wav2Lip (Colab-friendly)  |  VideoReTalking (high-fidelity)   │
│  Auto-downloads pretrained weights via gdown                    │
└─────────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────┐
│  modules/enhance.py                                             │
│  GFPGAN v1.4  (default)  |  CodeFormer (fidelity-controllable) │
│  Eliminates Wav2Lip face blur, frame-by-frame restoration       │
└─────────────────────────────────────────────────────────────────┘
       │
       ▼
  Output Video (Hindi Dubbed, Lip-Synced, Face-Restored MP4)
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- `ffmpeg` installed (`brew install ffmpeg` on macOS, `apt install ffmpeg` on Linux)
- CUDA GPU recommended (runs on CPU but slow)

### Installation

```bash
# Clone the repo
git clone https://github.com/Harshrj53/Supernan_video_dubbing_project.git
cd Supernan_video_dubbing_project/ai_voice_translater

# Create virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirement.txt
```

### Process the 15-Second Clip (0:15 – 0:30)

```bash
python dub_video.py \
  --input supernan_video.mp4 \
  --start 15 \
  --end 30 \
  --output output_hindi_15s.mp4
```

### Full Options

```bash
python dub_video.py --help
```

| Flag | Default | Description |
|------|---------|-------------|
| `--input` | required | Input video path |
| `--output` | required | Output video path |
| `--start` | 15.0 | Segment start (seconds) |
| `--end` | 30.0 | Segment end (seconds) |
| `--whisper-model` | base | `tiny/base/small/medium/large-v3` |
| `--lipsync-model` | wav2lip | `wav2lip/videoretalking` |
| `--enhance-model` | gfpgan | `gfpgan/codeformer/none` |
| `--gpu` | false | Enable CUDA GPU |
| `--long-audio` | false | Silence-based batching for long videos |
| `--skip-lipsync` | false | Audio dub only (faster) |
| `--add-subtitles` | false | Burn Hindi subtitles into output |
| `--hindi-text` | none | Provide Hindi text directly |
| `--dubbed-audio` | none | Provide dubbed WAV directly |

---

## 🎬 Google Colab (Recommended for GPU)

Open the included notebook:

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Harshrj53/Supernan_video_dubbing_project/blob/main/ai_voice_translater/colab_notebook.ipynb)

Or follow these steps in a new Colab notebook:

```python
# 1. Install dependencies
!apt install -y ffmpeg
!pip install -q openai-whisper transformers sentencepiece deep-translator \
    TTS gTTS librosa soundfile pydub opencv-python gfpgan gdown

# 2. Clone repo
!git clone https://github.com/Harshrj53/Supernan_video_dubbing_project.git
%cd Supernan_video_dubbing_project/ai_voice_translater

# 3. Upload your video
from google.colab import files
uploaded = files.upload()   # upload supernan_video.mp4

# 4. Run pipeline
!python dub_video.py --input supernan_video.mp4 \
    --start 15 --end 30 --output output.mp4 --gpu

# 5. Download output
files.download('output.mp4')
```

---

## 💰 Cost Analysis

### ₹0 Budget (This Project)
All models run locally / free-tier Colab:

| Component | Model | Cost |
|-----------|-------|------|
| Transcription | Whisper base (local) | ₹0 |
| Translation | Helsinki-NLP MarianMT (local) | ₹0 |
| TTS / Voice Clone | Coqui XTTS v2 (local) | ₹0 |
| Lip Sync | Wav2Lip (local) | ₹0 |
| Face Enhancement | GFPGAN (local) | ₹0 |
| Compute | Google Colab Free T4 | ₹0 |

### Estimated cost per minute at scale (500h+ batch on cloud)

| Component | Service | Cost/min of video |
|-----------|---------|-------------------|
| Transcription | Whisper on A100 (Lambda Labs) | ~₹1.2 |
| Translation | MarianMT (self-hosted) | ~₹0.5 |
| TTS | XTTS v2 (self-hosted) | ~₹2.0 |
| Lip Sync | Wav2Lip on A100 | ~₹8.0 |
| Enhancement | GFPGAN on A100 | ~₹3.0 |
| **Total** | | **~₹14.7/min** |

> For 500 hours = ~30,000 minutes → estimated **₹4,41,000** (~$5,400) using A100 spot instances on AWS/Lambda Labs.  
> With optimizations (batching, model quantization, caching), this can be reduced 40-60%.

---

## 🔧 Scaling to 500 Hours Overnight

The pipeline is designed for this scenario via:

1. **Silence-based audio chunking** (`transcribe_long_audio`): Splits audio on natural pauses, enabling parallel chunk-level processing
2. **`--workers N` flag**: Processes N video segments in parallel using `multiprocessing.Pool`
3. **Checkpointing**: Each step saves intermediates to `work_dir`, so failures resume from the last completed step
4. **Batch TTS synthesis**: `synthesize_segments()` processes all segments in a single model load
5. **GPU batching**: Wav2Lip and GFPGAN process multiple frames per forward pass

**Recommended overnight architecture:**
```
Input videos → S3 bucket
     ↓
AWS Batch / GCP Cloud Run (auto-scaled workers)
     ↓ each worker runs dub_video.py
     ↓
Output videos → S3 bucket
```

---

## ⚠️ Known Limitations

1. **Wav2Lip on CPU is slow**: ~5-10 mins per 15s clip. Use Colab T4 GPU for practical speeds (45s per clip).
2. **XTTS v2 voice cloning quality**: Requires at least 3 seconds of clean reference audio. Background music in the reference clip degrades quality.
3. **MarianMT translation**: Informal Hindi slang and compound words may be rendered too literally. A fine-tuned IndicTrans2 model would improve quality.
4. **Face detection failure**: Wav2Lip fails if the face is too small, occluded, or at extreme angles. The `resize_factor` and `pads` arguments help but don't fully solve this.
5. **Duration stretching limits**: Audio is clamped to 0.4x–2.5x stretch ratio. If the Hindi translation is very much longer/shorter than the English, a script rewrite with time constraints would be needed.
6. **No speaker diarization**: With multiple speakers, the pipeline dubs all speech with the primary speaker's voice cloned from the first 5 seconds.

---

## 🚀 What I'd Improve With More Time

1. **IndicTrans2**: Replace MarianMT with AI4Bharat's IndicTrans2 for significantly more natural Hindi (especially for instructional content like nanny training).
2. **Speaker diarization**: Add `pyannote.audio` to detect speaker boundaries, then clone each speaker's voice separately.
3. **Seamless voice cloning**: Use a 30-60 second reference clip with background music removed (via Demucs) for cleaner voice cloning.
4. **VideoReTalking by default**: With a proper GPU, VideoReTalking produces significantly sharper, more natural lip movements than Wav2Lip.
5. **Prosody transfer**: Match the emotional tone and pacing of the Hindi audio to the original English — currently, XTTS output can sound flat in comparison.
6. **A/B quality comparison UI**: A simple Gradio interface to compare original vs. dubbed videos side-by-side.

---

## 📁 Project Structure

```
ai_voice_translater/
├── dub_video.py          # Main pipeline orchestrator (CLI entry point)
├── requirement.txt       # All pip dependencies
├── colab_notebook.ipynb  # Google Colab end-to-end notebook
├── README.md             # This file
│
├── modules/
│   ├── extract.py        # FFmpeg-based video/audio segmentation
│   ├── transcribe.py     # Whisper transcription + silence batching
│   ├── translate.py      # MarianMT + Google Translate Hindi translation
│   ├── tts.py            # Coqui XTTS v2 voice cloning + duration stretch
│   ├── lipsync.py        # Wav2Lip + VideoReTalking lip-sync
│   └── enhance.py        # GFPGAN + CodeFormer face restoration
│
└── utils/
    ├── audio_utils.py    # Silence splitting, stretching, merging
    └── video_utils.py    # FFmpeg helpers, SRT generation
```
