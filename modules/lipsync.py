"""
modules/lipsync.py
~~~~~~~~~~~~~~~~~~
Lip-syncs a video to a new audio track using Wav2Lip.

Wav2Lip is the most Colab-friendly lip-sync model that:
  - Runs on CPU (slow) or free Colab T4 GPU (fast)
  - Produces crisp face region (not as blurry as early Wav2Lip)
  - Available as open-source with pretrained weights

VideoReTalking integration is also included as a higher-fidelity option
(requires slightly more VRAM — works on Colab Pro T4/V100).

Architecture:
  1. Download pretrained Wav2Lip weights (if not cached)
  2. Run inference: wav2lip/inference.py --face <video> --audio <wav>
  3. Merge lip-synced face back into original video (preserving background)

Dependencies: Wav2Lip repo (cloned), torch, face-alignment, opencv-python
"""

import os
import logging
import subprocess
import shutil
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Wav2Lip repository and model weight URLs
_WAV2LIP_REPO_URL = "https://github.com/Rudrabha/Wav2Lip.git"
_WAV2LIP_MODEL_URL = (
    "https://iiitaphyd-my.sharepoint.com/personal/radrabha_m_research_iiit_ac_in/"
    "_layouts/15/download.aspx?share=EdjI7bZlgApMqsVoEUUXpLsBxqXbn5z64TnOrYy_u1AkTg"
)
# Simplified wget-accessible mirror (via a public gdown link)
_WAV2LIP_GDRIVE_ID = "1HCNBBsKKV8Ij2QMqzn5BLTJmJPH1VZDO"

# VideoReTalking (higher fidelity alternative)
_VIDEORETALKING_REPO_URL = "https://github.com/OpenTalker/video-retalking.git"


def _ensure_wav2lip(base_dir: str) -> str:
    """
    Clone Wav2Lip repo and download pretrained weights if not already present.

    Returns:
        Path to the Wav2Lip repository root
    """
    wav2lip_dir = os.path.join(base_dir, "Wav2Lip")

    # ── Clone repo ────────────────────────────────────────────────────────
    if not os.path.exists(wav2lip_dir):
        logger.info(f"[lipsync] Cloning Wav2Lip → {wav2lip_dir}")
        subprocess.run(
            ["git", "clone", "--depth", "1", _WAV2LIP_REPO_URL, wav2lip_dir],
            check=True,
        )
    else:
        logger.info(f"[lipsync] Wav2Lip repo already exists: {wav2lip_dir}")

    # ── Download model weights ────────────────────────────────────────────
    checkpoint_dir = os.path.join(wav2lip_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    model_path = os.path.join(checkpoint_dir, "wav2lip_gan.pth")

    if not os.path.exists(model_path):
        logger.info("[lipsync] Downloading Wav2Lip GAN weights via gdown ...")
        try:
            import gdown
            gdown.download(id=_WAV2LIP_GDRIVE_ID, output=model_path, quiet=False)
        except ImportError:
            raise ImportError(
                "gdown is required to download model weights. "
                "Run: pip install gdown"
            )
        logger.info(f"[lipsync] Model weights saved → {model_path}")
    else:
        logger.info(f"[lipsync] Wav2Lip weights already cached: {model_path}")

    return wav2lip_dir


def run_lipsync(
    video_path: str,
    audio_path: str,
    output_path: str,
    models_dir: str = "/tmp/lipsync_models",
    use_gpu: bool = False,
    resize_factor: int = 1,
    pad_top: int = 0,
    pad_bottom: int = 10,
    pad_left: int = 0,
    pad_right: int = 0,
) -> str:
    """
    Apply Wav2Lip lip-sync: replace the speaker's lip movements to match
    the provided audio track.

    Args:
        video_path    : Input video (face must be clearly visible)
        audio_path    : Hindi dubbed audio (WAV, duration-matched)
        output_path   : Where to write the lip-synced output MP4
        models_dir    : Directory to store Wav2Lip repo and weights
        use_gpu       : Enable CUDA GPU (significantly faster)
        resize_factor : Downscale face region during inference (1 = no resize)
        pad_*         : Padding around face bounding box (helps include chin)

    Returns:
        Path to lip-synced output video
    """
    video_path  = str(Path(video_path).resolve())
    audio_path  = str(Path(audio_path).resolve())
    output_path = str(Path(output_path).resolve())

    os.makedirs(Path(output_path).parent, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    # ── Ensure Wav2Lip is set up ──────────────────────────────────────────
    wav2lip_dir = _ensure_wav2lip(models_dir)
    checkpoint  = os.path.join(wav2lip_dir, "checkpoints", "wav2lip_gan.pth")
    inference_script = os.path.join(wav2lip_dir, "inference.py")

    if not os.path.exists(inference_script):
        raise FileNotFoundError(
            f"Wav2Lip inference script not found: {inference_script}"
        )

    # ── Build Wav2Lip inference command ──────────────────────────────────
    cmd = [
        "python", inference_script,
        "--checkpoint_path", checkpoint,
        "--face",            video_path,
        "--audio",           audio_path,
        "--outfile",         output_path,
        "--resize_factor",   str(resize_factor),
        "--pads",
        str(pad_top), str(pad_bottom), str(pad_left), str(pad_right),
    ]
    if not use_gpu:
        cmd.append("--nosmooth")   # disable temporal smoothing on CPU for speed

    logger.info(f"[lipsync] Running Wav2Lip inference ...")
    logger.debug(f"[lipsync] Command: {' '.join(cmd)}")

    result = subprocess.run(
        cmd,
        cwd=wav2lip_dir,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        logger.error(f"[lipsync] Wav2Lip stderr:\n{result.stderr}")
        raise RuntimeError(f"Wav2Lip inference failed:\n{result.stderr[-2000:]}")

    if not os.path.exists(output_path):
        raise RuntimeError(
            f"Wav2Lip ran but output file not found: {output_path}\n"
            f"Wav2Lip stdout: {result.stdout}"
        )

    logger.info(f"[lipsync] Lip-synced video saved → {output_path}")
    return output_path


def run_lipsync_videoretalking(
    video_path: str,
    audio_path: str,
    output_path: str,
    models_dir: str = "/tmp/lipsync_models",
) -> str:
    """
    Higher-fidelity lip-sync using VideoReTalking.
    Requires more VRAM but produces natural head-pose handling and
    sharper output — recommended for submission if GPU is available.

    Falls back to run_lipsync (Wav2Lip) if VideoReTalking setup fails.

    Args:
        video_path  : Input video
        audio_path  : Hindi audio WAV
        output_path : Output MP4 path
        models_dir  : Directory to clone VideoReTalking into

    Returns:
        Path to lip-synced video
    """
    vr_dir = os.path.join(models_dir, "video-retalking")

    try:
        if not os.path.exists(vr_dir):
            logger.info("[lipsync] Cloning VideoReTalking ...")
            subprocess.run(
                ["git", "clone", "--depth", "1",
                 _VIDEORETALKING_REPO_URL, vr_dir],
                check=True,
            )

        # Install VideoReTalking requirements
        req_path = os.path.join(vr_dir, "requirements.txt")
        if os.path.exists(req_path):
            subprocess.run(
                ["pip", "install", "-q", "-r", req_path],
                check=True,
            )

        inference_script = os.path.join(vr_dir, "inference.py")
        cmd = [
            "python", inference_script,
            "--face",       video_path,
            "--audio",      audio_path,
            "--outfile",    output_path,
        ]
        logger.info("[lipsync] Running VideoReTalking inference ...")
        result = subprocess.run(cmd, cwd=vr_dir,
                                capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(result.stderr)

        logger.info(f"[lipsync] VideoReTalking output → {output_path}")
        return output_path

    except Exception as e:
        logger.warning(
            f"[lipsync] VideoReTalking failed ({e}); "
            "falling back to Wav2Lip"
        )
        return run_lipsync(video_path, audio_path, output_path, models_dir)
