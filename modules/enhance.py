"""
modules/enhance.py
~~~~~~~~~~~~~~~~~~
Applies face restoration to the lip-synced video to eliminate the
blurriness artifact that Wav2Lip commonly introduces.

Models:
  - GFPGAN v1.4  : Fast, lightweight, great for most cases
  - CodeFormer   : Slightly slower but fidelity-aware (controllable quality/fidelity tradeoff)

Both are free and run on CPU (slow) or GPU (fast on Colab).

Workflow:
  1. Extract all frames from the lip-synced video
  2. Detect face region in each frame
  3. Run super-resolution / restoration on face crop
  4. Paste restored face back into the full frame
  5. Re-encode frames into video with original audio

Dependencies: facexlib, gfpgan, basicsr (for CodeFormer), opencv-python, imageio
"""

import os
import logging
import subprocess
import glob
import shutil
import tempfile
from pathlib import Path
from typing import Optional, Literal

import cv2  # type: ignore

logger = logging.getLogger(__name__)

# GFPGAN model weight download URL (official release)
_GFPGAN_MODEL_URL = (
    "https://github.com/TencentARC/GFPGAN/releases/download/"
    "v1.3.0/GFPGANv1.4.pth"
)

EnhanceModel = Literal["gfpgan", "codeformer", "none"]


# ─── GFPGAN Enhancement ──────────────────────────────────────────────────────

def _ensure_gfpgan(models_dir: str) -> str:
    """
    Download GFPGAN v1.4 weights if not already present.

    Returns:
        Path to the GFPGAN model file (.pth)
    """
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "GFPGANv1.4.pth")

    if not os.path.exists(model_path):
        logger.info(f"[enhance] Downloading GFPGAN v1.4 weights ...")
        result = subprocess.run(
            ["wget", "-q", "-O", model_path, _GFPGAN_MODEL_URL],
            capture_output=True
        )
        if result.returncode != 0:
            # fallback: try with curl
            subprocess.run(
                ["curl", "-L", "-o", model_path, _GFPGAN_MODEL_URL],
                check=True
            )
        logger.info(f"[enhance] GFPGAN weights saved → {model_path}")
    else:
        logger.info(f"[enhance] GFPGAN weights cached: {model_path}")

    return model_path


def enhance_frame_gfpgan(
    frame: "np.ndarray",
    restorer,
) -> "np.ndarray":
    """
    Enhance a single BGR frame using the GFPGAN restorer.

    Args:
        frame    : BGR image (numpy array) from cv2.imread
        restorer : Initialized GFPGANer instance

    Returns:
        Enhanced BGR frame, or original if no face detected
    """
    import numpy as np

    _, _, output = restorer.enhance(
        frame,
        has_aligned=False,   # GFPGAN will detect and align face itself
        only_center_face=False,
        paste_back=True,     # paste restored face back into full frame
    )

    return output if output is not None else frame


def enhance_video(
    video_path: str,
    output_path: str,
    model: EnhanceModel = "gfpgan",
    models_dir: str = "/tmp/enhance_models",
    upscale: int = 1,   # 1 = no upscale, 2 = 2x resolution
    fidelity_weight: float = 0.5,  # CodeFormer only: 0=quality, 1=fidelity
) -> str:
    """
    Apply face restoration to every frame of the lip-synced video.

    Args:
        video_path       : Input lip-synced video (MP4)
        output_path      : Output path for enhanced video (MP4)
        model            : Enhancement model: "gfpgan" | "codeformer" | "none"
        models_dir       : Directory to store model weights
        upscale          : Output resolution multiplier (1 recommended for speed)
        fidelity_weight  : CodeFormer fidelity weight (ignored for GFPGAN)

    Returns:
        Path to enhanced output video
    """
    if model == "none":
        logger.info("[enhance] Skipping face enhancement (model='none')")
        shutil.copy(video_path, output_path)
        return output_path

    video_path  = str(Path(video_path).resolve())
    output_path = str(Path(output_path).resolve())
    os.makedirs(Path(output_path).parent, exist_ok=True)

    if model == "gfpgan":
        return _enhance_with_gfpgan(
            video_path, output_path, models_dir, upscale
        )
    elif model == "codeformer":
        return _enhance_with_codeformer(
            video_path, output_path, models_dir, fidelity_weight, upscale
        )
    else:
        raise ValueError(f"Unknown enhancement model: {model!r}")


def _enhance_with_gfpgan(
    video_path: str,
    output_path: str,
    models_dir: str,
    upscale: int,
) -> str:
    """
    Run GFPGAN frame-by-frame on the video and re-encode with original audio.
    """
    import numpy as np

    try:
        from gfpgan import GFPGANer
    except ImportError:
        raise ImportError(
            "gfpgan is not installed. Run: pip install gfpgan"
        )

    model_path = _ensure_gfpgan(models_dir)

    logger.info("[enhance] Initialising GFPGAN restorer ...")
    restorer = GFPGANer(
        model_path=model_path,
        upscale=upscale,
        arch="clean",
        channel_multiplier=2,
        bg_upsampler=None,  # set to RealESRGAN for background upscaling (optional)
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        frames_dir    = os.path.join(tmpdir, "frames")
        enhanced_dir  = os.path.join(tmpdir, "enhanced")
        audio_path    = os.path.join(tmpdir, "audio.wav")
        os.makedirs(frames_dir,   exist_ok=True)
        os.makedirs(enhanced_dir, exist_ok=True)

        # ── Extract frames and audio ─────────────────────────────────────
        logger.info("[enhance] Extracting frames ...")
        subprocess.run(
            ["ffmpeg", "-y", "-i", video_path,
             os.path.join(frames_dir, "frame_%06d.png")],
            capture_output=True, check=True
        )
        subprocess.run(
            ["ffmpeg", "-y", "-i", video_path, "-vn",
             "-acodec", "pcm_s16le", audio_path],
            capture_output=True, check=True
        )

        frame_files = sorted(glob.glob(os.path.join(frames_dir, "*.png")))
        logger.info(f"[enhance] Processing {len(frame_files)} frames with GFPGAN ...")

        for i, fpath in enumerate(frame_files):
            frame = cv2.imread(fpath)
            if frame is None:
                continue
            enhanced = enhance_frame_gfpgan(frame, restorer)
            out_fpath = os.path.join(
                enhanced_dir, os.path.basename(fpath)
            )
            cv2.imwrite(out_fpath, enhanced)
            if (i + 1) % 30 == 0:
                logger.info(f"[enhance] {i+1}/{len(frame_files)} frames done")

        # ── Re-encode frames + original audio into final video ───────────
        logger.info("[enhance] Re-encoding enhanced frames → video ...")
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-framerate", "25",
                "-i", os.path.join(enhanced_dir, "frame_%06d.png"),
                "-i", audio_path,
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "17",
                "-c:a", "aac",
                "-shortest",
                output_path,
            ],
            capture_output=True, check=True
        )

    logger.info(f"[enhance] Enhanced video saved → {output_path}")
    return output_path


def _enhance_with_codeformer(
    video_path: str,
    output_path: str,
    models_dir: str,
    fidelity_weight: float,
    upscale: int,
) -> str:
    """
    Run CodeFormer via CLI on the video.
    CodeFormer provides better fidelity control than GFPGAN.

    Falls back to GFPGAN if CodeFormer is not available.
    """
    cf_dir = os.path.join(models_dir, "CodeFormer")

    try:
        if not os.path.exists(cf_dir):
            logger.info("[enhance] Cloning CodeFormer ...")
            subprocess.run(
                [
                    "git", "clone", "--depth", "1",
                    "https://github.com/sczhou/CodeFormer.git",
                    cf_dir,
                ],
                check=True
            )
            subprocess.run(
                ["pip", "install", "-q", "-r",
                 os.path.join(cf_dir, "requirements.txt")],
                check=True
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            frames_in  = os.path.join(tmpdir, "frames_in")
            frames_out = os.path.join(tmpdir, "frames_out")
            audio_path = os.path.join(tmpdir, "audio.wav")
            os.makedirs(frames_in, exist_ok=True)

            # Extract frames
            subprocess.run(
                ["ffmpeg", "-y", "-i", video_path,
                 os.path.join(frames_in, "frame_%06d.png")],
                capture_output=True, check=True
            )
            subprocess.run(
                ["ffmpeg", "-y", "-i", video_path, "-vn",
                 "-acodec", "pcm_s16le", audio_path],
                capture_output=True, check=True
            )

            # Run CodeFormer
            inference_script = os.path.join(cf_dir, "inference_codeformer.py")
            subprocess.run(
                [
                    "python", inference_script,
                    "-w", str(fidelity_weight),
                    "--upscale", str(upscale),
                    "--input_path", frames_in,
                    "--output_path", frames_out,
                ],
                cwd=cf_dir, check=True
            )

            restored_frames = sorted(
                glob.glob(os.path.join(frames_out, "**", "*.png"), recursive=True)
            )
            if not restored_frames:
                raise RuntimeError("CodeFormer produced no output frames")

            # Re-sequence frames into numbered names
            seq_dir = os.path.join(tmpdir, "seq")
            os.makedirs(seq_dir, exist_ok=True)
            for i, fp in enumerate(restored_frames):
                shutil.copy(fp, os.path.join(seq_dir, f"frame_{i+1:06d}.png"))

            subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-framerate", "25",
                    "-i", os.path.join(seq_dir, "frame_%06d.png"),
                    "-i", audio_path,
                    "-c:v", "libx264",
                    "-preset", "fast",
                    "-crf", "17",
                    "-c:a", "aac",
                    "-shortest",
                    output_path,
                ],
                capture_output=True, check=True
            )

        logger.info(f"[enhance] CodeFormer output → {output_path}")
        return output_path

    except Exception as e:
        logger.warning(f"[enhance] CodeFormer failed ({e}), falling back to GFPGAN")
        return _enhance_with_gfpgan(video_path, output_path, models_dir, upscale)
