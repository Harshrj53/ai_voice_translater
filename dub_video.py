"""
dub_video.py
~~~~~~~~~~~~
Main entry point for the Hindi Video Dubbing Pipeline.

Usage:
  python dub_video.py --input video.mp4 --start 15 --end 30 --output output.mp4 [OPTIONS]

Full pipeline:
  1. extract_segment   → 15s video clip + 16kHz WAV audio
  2. transcribe_audio  → English transcript with word-level timestamps
  3. translate_to_hindi→ Hindi translation (context-aware, segment-by-segment)
  4. synthesize_hindi_voice → Cloned Hindi audio (XTTS v2), duration-matched
  5. run_lipsync       → Wav2Lip / VideoReTalking lip sync
  6. enhance_video     → GFPGAN face restoration (removes Wav2Lip blur)

Scalability:
  - Long audio is split on silence boundaries before transcription
  - Each module is independently resumable (intermediate files cached to disk)
  - For 500h+ of video, swap serial for parallel execution using multiprocessing.Pool
    (see --help for --workers flag)

Author  : Harsh Raj
Project : Supernan AI Intern Challenge — The Golden 15 Seconds
Date    : 2026
"""

import os
import sys
import json
import time
import logging
import argparse
import shutil
from pathlib import Path

# ─── Logging setup ────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("dub_video")


# ─── Module imports ───────────────────────────────────────────────────────────

def _import_modules():
    """Lazy import all pipeline modules (avoids slowing --help with heavy deps)."""
    from modules.extract    import extract_segment, get_video_info
    from modules.transcribe import transcribe_audio, transcribe_long_audio
    from modules.translate  import translate_to_hindi
    from modules.tts        import synthesize_hindi_voice
    from modules.lipsync    import run_lipsync, run_lipsync_videoretalking
    from modules.enhance    import enhance_video
    from utils.video_utils  import (
        replace_audio, get_video_duration,
        generate_srt, add_subtitles,
    )
    return {
        "extract_segment":            extract_segment,
        "get_video_info":             get_video_info,
        "transcribe_audio":           transcribe_audio,
        "transcribe_long_audio":      transcribe_long_audio,
        "translate_to_hindi":         translate_to_hindi,
        "synthesize_hindi_voice":     synthesize_hindi_voice,
        "run_lipsync":                run_lipsync,
        "run_lipsync_videoretalking": run_lipsync_videoretalking,
        "enhance_video":              enhance_video,
        "replace_audio":              replace_audio,
        "get_video_duration":         get_video_duration,
        "generate_srt":               generate_srt,
        "add_subtitles":              add_subtitles,
    }


# ─── Pipeline Steps ───────────────────────────────────────────────────────────

def step_extract(m, args, work_dir: str) -> dict[str, str]:
    """Step 1: Extract the 15-second video/audio segment."""
    logger.info("=" * 60)
    logger.info("STEP 1: Extracting segment ...")
    t0 = time.time()
    paths = m["extract_segment"](
        video_path=args.input,
        start_sec=args.start,
        end_sec=args.end,
        output_dir=os.path.join(work_dir, "01_extract"),
        segment_name="segment",
    )
    logger.info(f"  ✓ Clip    : {paths['video']}")
    logger.info(f"  ✓ Audio   : {paths['audio']}")
    logger.info(f"  ✓ Ref Wav : {paths['ref_audio']}")
    logger.info(f"  ⏱  {time.time() - t0:.1f}s")
    return paths


def step_transcribe(m, args, audio_path: str, work_dir: str):
    """Step 2: Transcribe audio with Whisper (auto-detects language, translates to English)."""
    logger.info("=" * 60)
    logger.info("STEP 2: Transcribing with Whisper (translate → English) ...")
    t0 = time.time()
    out_dir = os.path.join(work_dir, "02_transcribe")

    src_lang = getattr(args, "src_lang", None)  # None = auto-detect

    if args.long_audio:
        result = m["transcribe_long_audio"](
            audio_path=audio_path,
            model_size=args.whisper_model,
            output_dir=out_dir,
        )
    else:
        result = m["transcribe_audio"](
            audio_path=audio_path,
            model_size=args.whisper_model,
            language=src_lang,     # None → auto-detect
            task="translate",      # always translate to English first
            output_dir=out_dir,
        )

    logger.info(f"  ✓ Detected language : {result.language}")
    logger.info(f"  ✓ Segments          : {len(result.segments)}")
    logger.info(f"  ✓ Full text         : {result.full_text[:120]}...")
    logger.info(f"  ⏱  {time.time() - t0:.1f}s")
    return result



def step_translate(m, args, transcript, work_dir: str):
    """Step 3: Translate English transcript segments to Hindi."""
    logger.info("=" * 60)
    logger.info("STEP 3: Translating to Hindi ...")
    out_dir = os.path.join(work_dir, "03_translate")
    os.makedirs(out_dir, exist_ok=True)

    hindi_full, translated_segs = m["translate_to_hindi"](
        transcript_segments=transcript.segments,
        full_text=transcript.full_text,
    )

    # Save translation to JSON
    trans_data = {
        "hindi_full_text": hindi_full,
        "segments": [ts.to_dict() for ts in translated_segs],
    }
    with open(os.path.join(out_dir, "translation.json"), "w", encoding="utf-8") as f:
        json.dump(trans_data, f, ensure_ascii=False, indent=2)

    logger.info(f"  ✓ Hindi: {hindi_full[:120]}...")
    return hindi_full, translated_segs


def step_tts(m, args, hindi_text: str, ref_audio: str,
             target_duration: float, work_dir: str) -> str:
    """Step 4: Synthesize Hindi speech (voice-cloned, duration-matched)."""
    logger.info("=" * 60)
    logger.info("STEP 4: Synthesizing Hindi audio (XTTS v2) ...")
    out_dir = os.path.join(work_dir, "04_tts")
    os.makedirs(out_dir, exist_ok=True)
    output_audio = os.path.join(out_dir, "hindi_dubbed.wav")

    m["synthesize_hindi_voice"](
        hindi_text=hindi_text,
        reference_audio=ref_audio,
        output_path=output_audio,
        target_duration_sec=target_duration,
        use_gpu=args.gpu,
        use_fallback=True,
    )
    logger.info(f"  ✓ Hindi audio → {output_audio}")
    return output_audio


def step_lipsync(m, args, clip_video: str, hindi_audio: str, work_dir: str) -> str:
    """Step 5: Apply lip-sync using Wav2Lip or VideoReTalking."""
    logger.info("=" * 60)
    logger.info(f"STEP 5: Lip-sync ({args.lipsync_model}) ...")
    out_dir = os.path.join(work_dir, "05_lipsync")
    os.makedirs(out_dir, exist_ok=True)
    output_video = os.path.join(out_dir, "lipsynced.mp4")
    models_dir   = os.path.join(work_dir, "models")

    if args.lipsync_model == "videoretalking":
        m["run_lipsync_videoretalking"](
            video_path=clip_video,
            audio_path=hindi_audio,
            output_path=output_video,
            models_dir=models_dir,
        )
    else:
        m["run_lipsync"](
            video_path=clip_video,
            audio_path=hindi_audio,
            output_path=output_video,
            models_dir=models_dir,
            use_gpu=args.gpu,
        )

    logger.info(f"  ✓ Lip-synced video → {output_video}")
    return output_video


def step_enhance(m, args, lipsynced_video: str, work_dir: str) -> str:
    """Step 6: Face restoration with GFPGAN / CodeFormer."""
    logger.info("=" * 60)
    logger.info(f"STEP 6: Face enhancement ({args.enhance_model}) ...")
    out_dir      = os.path.join(work_dir, "06_enhance")
    os.makedirs(out_dir, exist_ok=True)
    output_video = os.path.join(out_dir, "enhanced.mp4")
    models_dir   = os.path.join(work_dir, "models")

    m["enhance_video"](
        video_path=lipsynced_video,
        output_path=output_video,
        model=args.enhance_model,
        models_dir=models_dir,
    )
    logger.info(f"  ✓ Enhanced video → {output_video}")
    return output_video


# ─── Full Pipeline Orchestrator ──────────────────────────────────────────────

def run_pipeline(args) -> str:
    """
    Execute the full dubbing pipeline end-to-end.

    Returns:
        Path to the final output video file
    """
    # Validate inputs
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input video not found: {args.input}")
    if args.end <= args.start:
        raise ValueError(f"--end must be greater than --start")

    # Set up working directory
    work_dir = args.work_dir or os.path.join(
        os.path.dirname(os.path.abspath(args.output)),
        f"dub_workspace_{int(time.time())}",
    )
    os.makedirs(work_dir, exist_ok=True)
    logger.info(f"Working directory: {work_dir}")

    # Save run config
    config_path = os.path.join(work_dir, "run_config.json")
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2, default=str)

    # Lazy-import all modules
    logger.info("Loading pipeline modules ...")
    m = _import_modules()

    start_time = time.time()

    # ── Step 1: Extract ───────────────────────────────────────────────────
    paths = step_extract(m, args, work_dir)
    clip_video  = paths["video"]
    clip_audio  = paths["audio"]
    ref_audio   = paths["ref_audio"]
    target_duration = args.end - args.start

    # ── Step 2: Transcribe ────────────────────────────────────────────────
    if not args.skip_transcribe:
        transcript = step_transcribe(m, args, clip_audio, work_dir)
    else:
        logger.info("STEP 2: Skipped (--skip-transcribe)")
        transcript = None

    # ── Step 3: Translate ─────────────────────────────────────────────────
    if transcript and not args.skip_translate:
        hindi_full, translated_segs = step_translate(m, args, transcript, work_dir)
    elif args.hindi_text:
        logger.info("STEP 3: Using provided --hindi-text")
        hindi_full   = args.hindi_text
        translated_segs = []
    else:
        logger.info("STEP 3: Skipped (no transcript or --hindi-text provided)")
        hindi_full   = ""
        translated_segs = []

    # ── Step 4: TTS ───────────────────────────────────────────────────────
    if hindi_full and not args.skip_tts:
        hindi_audio = step_tts(
            m, args, hindi_full, ref_audio, target_duration, work_dir
        )
    elif args.dubbed_audio:
        logger.info(f"STEP 4: Using provided --dubbed-audio: {args.dubbed_audio}")
        hindi_audio = args.dubbed_audio
    else:
        logger.info("STEP 4: Skipped — using original audio")
        hindi_audio = clip_audio

    # ── Step 5: Lip Sync ──────────────────────────────────────────────────
    if not args.skip_lipsync:
        lipsynced_video = step_lipsync(m, args, clip_video, hindi_audio, work_dir)
    else:
        logger.info("STEP 5: Skipping lip-sync, using audio-replaced video ...")
        lipsynced_video = os.path.join(work_dir, "05_lipsync", "audio_replaced.mp4")
        os.makedirs(os.path.dirname(lipsynced_video), exist_ok=True)
        m["replace_audio"](clip_video, hindi_audio, lipsynced_video)

    # ── Step 6: Face Enhancement ──────────────────────────────────────────
    if args.enhance_model != "none":
        final_video = step_enhance(m, args, lipsynced_video, work_dir)
    else:
        logger.info("STEP 6: Skipping face enhancement (--enhance-model=none)")
        final_video = lipsynced_video

    # ── Optional: Add subtitles ───────────────────────────────────────────
    if args.add_subtitles and translated_segs:
        srt_path = os.path.join(work_dir, "subtitles.srt")
        m["generate_srt"](translated_segs, srt_path, use_hindi=True)
        subtitled_video = os.path.join(work_dir, "subtitled.mp4")
        m["add_subtitles"](final_video, srt_path, subtitled_video)
        final_video = subtitled_video

    # ── Copy to requested output path ─────────────────────────────────────
    os.makedirs(Path(args.output).parent, exist_ok=True)
    shutil.copy(final_video, args.output)

    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info(f"✅ PIPELINE COMPLETE in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    logger.info(f"✅ Output : {args.output}")
    logger.info(f"✅ Clip   : {args.start}s — {args.end}s ({args.end - args.start}s)")
    logger.info("=" * 60)

    return args.output


# ─── CLI ─────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Hindi Video Dubbing Pipeline — Supernan AI Intern Challenge",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process 0:15–0:30 of a video (default settings)
  python dub_video.py --input supernan.mp4 --start 15 --end 30 --output out_15s.mp4

  # Use VideoReTalking + CodeFormer for maximum quality (GPU required)
  python dub_video.py --input supernan.mp4 --start 15 --end 30 --output out_hq.mp4 \\
    --lipsync-model videoretalking --enhance-model codeformer --gpu

  # Skip lip-sync and enhancement (audio dub only, for quick testing)
  python dub_video.py --input supernan.mp4 --start 15 --end 30 --output out_audio.mp4 \\
    --skip-lipsync --enhance-model none

  # Use large Whisper model for better transcription accuracy
  python dub_video.py --input supernan.mp4 --start 15 --end 30 --output out.mp4 \\
    --whisper-model large-v3 --gpu

  # Process long video with silence-based batching
  python dub_video.py --input long_video.mp4 --start 0 --end 300 --output out.mp4 \\
    --long-audio --workers 4
        """,
    )

    # Input / output
    p.add_argument("--input",    required=True,  help="Path to source video (MP4/AVI/MOV)")
    p.add_argument("--output",   required=True,  help="Path to output video (MP4)")
    p.add_argument("--start",    type=float, default=15.0, help="Segment start (seconds)")
    p.add_argument("--end",      type=float, default=30.0, help="Segment end (seconds)")
    p.add_argument("--work-dir", default=None,
                   help="Working directory for intermediate files (default: auto)")

    # Model selection
    p.add_argument("--whisper-model",  default="base",
                   choices=["tiny", "base", "small", "medium", "large-v3"],
                   help="Whisper model size (default: base)")
    p.add_argument("--lipsync-model",  default="wav2lip",
                   choices=["wav2lip", "videoretalking"],
                   help="Lip-sync model (default: wav2lip)")
    p.add_argument("--enhance-model",  default="gfpgan",
                   choices=["gfpgan", "codeformer", "none"],
                   help="Face enhancement model (default: gfpgan)")

    # Compute
    p.add_argument("--gpu",       action="store_true",
                   help="Use CUDA GPU (recommended on Colab T4)")
    p.add_argument("--workers",   type=int, default=1,
                   help="Parallel workers for batch processing (default: 1)")
    p.add_argument("--long-audio", action="store_true",
                   help="Use silence-based batching for long audio (>10 min)")

    # Skip flags (for partial pipeline runs)
    p.add_argument("--skip-transcribe", action="store_true", help="Skip Whisper transcription")
    p.add_argument("--skip-translate",  action="store_true", help="Skip Hindi translation")
    p.add_argument("--skip-tts",        action="store_true", help="Skip TTS synthesis")
    p.add_argument("--skip-lipsync",    action="store_true", help="Skip lip-sync (audio dub only)")

    # Overrides
    p.add_argument("--hindi-text",   default=None,
                   help="Provide Hindi text directly (skips transcribe + translate)")
    p.add_argument("--dubbed-audio", default=None,
                   help="Provide dubbed audio WAV directly (skips TTS)")

    # Output options
    p.add_argument("--add-subtitles", action="store_true",
                   help="Burn Hindi subtitles into the output video")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Enable verbose DEBUG logging")

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        output = run_pipeline(args)
        print(f"\n✅ Done! Output video: {output}")
        sys.exit(0)
    except KeyboardInterrupt:
        logger.info("\nPipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n❌ Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
