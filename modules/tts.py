"""
modules/tts.py
~~~~~~~~~~~~~~
Generates Hindi speech using voice cloning to match the original speaker.

Primary  : Coqui XTTS v2 — free, offline, GPU-optional, multilingual
Fallback : gTTS (Google Text-to-Speech, no voice cloning, internet required)

Key responsibilities:
  - Clone speaker's voice from a short reference audio clip
  - Synthesize Hindi speech segment by segment
  - Time-stretch the output to match the original segment duration
    (so lips and audio stay in sync even after translation length changes)

Dependencies: TTS (coqui-ai), librosa, soundfile, gtts (fallback)
"""

import os
import logging
import tempfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Coqui XTTS v2 model identifier
_XTTS_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"


# ─── XTTS v2 Voice Cloning ───────────────────────────────────────────────────

class CoquiXTTS:
    """
    Wraps Coqui XTTS v2 for Hindi voice cloning.

    XTTS v2 supports zero-shot voice cloning: it only needs a 3-6 second
    reference clip of the original speaker to match tone, timbre, and pace.
    It natively supports Hindi (language code: "hi").
    """

    def __init__(self, use_gpu: bool = False):
        try:
            from TTS.api import TTS
        except ImportError:
            raise ImportError(
                "Coqui TTS is not installed. "
                "Run: pip install TTS"
            )

        logger.info(f"[tts] Loading Coqui XTTS v2 (gpu={use_gpu}) ...")
        self.tts = TTS(model_name=_XTTS_MODEL, gpu=use_gpu)
        logger.info("[tts] XTTS v2 model ready")

    def synthesize(
        self,
        text: str,
        reference_audio: str,
        output_path: str,
        language: str = "hi",
    ) -> str:
        """
        Synthesize speech from text using the reference speaker's voice.

        Args:
            text            : Hindi text to synthesize
            reference_audio : Path to 3-10s WAV clip of the original speaker
            output_path     : Path to write the output WAV file
            language        : Language code (default: "hi" = Hindi)

        Returns:
            Path to generated WAV file
        """
        if not text.strip():
            raise ValueError("[tts] Cannot synthesize empty text")

        logger.info(f"[tts] Synthesizing: {text[:60]}...")
        self.tts.tts_to_file(
            text=text,
            speaker_wav=reference_audio,
            language=language,
            file_path=output_path,
        )
        logger.info(f"[tts] Audio saved → {output_path}")
        return output_path


# ─── gTTS Fallback ────────────────────────────────────────────────────────────

class GTTSFallback:
    """
    Google Text-to-Speech fallback — no voice cloning, but supports Hindi.
    Requires internet. Free with no API key.
    """

    def synthesize(
        self,
        text: str,
        reference_audio: str,   # unused in fallback, kept for API parity
        output_path: str,
        language: str = "hi",
    ) -> str:
        try:
            from gtts import gTTS
        except ImportError:
            raise ImportError(
                "gTTS is not installed. Run: pip install gTTS"
            )

        logger.info(f"[tts] gTTS fallback — synthesizing Hindi ...")
        # gTTS writes MP3; we'll convert to WAV via ffmpeg
        mp3_path = output_path.replace(".wav", "_gtts.mp3")
        tts_obj = gTTS(text=text, lang=language, slow=False)
        tts_obj.save(mp3_path)

        # Convert MP3 → WAV
        import subprocess
        subprocess.run(
            ["ffmpeg", "-y", "-i", mp3_path, "-ar", "22050", "-ac", "1", output_path],
            check=True, capture_output=True
        )
        os.remove(mp3_path)
        logger.info(f"[tts] gTTS audio saved → {output_path}")
        return output_path


# ─── Audio Duration Matching ─────────────────────────────────────────────────

def stretch_to_duration(
    audio_path: str,
    target_duration_sec: float,
    output_path: str,
) -> str:
    """
    Time-stretch synthesized Hindi audio to match the target duration.

    Uses librosa's time-stretch (phase vocoder) which preserves pitch.
    This is critical for lip-sync: the Hindi audio must start and end
    at the same moments as the original English audio.

    Args:
        audio_path          : Input WAV path
        target_duration_sec : Desired output duration in seconds
        output_path         : Output WAV path

    Returns:
        Path to time-stretched WAV
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
    current_duration = len(y) / sr

    if abs(current_duration - target_duration_sec) < 0.05:
        logger.info("[tts] Audio duration already matches target; no stretching needed")
        import shutil
        shutil.copy(audio_path, output_path)
        return output_path

    stretch_rate = current_duration / target_duration_sec
    # librosa time_stretch: rate > 1 → speeds up, rate < 1 → slows down
    # Wider clamp (0.4x–2.5x) consistent with config.STRETCH_MIN/MAX_RATE.
    # Below 0.4x or above 2.5x the pitch-vocoder artifacts become audible.
    clamped = float(np.clip(stretch_rate, 0.4, 2.5))
    if clamped != stretch_rate:
        logger.warning(
            f"[tts] Stretch rate {stretch_rate:.3f}x clamped to {clamped:.3f}x — "
            "translation is very much longer/shorter than original. "
            "Consider a tighter translation for better sync."
        )
    stretch_rate = clamped

    logger.info(
        f"[tts] Stretching audio: {current_duration:.2f}s → {target_duration_sec:.2f}s "
        f"(rate={stretch_rate:.3f})"
    )
    y_stretched = librosa.effects.time_stretch(y, rate=stretch_rate)
    sf.write(output_path, y_stretched, sr)
    logger.info(f"[tts] Stretched audio saved → {output_path}")
    return output_path


# ─── Public API ──────────────────────────────────────────────────────────────

def synthesize_hindi_voice(
    hindi_text: str,
    reference_audio: str,
    output_path: str,
    target_duration_sec: Optional[float] = None,
    use_gpu: bool = False,
    use_fallback: bool = True,
) -> str:
    """
    Synthesize Hindi speech and optionally time-stretch to match original duration.

    Args:
        hindi_text          : Hindi text to speak
        reference_audio     : WAV clip of the original speaker (3-10 seconds)
        output_path         : Where to save the final audio WAV
        target_duration_sec : If provided, stretches audio to this duration
        use_gpu             : Use CUDA GPU for XTTS (much faster)
        use_fallback        : Fall back to gTTS if XTTS fails

    Returns:
        Path to the final WAV file (stretched if target_duration_sec given)
    """
    os.makedirs(Path(output_path).parent, exist_ok=True)

    raw_output = output_path.replace(".wav", "_raw.wav") if target_duration_sec else output_path

    # ── 1. Synthesize ─────────────────────────────────────────────────────────
    tts_engine = None
    try:
        tts_engine = CoquiXTTS(use_gpu=use_gpu)
        tts_engine.synthesize(hindi_text, reference_audio, raw_output)
    except Exception as e:
        logger.warning(f"[tts] Coqui XTTS failed: {e}")
        if use_fallback:
            logger.info("[tts] Switching to gTTS fallback ...")
            tts_engine = GTTSFallback()
            tts_engine.synthesize(hindi_text, reference_audio, raw_output)
        else:
            raise

    # ── 2. Time-stretch to match original duration ────────────────────────────
    if target_duration_sec and target_duration_sec > 0:
        stretch_to_duration(raw_output, target_duration_sec, output_path)
        if raw_output != output_path and os.path.exists(raw_output):
            os.remove(raw_output)
    else:
        if raw_output != output_path:
            import shutil
            shutil.move(raw_output, output_path)

    return output_path


def synthesize_segments(
    translated_segments: list,  # list of TranslatedSegment from translate.py
    reference_audio: str,
    output_dir: str,
    use_gpu: bool = False,
    use_fallback: bool = True,
) -> list[dict]:
    """
    Synthesize each segment individually, duration-matched to original.
    Useful for segment-level audio that feeds into lip-sync per utterance.

    Returns:
        List of dicts: {"start", "end", "audio_path"}
    """
    os.makedirs(output_dir, exist_ok=True)
    results: list[dict] = []

    for idx, seg in enumerate(translated_segments):
        out_path = os.path.join(output_dir, f"seg_{idx:04d}.wav")
        target_dur = seg.end - seg.start

        logger.info(
            f"[tts] Segment {idx}: {seg.hindi[:40]}... "
            f"(target {target_dur:.2f}s)"
        )

        synthesize_hindi_voice(
            hindi_text=seg.hindi,
            reference_audio=reference_audio,
            output_path=out_path,
            target_duration_sec=target_dur,
            use_gpu=use_gpu,
            use_fallback=use_fallback,
        )

        results.append({
            "start":      seg.start,
            "end":        seg.end,
            "audio_path": out_path,
        })

    return results
