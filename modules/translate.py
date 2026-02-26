"""
modules/translate.py
~~~~~~~~~~~~~~~~~~~~~
Translates English transcript segments to contextual Hindi.

Strategy:
  - Primary  : Helsinki-NLP/opus-mt-en-hi (MarianMT) — 100% free, offline
  - Fallback : deep_translator Google Translate (free tier, internet required)

Context-awareness:
  - Translates segment-by-segment (not word-by-word) so sentences retain meaning
  - Batches segments to avoid repeated model load overhead

Dependencies: transformers, sentencepiece, deep-translator (fallback)
"""

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

# Helsinki-NLP MarianMT model for English → Hindi
_MARIAN_MODEL_NAME = "Helsinki-NLP/opus-mt-en-hi"


# ─── Data Structures ─────────────────────────────────────────────────────────

class TranslatedSegment:
    """An English segment paired with its Hindi translation."""

    def __init__(self, start: float, end: float,
                 english: str, hindi: str):
        self.start   = start
        self.end     = end
        self.english = english
        self.hindi   = hindi

    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> dict:
        return {
            "start":   self.start,
            "end":     self.end,
            "english": self.english,
            "hindi":   self.hindi,
        }

    def __repr__(self) -> str:
        return (
            f"TranslatedSegment({self.start:.2f}→{self.end:.2f}) "
            f"EN: {self.english!r} | HI: {self.hindi!r}"
        )


# ─── MarianMT Translator ──────────────────────────────────────────────────────

class MarianTranslator:
    """
    Wraps Helsinki-NLP opus-mt-en-hi for batch segment translation.
    Downloads the model once and caches it locally (~300MB).
    """

    def __init__(self, model_name: str = _MARIAN_MODEL_NAME):
        try:
            from transformers import MarianMTModel, MarianTokenizer
        except ImportError:
            raise ImportError(
                "transformers and sentencepiece are required. "
                "Run: pip install transformers sentencepiece"
            )

        logger.info(f"[translate] Loading MarianMT model: {model_name}")
        self.tokenizer = MarianTokenizer.from_pretrained(model_name)
        self.model     = MarianMTModel.from_pretrained(model_name)
        self.model.eval()
        logger.info("[translate] MarianMT model ready")

    def translate_batch(self, texts: list[str], max_length: int = 512) -> list[str]:
        """Translate a batch of English texts to Hindi."""
        if not texts:
            return []

        # MarianMT expects text to be wrapped for some models; keep as-is for en-hi
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

        import torch
        with torch.no_grad():
            translated_ids = self.model.generate(**inputs, max_length=max_length)

        translations = self.tokenizer.batch_decode(
            translated_ids, skip_special_tokens=True
        )
        return translations

    def translate(self, text: str) -> str:
        """Translate a single English string to Hindi."""
        results = self.translate_batch([text])
        return results[0] if results else ""


# ─── Google Translate Fallback ────────────────────────────────────────────────

class GoogleTranslateFallback:
    """Uses deep_translator (free, no API key) as a fallback."""

    def __init__(self):
        try:
            from deep_translator import GoogleTranslator
            self._translator = GoogleTranslator(source="en", target="hi")
            logger.info("[translate] Using Google Translate fallback")
        except ImportError:
            raise ImportError(
                "deep_translator is not installed. "
                "Run: pip install deep-translator"
            )

    def translate(self, text: str) -> str:
        try:
            return self._translator.translate(text) or text
        except Exception as e:
            logger.warning(f"[translate] Google Translate failed: {e}")
            return text

    def translate_batch(self, texts: list[str]) -> list[str]:
        return [self.translate(t) for t in texts]


# ─── Public API ──────────────────────────────────────────────────────────────

def translate_to_hindi(
    transcript_segments: list,   # list of Segment objects from transcribe.py
    full_text: str = "",
    use_fallback: bool = True,
    batch_size: int = 8,
) -> tuple[str, list[TranslatedSegment]]:
    """
    Translate English transcript segments to Hindi.

    Args:
        transcript_segments : List of Segment objects (from transcribe.py)
        full_text           : Full transcript text (for logging/sanity check)
        use_fallback        : Fall back to Google Translate if MarianMT fails
        batch_size          : Number of segments to translate per batch

    Returns:
        (hindi_full_text, list[TranslatedSegment])
        where each TranslatedSegment holds timing + EN + HI text
    """
    if not transcript_segments:
        logger.warning("[translate] No segments to translate")
        return "", []

    # ── Try MarianMT first ────────────────────────────────────────────────────
    translator = None
    try:
        translator = MarianTranslator()
    except Exception as e:
        logger.warning(f"[translate] MarianMT unavailable ({e}), using fallback")
        if not use_fallback:
            raise
        translator = GoogleTranslateFallback()

    # ── Batch-translate all segment texts ────────────────────────────────────
    texts = [seg.text for seg in transcript_segments]
    translated_texts: list[str] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        logger.info(
            f"[translate] Translating batch {i // batch_size + 1} "
            f"({len(batch)} segments) ..."
        )
        try:
            batch_translations = translator.translate_batch(batch)
            translated_texts.extend(batch_translations)
        except Exception as e:
            logger.error(f"[translate] Batch translation failed: {e}")
            if use_fallback and not isinstance(translator, GoogleTranslateFallback):
                logger.info("[translate] Retrying with Google Translate fallback")
                fb = GoogleTranslateFallback()
                translated_texts.extend(fb.translate_batch(batch))
            else:
                # Return original text if all else fails
                translated_texts.extend(batch)

    # ── Build TranslatedSegment list ─────────────────────────────────────────
    translated_segments: list[TranslatedSegment] = []
    for seg, hindi_text in zip(transcript_segments, translated_texts):
        ts = TranslatedSegment(
            start=seg.start,
            end=seg.end,
            english=seg.text,
            hindi=_clean_hindi(hindi_text),
        )
        translated_segments.append(ts)
        logger.debug(f"[translate] {ts}")

    hindi_full_text = " ".join(ts.hindi for ts in translated_segments)
    logger.info(f"[translate] Translation complete: {hindi_full_text}")

    return hindi_full_text, translated_segments


def _clean_hindi(text: str) -> str:
    """
    Post-process translated Hindi text:
    - Remove spurious repeated words (a common MarianMT artifact)
    - Strip extra whitespace
    """
    text = text.strip()
    # Remove consecutive duplicate tokens (e.g. "कि कि कि")
    text = re.sub(r'\b(\w+)(\s+\1)+\b', r'\1', text)
    return text
