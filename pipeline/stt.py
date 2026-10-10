import logging
import time
from dataclasses import dataclass

import numpy as np

from config import SUPPORTED_LANGS

log = logging.getLogger(__name__)

_INITIAL_PROMPTS = {
    "hi": "नमस्ते, आप कैसे हैं? मैं ठीक हूँ।",
    "en": None,
}


@dataclass
class Transcript:
    text: str
    language: str


def _cuda_available() -> bool:
    try:
        import ctranslate2
        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


class SpeechToText:
    def __init__(self, model_size="tiny", device="auto", compute_type="auto",
                 languages=SUPPORTED_LANGS):
        from faster_whisper import WhisperModel

        if device == "auto":
            device = "cuda" if _cuda_available() else "cpu"
        if compute_type == "auto":
            compute_type = "float16" if device == "cuda" else "int8"
        self.device, self.model_size = device, model_size
        self.languages = languages
        log.info("STT '%s' on %s (%s)", model_size, device, compute_type)
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def warmup(self):
        self.model.transcribe(np.zeros(16_000, dtype=np.float32), language="en", beam_size=1)

    def _run(self, audio, language):
        segments, info = self.model.transcribe(
            audio,
            language=language,
            beam_size=1,
            condition_on_previous_text=False,
            without_timestamps=True,
            vad_filter=False,
            initial_prompt=_INITIAL_PROMPTS.get(language) if language else None,
        )
        segments = list(segments)
        kept = [s for s in segments if s.no_speech_prob < 0.6 or s.avg_logprob > -0.5]
        return " ".join(s.text.strip() for s in kept).strip(), info

    def transcribe(self, audio, language="auto"):
        forced = language if language in self.languages else None
        text, info = self._run(audio, forced)
        lang = info.language if forced else info.language

        if forced is None and lang not in self.languages:
            probs = dict(getattr(info, "all_language_probs", None) or [])
            lang = max(self.languages, key=lambda c: probs.get(c, 0.0))
            text, _ = self._run(audio, lang)

        return Transcript(text=text, language=lang)

