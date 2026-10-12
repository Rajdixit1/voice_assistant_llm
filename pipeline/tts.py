import logging
from pathlib import Path

log = logging.getLogger(__name__)


class TextToSpeech:
    def __init__(self, voices_dir: Path, voice_map: dict, use_cuda: bool = False):
        from piper import PiperVoice

        self.voices = {}
        for lang, name in voice_map.items():
            model = Path(voices_dir) / f"{name}.onnx"
            if not model.exists():
                log.warning("Voice '%s' missing: %s", lang, model)
                continue
            self.voices[lang] = PiperVoice.load(str(model), config_path=f"{model}.json", use_cuda=use_cuda)
        if not self.voices:
            raise RuntimeError("No Piper voices found in " + str(voices_dir))
        self.fallback = "en" if "en" in self.voices else next(iter(self.voices))

    @property
    def languages(self) -> list:
        return list(self.voices)

    def synthesize(self, text: str, lang: str = "en") -> tuple[bytes, int]:
        """Return raw 16-bit mono PCM and its sample rate."""
        voice = self.voices.get(lang) or self.voices[self.fallback]
        if hasattr(voice, "synthesize_stream_raw"):
            pcm = b"".join(voice.synthesize_stream_raw(text))
        else:
            pcm = b"".join(chunk.audio_int16_bytes for chunk in voice.synthesize(text))
        return pcm, voice.config.sample_rate

    def warmup(self) -> None:
        for lang in self.voices:
            self.synthesize("नमस्ते" if lang == "hi" else "Hello.", lang)

