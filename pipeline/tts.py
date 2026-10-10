import io
import logging
import wave
from pathlib import Path

log = logging.getLogger(__name__)


def pcm16_to_wav(pcm: bytes, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    return buf.getvalue()


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

    def synthesize(self, text: str, lang: str = "en") -> bytes:
        voice = self.voices.get(lang) or self.voices[self.fallback]
        if hasattr(voice, "synthesize_stream_raw"):
            pcm = b"".join(voice.synthesize_stream_raw(text))
        else:
            pcm = b"".join(chunk.audio_int16_bytes for chunk in voice.synthesize(text))
        return pcm16_to_wav(pcm, voice.config.sample_rate)

    def warmup(self) -> None:
        for lang in self.voices:
            self.synthesize("नमस्ते" if lang == "hi" else "Hello.", lang)

