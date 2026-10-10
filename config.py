import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

ROOT = Path(__file__).resolve().parent
SAMPLE_RATE = 16_000
SUPPORTED_LANGS = ("en", "hi")


def _bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class Settings:
    whisper_model: str = os.getenv("WHISPER_MODEL", "tiny")
    whisper_device: str = os.getenv("WHISPER_DEVICE", "auto")
    whisper_compute_type: str = os.getenv("WHISPER_COMPUTE_TYPE", "auto")

    ollama_url: str = os.getenv("OLLAMA_URL", "http://localhost:11434")
    llm_model: str = os.getenv("LLM_MODEL", "qwen2.5:3b")
    llm_temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.6"))
    llm_max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "200"))
    max_history_turns: int = int(os.getenv("MAX_HISTORY_TURNS", "8"))

    voices_dir: Path = Path(os.getenv("VOICES_DIR", str(ROOT / "voices")))
    voice_en: str = os.getenv("PIPER_VOICE_EN", "en_US-lessac-medium")
    voice_hi: str = os.getenv("PIPER_VOICE_HI", "hi_IN-pratham-medium")
    piper_use_cuda: bool = _bool("PIPER_USE_CUDA", False)

    assistant_name: str = os.getenv("ASSISTANT_NAME", "Aria")
    host: str = os.getenv("HOST", "127.0.0.1")
    port: int = int(os.getenv("PORT", "8000"))
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

    @property
    def voice_map(self) -> dict:
        return {"en": self.voice_en, "hi": self.voice_hi}


settings = Settings()

