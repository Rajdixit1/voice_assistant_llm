# Voice Assistant (STT → LLM → TTS)

Local voice assistant. Speak English or Hindi — it listens (faster-whisper),
thinks (Ollama + Qwen), and speaks back (Piper). No paid APIs.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # or copy manually on Windows

# Piper voices (~126 MB)
python -c "import urllib.request; ..."
# Or download manually from https://huggingface.co/rhasspy/piper-voices:
# en/en_US/lessac/medium/en_US-lessac-medium.onnx (+ .json)
# hi/hi_IN/pratham/medium/hi_IN-pratham-medium.onnx (+ .json)
# into ./voices/

# LLM
ollama pull qwen2.5:3b
```

## Run

```bash
python server.py
```

Open http://127.0.0.1:8000 in Chrome/Edge, click **Start talking**, allow the mic.

## How it works

1. Browser VAD (Silero) detects end of speech, sends 16 kHz PCM over WebSocket.
2. Server transcribes with faster-whisper (auto English/Hindi).
3. Transcript + recent history go to Ollama (streaming).
4. Tokens stream to UI and into a sentence chunker; each sentence goes to Piper
   while the LLM still generates the next one (LLM/TTS overlap = low latency).
5. WAV chunks stream back for gapless playback.
6. Speaking mid-reply pauses playback; real speech cancels the turn (barge-in).

Config: `.env` — `WHISPER_MODEL`, `LLM_MODEL`, `MAX_HISTORY_TURNS`, voices, host/port.
