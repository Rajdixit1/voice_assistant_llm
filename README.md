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

Open http://127.0.0.1:8000 in Chrome/Edge, click **Start call**, allow the mic, and talk.
The call stays open: no buttons between turns, and you can interrupt the assistant.

Use an NVIDIA GPU for speech-to-text (`WHISPER_DEVICE=auto` or `cuda`,
`WHISPER_COMPUTE_TYPE=auto`). On CPU, Whisper `small` alone takes ~4 s per turn.

## How it works

1. During a call the browser streams every 32 ms mic frame plus its Silero VAD
   speech probability over a WebSocket.
2. The server's turn detector (`pipeline/turn_detector.py`) tracks who is talking.
   While you speak, partial transcripts appear live (GPU only).
3. After ~200 ms of silence the server starts the reply speculatively
   (faster-whisper → Ollama → Piper) but holds the output. At ~500 ms of silence the
   turn is final and the held reply is released at once; if you keep talking it is
   discarded.
4. LLM tokens feed a sentence chunker; each chunk goes to Piper while the LLM
   generates the next. Raw PCM streams back for gapless playback.
5. If you talk over the assistant, its volume ducks immediately. Sustained speech
   (~300 ms) cancels the reply (barge-in); a short "mm-hmm" is ignored.
6. Each reply shows its latency (end of your speech → first audio) in the UI.

Config: `.env`. Main settings are `WHISPER_MODEL`, `WHISPER_DEVICE`, `LLM_MODEL`,
`MAX_HISTORY_TURNS`, voices and host/port. Turn-taking settings are
`SPECULATIVE_SILENCE_MS`, `ENDPOINT_SILENCE_MS`, `BARGE_IN_MS`,
`PARTIAL_INTERVAL_MS` and `GREETING` (empty disables the greeting).
