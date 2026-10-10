import asyncio
import json
import logging
import struct
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from config import ROOT, SAMPLE_RATE, SUPPORTED_LANGS, settings
from pipeline.llm import OllamaChat
from pipeline.prompts import build_messages
from pipeline.stt import SpeechToText
from pipeline.text_utils import SentenceChunker, clean_for_tts, guess_lang
from pipeline.tts import TextToSpeech

log = logging.getLogger("voice")
EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="infer")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    loop = asyncio.get_running_loop()

    llm = OllamaChat(settings.ollama_url, settings.llm_model, settings.llm_temperature, settings.llm_max_tokens)
    await llm.check()
    stt = await loop.run_in_executor(EXECUTOR, lambda: SpeechToText(
        settings.whisper_model, settings.whisper_device, settings.whisper_compute_type))
    tts = await loop.run_in_executor(EXECUTOR, lambda: TextToSpeech(
        settings.voices_dir, settings.voice_map, settings.piper_use_cuda))

    # Warm everything up so the first real turn is fast.
    await asyncio.gather(
        loop.run_in_executor(EXECUTOR, stt.warmup),
        loop.run_in_executor(EXECUTOR, tts.warmup),
        llm.warmup(),
    )
    log.info("Ready. Open http://%s:%d", settings.host, settings.port)

    app.state.stt, app.state.llm, app.state.tts = stt, llm, tts
    yield
    await llm.aclose()


app = FastAPI(title="Voice Assistant", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


@app.get("/")
async def index():
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/health")
async def health():
    return {"ok": True, "llm": settings.llm_model, "stt": settings.whisper_model}


class VoiceSession:
    def __init__(self, ws: WebSocket, state):
        self.ws = ws
        self.stt = state.stt
        self.llm = state.llm
        self.tts = state.tts
        self.history = []
        self.language = "auto"
        self.turn_id = 0
        self.task = None
        self._send_lock = asyncio.Lock()

    async def send_json(self, obj) -> None:
        async with self._send_lock:
            await self.ws.send_text(json.dumps(obj, ensure_ascii=False))

    async def send_audio(self, turn: int, wav: bytes) -> None:
        async with self._send_lock:
            await self.ws.send_bytes(struct.pack("<I", turn) + wav)

    async def run(self) -> None:
        await self.send_json({
            "type": "ready",
            "stt": f"faster-whisper {self.stt.model_size} ({self.stt.device})",
            "llm": settings.llm_model,
            "tts": {lang: settings.voice_map[lang] for lang in self.tts.languages},
            "name": settings.assistant_name,
        })
        try:
            while True:
                msg = await self.ws.receive()
                if msg["type"] == "websocket.disconnect":
                    break
                if msg.get("bytes") is not None:
                    await self.on_audio(msg["bytes"])
                elif msg.get("text"):
                    await self.on_control(json.loads(msg["text"]))
        except WebSocketDisconnect:
            pass
        finally:
            await self.cancel_turn()

    async def on_audio(self, data) -> None:
        audio = np.frombuffer(data, dtype=np.float32).copy()
        if audio.size < SAMPLE_RATE * 0.25:
            return
        await self.start_turn(audio=audio)

    async def on_control(self, msg) -> None:
        kind = msg.get("type")
        if kind == "config":
            lang = msg.get("language", "auto")
            self.language = lang if lang in SUPPORTED_LANGS else "auto"
        elif kind == "text" and msg.get("text", "").strip():
            await self.start_turn(text=msg["text"].strip())
        elif kind == "interrupt":
            await self.cancel_turn()
            await self.send_json({"type": "interrupted"})
        elif kind == "reset":
            await self.cancel_turn()
            self.history.clear()
            await self.send_json({"type": "reset_ok"})

    async def cancel_turn(self) -> None:
        if self.task and not self.task.done():
            self.task.cancel()
            try:
                await self.task
            except (asyncio.CancelledError, Exception):
                pass
        self.task = None

    async def start_turn(self, audio=None, text=None) -> None:
        await self.cancel_turn()
        self.turn_id += 1
        self.task = asyncio.create_task(self.run_turn(self.turn_id, audio, text))

    async def run_turn(self, turn, audio, text) -> None:
        loop = asyncio.get_running_loop()
        reply_parts = []
        interrupted = False
        await self.send_json({"type": "turn_start", "turn": turn})

        try:
            if audio is not None:
                tr = await loop.run_in_executor(EXECUTOR, self.stt.transcribe, audio, self.language)
                text, lang = tr.text, tr.language
            else:
                lang = self.language if self.language != "auto" else guess_lang(text)
            if not text:
                await self.send_json({"type": "no_speech", "turn": turn})
                return

            await self.send_json({"type": "transcript", "turn": turn, "text": text, "lang": lang})
            self.history.append({"role": "user", "content": text})
            messages = build_messages(self.history, lang, settings.max_history_turns, settings.assistant_name)

            queue = asyncio.Queue()

            async def produce() -> None:
                chunker = SentenceChunker()
                try:
                    async for token in self.llm.stream(messages):
                        reply_parts.append(token)
                        await self.send_json({"type": "token", "turn": turn, "text": token})
                        for chunk in chunker.feed(token):
                            queue.put_nowait(chunk)
                    for chunk in chunker.flush():
                        queue.put_nowait(chunk)
                finally:
                    queue.put_nowait(None)

            async def consume() -> None:
                while (chunk := await queue.get()) is not None:
                    chunk = clean_for_tts(chunk)
                    if not chunk:
                        continue
                    voice_lang = guess_lang(chunk, default=lang)
                    wav = await loop.run_in_executor(EXECUTOR, self.tts.synthesize, chunk, voice_lang)
                    await self.send_audio(turn, wav)

            tasks = [asyncio.create_task(produce()), asyncio.create_task(consume())]
            try:
                await asyncio.gather(*tasks)
            except BaseException:
                for t in tasks:
                    t.cancel()
                raise

            await self.send_json({"type": "done", "turn": turn})

        except asyncio.CancelledError:
            interrupted = True
            raise
        except Exception as exc:
            log.exception("turn %d failed", turn)
            await self.send_json({"type": "error", "turn": turn, "message": str(exc)})
        finally:
            reply = "".join(reply_parts).strip()
            if reply:
                self.history.append({"role": "assistant", "content": reply + (" …" if interrupted else "")})


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    await VoiceSession(ws, ws.app.state).run()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host=settings.host, port=settings.port, log_level=settings.log_level.lower())
