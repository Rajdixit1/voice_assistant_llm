import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from config import ROOT, settings
from pipeline.llm import OllamaChat
from pipeline.session import CallSession
from pipeline.stt import SpeechToText
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


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    await CallSession(ws, ws.app.state).run()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host=settings.host, port=settings.port, log_level=settings.log_level.lower())
