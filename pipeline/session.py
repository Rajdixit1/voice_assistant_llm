import asyncio
import json
import logging
import struct
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from fastapi import WebSocket, WebSocketDisconnect

from config import SUPPORTED_LANGS, settings
from pipeline.prompts import build_messages
from pipeline.text_utils import SentenceChunker, clean_for_tts, guess_lang
from pipeline.turn_detector import FRAME_MS, FRAME_SAMPLES, Event, TurnDetector

log = logging.getLogger("voice")

# One STT worker so partial and final transcriptions never fight over the model.
STT_POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix="stt")
TTS_POOL = ThreadPoolExecutor(max_workers=2, thread_name_prefix="tts")

FRAME_BYTES = 4 + FRAME_SAMPLES * 2  # float32 speech probability + int16 PCM


def _ms(seconds: float) -> int:
    return round(seconds * 1000)


class Turn:
    """One assistant reply.

    A turn can start speculatively, during a short pause in the user's speech. Its
    output is held until commit(); if the user keeps talking it is cancelled instead
    and nothing ever reaches the browser.
    """

    def __init__(self, session, turn_id: int, committed: bool, t_end: float):
        self.session = session
        self.id = turn_id
        self.committed = committed
        self.speculative = not committed
        self.t_end = t_end  # when the user stopped speaking (or pressed send)
        self.task = None
        self._pending = []
        self.user_text = ""
        self.lang = "en"
        self.reply_parts = []
        self.finished = False
        self.interrupted = False
        self._recorded = False
        self.metrics = {}
        self._first_audio = False

    @property
    def running(self) -> bool:
        return self.task is not None and not self.task.done()

    async def emit_json(self, obj) -> None:
        if self.committed:
            await self.session.send_json(obj)
        else:
            self._pending.append((False, obj))

    async def emit_audio(self, pcm: bytes, rate: int) -> None:
        if self.committed:
            await self._send_audio(pcm, rate)
        else:
            self._pending.append((True, (pcm, rate)))

    async def _send_audio(self, pcm: bytes, rate: int) -> None:
        await self.session.send_audio(self.id, rate, pcm)
        if not self._first_audio:
            self._first_audio = True
            self.metrics["first_audio_ms"] = _ms(time.perf_counter() - self.t_end)
            self.metrics["speculative"] = self.speculative
            log.info("turn %d first audio after %s", self.id, self.metrics)
            await self.session.send_json({"type": "metrics", "turn": self.id, **self.metrics})

    async def commit(self) -> None:
        # Anything emitted while flushing lands in _pending and is sent in order.
        while self._pending:
            is_audio, item = self._pending.pop(0)
            if is_audio:
                await self._send_audio(*item)
            else:
                await self.session.send_json(item)
        self.committed = True
        self.record()

    def record(self) -> None:
        """Add the exchange to history once the turn is both committed and finished."""
        if self._recorded or not (self.committed and self.finished):
            return
        self._recorded = True
        history = self.session.history
        if self.user_text:
            history.append({"role": "user", "content": self.user_text})
        reply = "".join(self.reply_parts).strip()
        if reply:
            history.append({"role": "assistant", "content": reply + (" …" if self.interrupted else "")})


class CallSession:
    def __init__(self, ws: WebSocket, state):
        self.ws = ws
        self.stt = state.stt
        self.llm = state.llm
        self.tts = state.tts
        self.history = []
        self.language = "auto"
        self.barge_in = True
        self.in_call = False
        self.turn_id = 0
        self.turn = None
        self.playback_until = 0.0  # perf_counter estimate of when the browser finishes playing
        self._send_lock = asyncio.Lock()

        self.detector = TurnDetector(
            positive=settings.vad_positive, negative=settings.vad_negative,
            min_speech_ms=settings.min_speech_ms, pause_ms=settings.speculative_silence_ms,
            end_ms=settings.endpoint_silence_ms,
        )
        self.utt_id = 0
        self.utt_mode = "normal"  # normal | interrupting | ignored
        self.speech_end_at = 0.0
        self.utt_lang = None  # language confidently detected by this utterance's partials
        self.partial_task = None
        self.partial_at = 0
        self.partial_frames = 0
        if settings.partial_interval_ms > 0 and self.stt.device == "cuda":
            self.partial_frames = max(1, round(settings.partial_interval_ms / FRAME_MS))

    # transport

    async def send_json(self, obj) -> None:
        async with self._send_lock:
            await self.ws.send_text(json.dumps(obj, ensure_ascii=False))

    async def send_audio(self, turn: int, rate: int, pcm: bytes) -> None:
        now = time.perf_counter()
        self.playback_until = max(self.playback_until, now) + len(pcm) / 2 / rate
        async with self._send_lock:
            await self.ws.send_bytes(struct.pack("<II", turn, rate) + pcm)

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
                    await self.on_frame(msg["bytes"])
                elif msg.get("text"):
                    await self.on_control(json.loads(msg["text"]))
        except WebSocketDisconnect:
            pass
        finally:
            await self.cancel_turn()

    async def on_control(self, msg) -> None:
        kind = msg.get("type")
        if kind == "config":
            lang = msg.get("language", "auto")
            self.language = lang if lang in SUPPORTED_LANGS else "auto"
            self.barge_in = bool(msg.get("barge_in", True))
        elif kind == "call_start":
            self.in_call = True
            self.detector.reset()
            if msg.get("greeting", True) and settings.greeting:
                await self.start_turn(say=settings.greeting.format(name=settings.assistant_name))
        elif kind == "call_end":
            self.in_call = False
            self.detector.reset()
            await self.cancel_turn()
            self.playback_until = 0.0
        elif kind == "text" and msg.get("text", "").strip():
            await self.start_turn(text=msg["text"].strip())
        elif kind == "reset":
            await self.cancel_turn()
            self.history.clear()
            self.playback_until = 0.0
            await self.send_json({"type": "reset_ok"})

    # turn-taking

    def assistant_active(self) -> bool:
        t = self.turn
        talking = t is not None and t.committed and t.running
        return talking or time.perf_counter() < self.playback_until + 0.15

    async def on_frame(self, data: bytes) -> None:
        if not self.in_call or len(data) != FRAME_BYTES:
            return
        prob = struct.unpack_from("<f", data)[0]
        frame = np.frombuffer(data, dtype=np.int16, offset=4).astype(np.float32) / 32768.0
        det = self.detector
        for event in det.feed(prob, frame):
            await self.on_event(event)

        if det.active and self.utt_mode == "interrupting" and det.speech_ms >= settings.barge_in_ms:
            await self.barge_in_now()
        if (self.partial_frames and det.confirmed and not det.paused and self.utt_mode == "normal"
                and det.utterance_frames - self.partial_at >= self.partial_frames
                and (self.partial_task is None or self.partial_task.done())):
            self.partial_at = det.utterance_frames
            self.partial_task = asyncio.create_task(self.run_partial(self.utt_id, det.audio()))

    async def on_event(self, event: Event) -> None:
        det = self.detector
        if event == Event.SPEECH_START:
            self.utt_id += 1
            self.partial_at = 0
            self.utt_lang = None
            if not self.assistant_active():
                self.utt_mode = "normal"
            elif self.barge_in:
                self.utt_mode = "interrupting"
                await self.send_json({"type": "duck"})
            else:
                self.utt_mode = "ignored"

        elif event == Event.MISFIRE:
            if self.utt_mode == "interrupting":
                await self.send_json({"type": "unduck"})

        elif event == Event.PAUSE:
            self.speech_end_at = time.perf_counter() - det.silence * FRAME_MS / 1000
            if self.utt_mode == "normal":
                await self.start_turn(audio=det.audio(), speculative=True, t_end=self.speech_end_at)

        elif event == Event.RESUME:
            if self.turn and not self.turn.committed:
                await self.cancel_turn()

        elif event == Event.END:
            mode = self.utt_mode
            audio = det.audio()
            det.finish()
            if mode == "normal":
                if self.turn and not self.turn.committed:
                    await self.turn.commit()
                else:
                    t_end = self.speech_end_at or time.perf_counter()
                    await self.start_turn(audio=audio, t_end=t_end)
            elif mode == "interrupting":
                # Short sound while the assistant talks ("mm-hmm", a cough): keep going.
                await self.send_json({"type": "unduck"})
                await self.send_json({"type": "partial_cancel"})
            self.utt_mode = "normal"
            self.speech_end_at = 0.0

    async def barge_in_now(self) -> None:
        self.utt_mode = "normal"
        was_playing = time.perf_counter() < self.playback_until
        await self.cancel_turn()
        self.playback_until = 0.0
        if was_playing and self.history and self.history[-1]["role"] == "assistant":
            last = self.history[-1]
            if not last["content"].endswith("…"):
                last["content"] += " …"
        await self.send_json({"type": "clear"})

    async def run_partial(self, utt_id: int, audio: np.ndarray) -> None:
        loop = asyncio.get_running_loop()
        try:
            tr = await loop.run_in_executor(STT_POOL, self.stt.transcribe, audio, self.stt_language())
        except Exception:
            log.exception("partial transcription failed")
            return
        if utt_id == self.utt_id and tr.probability >= 0.8:
            self.utt_lang = tr.language
        # Once the utterance has ended, the turn's final transcript wins.
        if utt_id != self.utt_id or self.utt_mode != "normal" or not self.detector.active or not tr.text:
            return
        await self.send_json({"type": "partial", "utt": utt_id, "text": tr.text})

    def stt_language(self) -> str:
        """Selected language, else one the partials already detected (skips re-detection)."""
        return self.language if self.language != "auto" else (self.utt_lang or "auto")

    # turns

    async def cancel_turn(self) -> None:
        turn, self.turn = self.turn, None
        if turn and turn.running:
            turn.task.cancel()
            try:
                await turn.task
            except (asyncio.CancelledError, Exception):
                pass

    async def start_turn(self, audio=None, text=None, say=None, speculative=False, t_end=None) -> None:
        await self.cancel_turn()
        self.turn_id += 1
        turn = Turn(self, self.turn_id, committed=not speculative, t_end=t_end or time.perf_counter())
        turn.task = asyncio.create_task(self.run_turn(turn, audio, text, say))
        self.turn = turn

    async def run_turn(self, turn: Turn, audio, text, say) -> None:
        loop = asyncio.get_running_loop()
        try:
            await turn.emit_json({"type": "turn_start", "turn": turn.id})
            if audio is not None:
                t0 = time.perf_counter()
                tr = await loop.run_in_executor(STT_POOL, self.stt.transcribe, audio, self.stt_language())
                turn.metrics["stt_ms"] = _ms(time.perf_counter() - t0)
                text, lang = tr.text, tr.language
            elif say is not None:
                lang = self.language if self.language != "auto" else "en"
            else:
                lang = self.language if self.language != "auto" else guess_lang(text)
            turn.lang = lang

            if say is None:
                if not text:
                    await turn.emit_json({"type": "no_speech", "turn": turn.id})
                    return
                turn.user_text = text
                await turn.emit_json({"type": "transcript", "turn": turn.id, "text": text, "lang": lang})
                messages = build_messages(
                    [*self.history, {"role": "user", "content": text}],
                    lang, settings.max_history_turns, settings.assistant_name)

            queue = asyncio.Queue()

            async def produce() -> None:
                chunker = SentenceChunker()
                try:
                    if say is not None:
                        turn.reply_parts.append(say)
                        await turn.emit_json({"type": "token", "turn": turn.id, "text": say})
                        queue.put_nowait(say)
                        return
                    t0 = time.perf_counter()
                    async for token in self.llm.stream(messages):
                        if not turn.reply_parts:
                            turn.metrics["llm_first_ms"] = _ms(time.perf_counter() - t0)
                        turn.reply_parts.append(token)
                        await turn.emit_json({"type": "token", "turn": turn.id, "text": token})
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
                    t0 = time.perf_counter()
                    pcm, rate = await loop.run_in_executor(
                        TTS_POOL, self.tts.synthesize, chunk, guess_lang(chunk, default=lang))
                    turn.metrics.setdefault("tts_first_ms", _ms(time.perf_counter() - t0))
                    await turn.emit_audio(pcm, rate)

            tasks = [asyncio.create_task(produce()), asyncio.create_task(consume())]
            try:
                await asyncio.gather(*tasks)
            except BaseException:
                for t in tasks:
                    t.cancel()
                raise
            await turn.emit_json({"type": "done", "turn": turn.id})

        except asyncio.CancelledError:
            turn.interrupted = True
            raise
        except Exception as exc:
            log.exception("turn %d failed", turn.id)
            await turn.emit_json({"type": "error", "turn": turn.id, "message": str(exc)})
        finally:
            turn.finished = True
            turn.record()
