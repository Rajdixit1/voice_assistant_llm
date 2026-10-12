from collections import deque
from enum import Enum

import numpy as np

FRAME_SAMPLES = 512  # Silero v5 frame at 16 kHz = 32 ms
FRAME_MS = FRAME_SAMPLES * 1000 / 16_000


class Event(str, Enum):
    SPEECH_START = "speech_start"   # first speech-like frame (may still be noise)
    SPEECH = "speech"               # enough speech to count as a real utterance
    PAUSE = "pause"                 # short silence: good moment to start a reply speculatively
    RESUME = "resume"               # user kept talking after a PAUSE
    END = "end"                     # silence long enough to end the user's turn
    MISFIRE = "misfire"             # speech start that never became real speech


def _frames(ms: float) -> int:
    return max(1, round(ms / FRAME_MS))


class TurnDetector:
    """Turns a stream of (speech probability, PCM frame) into turn-taking events.

    Works in audio time (frame counts), so decisions don't depend on network jitter.
    """

    def __init__(self, positive=0.6, negative=0.35, min_speech_ms=160, pause_ms=200,
                 end_ms=550, resume_ms=96, pre_pad_ms=300, max_utterance_s=30):
        self.positive, self.negative = positive, negative
        self.min_speech = _frames(min_speech_ms)
        self.pause = _frames(pause_ms)
        self.end = max(_frames(end_ms), self.pause + 1)
        self.resume = _frames(resume_ms)
        self.max_frames = _frames(max_utterance_s * 1000)
        self._pad = deque(maxlen=_frames(pre_pad_ms))
        self._reset()

    def _reset(self):
        self.active = False
        self.confirmed = False
        self.paused = False
        self._frames = []
        self._speechy = False
        self.speech_frames = 0
        self.silence = 0
        self._resume_run = 0

    @property
    def speech_ms(self) -> float:
        return self.speech_frames * FRAME_MS

    @property
    def utterance_frames(self) -> int:
        return len(self._frames)

    def audio(self) -> np.ndarray:
        if not self._frames:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(self._frames)

    def reset(self):
        self._pad.clear()
        self._reset()

    def feed(self, prob: float, frame: np.ndarray) -> list:
        if not self.active:
            if prob < self.positive:
                self._pad.append(frame)
                return []
            self.active = True
            self._frames = [*self._pad, frame]
            self._pad.clear()
            self._speechy = True
            self.speech_frames = 1
            return [Event.SPEECH_START]

        self._frames.append(frame)
        # Hysteresis: stay "speechy" until the probability drops below the negative threshold.
        self._speechy = prob >= self.positive or (self._speechy and prob >= self.negative)
        events = []

        if self._speechy:
            self.speech_frames += 1
            if self.paused:
                self._resume_run += 1
                if self._resume_run >= self.resume:
                    self.paused = False
                    self.silence = 0
                    self._resume_run = 0
                    events.append(Event.RESUME)
            else:
                self.silence = 0
            if not self.confirmed and self.speech_frames >= self.min_speech:
                self.confirmed = True
                events.append(Event.SPEECH)
        else:
            self._resume_run = 0
            self.silence += 1
            if not self.confirmed:
                if self.silence >= self.pause:
                    self._reset()
                    events.append(Event.MISFIRE)
                return events
            if not self.paused and self.silence >= self.pause:
                self.paused = True
                events.append(Event.PAUSE)
            if self.silence >= self.end:
                events.append(Event.END)

        if Event.END not in events and self.confirmed and len(self._frames) >= self.max_frames:
            events.append(Event.END)
        return events

    def finish(self):
        """Call after handling END: start listening for the next utterance."""
        self._reset()
