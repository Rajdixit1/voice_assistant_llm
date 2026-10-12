import re

_SENTENCE_END = re.compile(r"[.!?।॥]+[\"'”’)\]]*(?=\s)")
_SOFT_BREAK = re.compile(r"[,;:—–](?=\s)")
_DEVANAGARI = re.compile(r"[\u0900-\u097F]")
_LATIN = re.compile(r"[A-Za-z]")

_EMOJI = re.compile(
    "[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF\u200d\ufe0f]",
    flags=re.UNICODE,
)
_MD_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+", flags=re.MULTILINE)
_MD_CHARS = re.compile(r"[*_#`>|~]+")
_URL = re.compile(r"https?://\S+")
_SPACES = re.compile(r"\s+")


class SentenceChunker:
    def __init__(self, first_min_chars=24, soft_min_chars=120, max_chars=240):
        self.first_min_chars = first_min_chars
        self.soft_min_chars = soft_min_chars
        self.max_chars = max_chars
        self._buf = ""
        self._emitted = 0

    def feed(self, text: str) -> list[str]:
        self._buf += text
        chunks = []
        while (chunk := self._next_chunk()) is not None:
            if chunk:
                chunks.append(chunk)
                self._emitted += 1
        return chunks

    def flush(self) -> list[str]:
        rest, self._buf = self._buf.strip(), ""
        if rest:
            self._emitted += 1
            return [rest]
        return []

    def _cut(self, end: int) -> str:
        chunk, self._buf = self._buf[:end].strip(), self._buf[end:].lstrip()
        return chunk

    def _next_chunk(self):
        match = _SENTENCE_END.search(self._buf)
        if match:
            return self._cut(match.end())

        soft_min = self.first_min_chars if self._emitted == 0 else self.soft_min_chars
        if len(self._buf) >= soft_min:
            breaks = [m.end() for m in _SOFT_BREAK.finditer(self._buf) if m.end() >= 15]
            if breaks:
                return self._cut(breaks[-1])

        if len(self._buf) >= self.max_chars:
            space = self._buf.rfind(" ", 0, self.max_chars)
            return self._cut(space if space > 0 else self.max_chars)
        return None


def clean_for_tts(text):
    text = _URL.sub("", text)
    text = _MD_BULLET.sub("", text)
    text = _MD_CHARS.sub("", text)
    text = _EMOJI.sub("", text)
    return _SPACES.sub(" ", text).strip()


def guess_lang(text, default="en"):
    deva = len(_DEVANAGARI.findall(text))
    latin = len(_LATIN.findall(text))
    if deva == 0 and latin == 0:
        return default
    return "hi" if deva >= latin else "en"
