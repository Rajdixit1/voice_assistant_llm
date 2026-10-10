import json
from typing import AsyncIterator

import httpx


class OllamaChat:
    def __init__(self, base_url, model, temperature=0.6, max_tokens=200,
                 keep_alive="30m"):
        self.model = model
        self.options = {"temperature": temperature, "num_predict": max_tokens}
        self.keep_alive = keep_alive
        self.client = httpx.AsyncClient(base_url=base_url, timeout=httpx.Timeout(120.0, connect=5.0))

    async def check(self):
        try:
            resp = await self.client.get("/api/tags")
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise RuntimeError(
                f"Cannot reach Ollama at {self.client.base_url}. Is `ollama serve` running?"
            ) from exc
        names = {m.get("name") for m in resp.json().get("models", [])}
        wanted = self.model if ":" in self.model else f"{self.model}:latest"
        if wanted not in names:
            raise RuntimeError(f"Model '{self.model}' not found. Run: ollama pull {self.model}")

    async def warmup(self):
        await self.client.post("/api/chat", json={
            "model": self.model, "messages": [], "keep_alive": self.keep_alive, "stream": False,
        })

    async def stream(self, messages) -> AsyncIterator[str]:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "keep_alive": self.keep_alive,
            "options": self.options,
        }
        async with self.client.stream("POST", "/api/chat", json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line:
                    continue
                data = json.loads(line)
                if "error" in data:
                    raise RuntimeError(f"Ollama error: {data['error']}")
                token = data.get("message", {}).get("content", "")
                if token:
                    yield token
                if data.get("done"):
                    break

    async def aclose(self):
        await self.client.aclose()

