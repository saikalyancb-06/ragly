"""OpenAI-compatible client for the local answer engine (llama-server or GenieX)."""
from __future__ import annotations

import time
from typing import Iterator

import httpx
from openai import OpenAI

from .config import settings
from .offline import assert_local_url


class LLMClient:
    def __init__(self, base_url: str, model: str = ""):
        assert_local_url(base_url)
        self.base_url = base_url.rstrip("/")
        self._model = model
        self.client = OpenAI(
            base_url=self.base_url,
            api_key="local-no-key",
            timeout=settings.request_timeout,
            max_retries=0,
            http_client=httpx.Client(trust_env=False, timeout=settings.request_timeout),  # ignore system proxies
        )

    def list_models(self) -> list[str]:
        return [m.id for m in self.client.models.list().data]

    def healthy(self, timeout: float = 2.0) -> bool:
        try:
            r = httpx.get(f"{self.base_url}/models", timeout=timeout, trust_env=False)
            return r.status_code == 200
        except httpx.HTTPError:
            return False

    @property
    def model(self) -> str:
        if not self._model:
            models = self.list_models()
            self._model = models[0] if models else "local"
        return self._model

    def stream_chat(self, messages: list[dict], max_tokens: int | None = None) -> Iterator[dict]:
        """Yield {'token': str} events, then a final {'stats': {...}} event."""
        t0 = time.perf_counter()
        first = None
        n_tokens = 0
        usage_tokens = None
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=settings.temperature,
            max_tokens=max_tokens or settings.max_tokens,
            stream=True,
        )
        for part in stream:
            if getattr(part, "usage", None) and part.usage and part.usage.completion_tokens:
                usage_tokens = part.usage.completion_tokens
            if not part.choices:
                continue
            delta = part.choices[0].delta.content or ""
            if delta:
                if first is None:
                    first = time.perf_counter()
                n_tokens += 1
                yield {"token": delta}
        end = time.perf_counter()
        tokens = usage_tokens or n_tokens
        gen_time = end - (first or end)
        yield {
            "stats": {
                "model": self.model,
                "ttft_ms": round(((first or end) - t0) * 1000, 1),
                "total_ms": round((end - t0) * 1000, 1),
                "completion_tokens": tokens,
                "tokens_per_sec": round(tokens / gen_time, 2) if gen_time > 0 else None,
            }
        }

    def chat(self, messages: list[dict], max_tokens: int | None = None) -> tuple[str, dict]:
        text, stats = [], {}
        for ev in self.stream_chat(messages, max_tokens):
            if "token" in ev:
                text.append(ev["token"])
            else:
                stats = ev["stats"]
        return "".join(text), stats
