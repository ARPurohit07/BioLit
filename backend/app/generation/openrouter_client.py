"""OpenRouter chat client with the same interface as OllamaClient, so the pipeline and the claim verifier use either.

Unlike the rest of the stack this sends text to a third party: the question and the retrieved passages of the indexed
papers go to OpenRouter. Retrieval, embeddings and reranking stay local. Selected by `generation.provider` in
configs/models.yaml, and the health endpoint reports which one is active.
"""
from __future__ import annotations

import time
from typing import Any, Optional

import requests


class OpenRouterClient:
    def __init__(self, api_key: str, model: str, base_url: str = "https://openrouter.ai/api/v1",
                 timeout_s: int = 120, max_tokens: int = 2048, reasoning_effort: Optional[str] = "low"):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort

    def is_available(self) -> bool:
        return bool(self.api_key)

    def generate(self, prompt: str, system: Optional[str] = None, temperature: float = 0.1,
                 top_p: float = 0.9, num_ctx: int = 0) -> str:
        return self.generate_with_stats(prompt, system=system, temperature=temperature, top_p=top_p)[0]

    def generate_with_stats(self, prompt: str, system: Optional[str] = None, temperature: float = 0.1,
                            top_p: float = 0.9, num_ctx: int = 0) -> tuple[str, dict[str, Any]]:
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        payload: dict[str, Any] = {"model": self.model, "messages": messages, "temperature": temperature,
                                   "top_p": top_p, "max_tokens": self.max_tokens}
        if self.reasoning_effort:
            payload["reasoning"] = {"effort": self.reasoning_effort}

        start = time.perf_counter()
        for attempt in range(3):
            resp = requests.post(f"{self.base_url}/chat/completions", json=payload, timeout=self.timeout_s,
                                 headers={"Authorization": f"Bearer {self.api_key}"})
            if resp.status_code in (429, 502, 503, 504) and attempt < 2:
                time.sleep(2 ** attempt * 2)
                continue
            resp.raise_for_status()
            break
        latency_ms = (time.perf_counter() - start) * 1000.0
        data = resp.json()
        text = ((data.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
        tokens = (data.get("usage") or {}).get("completion_tokens", 0) or 0
        return text.strip(), {"latency_ms": latency_ms, "eval_count": tokens,
                              "tokens_per_second": tokens / (latency_ms / 1000.0) if latency_ms else 0.0}
