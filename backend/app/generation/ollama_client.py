"""Thin REST client for a local Ollama server.

No cloud LLM APIs are used anywhere in BioLit — this is the only place the
system talks to a language model, and it always talks to a local Ollama
instance (self.host), never a hard-coded address.
"""
from __future__ import annotations

import json
import time
from typing import Any, Iterator, Optional

import requests


class OllamaClient:
    def __init__(self, host: str, model: str, timeout_s: int = 120):
        self.host = host.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s

    def is_available(self) -> bool:
        try:
            resp = requests.get(f"{self.host}/api/tags", timeout=3)
            return resp.status_code == 200
        except Exception:
            return False

    def _options(self, temperature: float, top_p: float, num_ctx: int) -> dict:
        return {"temperature": temperature, "top_p": top_p, "num_ctx": num_ctx}

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.1,
        top_p: float = 0.9,
        num_ctx: int = 8192,
    ) -> str:
        text, _ = self.generate_with_stats(
            prompt, system=system, temperature=temperature, top_p=top_p, num_ctx=num_ctx
        )
        return text

    def generate_with_stats(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.1,
        top_p: float = 0.9,
        num_ctx: int = 8192,
    ) -> tuple[str, dict[str, Any]]:
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": self._options(temperature, top_p, num_ctx),
        }
        if system:
            payload["system"] = system

        start = time.perf_counter()
        resp = requests.post(f"{self.host}/api/generate", json=payload, timeout=self.timeout_s)
        latency_ms = (time.perf_counter() - start) * 1000.0
        resp.raise_for_status()
        data = resp.json()

        eval_count = data.get("eval_count", 0) or 0
        eval_duration_ns = data.get("eval_duration", 0) or 0
        tokens_per_second = 0.0
        if eval_duration_ns > 0:
            tokens_per_second = eval_count / (eval_duration_ns / 1e9)

        stats = {
            "latency_ms": latency_ms,
            "eval_count": eval_count,
            "tokens_per_second": tokens_per_second,
        }
        return data.get("response", ""), stats

    def chat(self, messages: list[dict], temperature: float = 0.1) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        }
        resp = requests.post(f"{self.host}/api/chat", json=payload, timeout=self.timeout_s)
        resp.raise_for_status()
        data = resp.json()
        return data.get("message", {}).get("content", "")

    def stream(self, prompt: str, system: Optional[str] = None) -> Iterator[str]:
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
        }
        if system:
            payload["system"] = system

        with requests.post(f"{self.host}/api/generate", json=payload, timeout=self.timeout_s, stream=True) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                piece = chunk.get("response", "")
                if piece:
                    yield piece
                if chunk.get("done"):
                    break
