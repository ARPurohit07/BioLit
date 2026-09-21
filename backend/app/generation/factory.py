"""Choose the LLM client and say plainly whether it sends text off this machine."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Mapping, Optional

from backend.app.generation.ollama_client import OllamaClient
from backend.app.generation.openrouter_client import OpenRouterClient


@dataclass
class LLMChoice:
    client: Any
    label: str
    remote: bool                    # true when the question and retrieved passages leave this machine
    warning: Optional[str] = None


def build_llm_client(models_config: Mapping, ollama_host: str, ollama_model: str,
                     environ: Optional[Mapping[str, str]] = None) -> LLMChoice:
    """provider "ollama" runs the model Ollama serves (a model tagged -cloud is served by Ollama's cloud, so it is remote);
    provider "openrouter" needs OPENROUTER_KEY and falls back to Ollama, with a warning, when the key is missing.
    BIOLIT_GENERATION_PROVIDER overrides the configured provider."""
    env = os.environ if environ is None else environ
    gen = models_config.get("generation", {})
    provider = env.get("BIOLIT_GENERATION_PROVIDER") or gen.get("provider", "ollama")
    timeout = models_config.get("ollama", {}).get("request_timeout_s", 120)
    warning = None
    if provider == "openrouter":
        key = env.get("OPENROUTER_KEY", "")
        cfg = gen.get("openrouter", {})
        if key:
            client = OpenRouterClient(
                api_key=key, model=cfg.get("model", "openai/gpt-oss-120b"),
                base_url=cfg.get("base_url", "https://openrouter.ai/api/v1"), timeout_s=cfg.get("timeout_s", 120),
                max_tokens=cfg.get("max_tokens", 2048), reasoning_effort=cfg.get("reasoning_effort", "low"))
            return LLMChoice(client, f"openrouter:{client.model}", True)
        warning = "generation.provider is openrouter but OPENROUTER_KEY is not set; using local Ollama"
    client = OllamaClient(host=ollama_host, model=ollama_model, timeout_s=timeout)
    return LLMChoice(client, f"ollama:{ollama_model}", ollama_model.endswith("-cloud"), warning)
