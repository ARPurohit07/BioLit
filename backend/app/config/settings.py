"""Central application settings.

Loads configs/*.yaml plus environment variables (via .env). This is the
single place other modules should read configuration from — never hard-code
paths, hosts, or model names elsewhere.
"""
from __future__ import annotations

import functools
import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

# Repo root = three levels up from this file (backend/app/config/settings.py -> BioLit/)
REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIGS_DIR = REPO_ROOT / "configs"
DATA_DIR = REPO_ROOT / "data"


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _env(key: str, default: str) -> str:
    return os.environ.get(key, default)


class Settings(BaseModel):
    # Paths
    repo_root: Path = REPO_ROOT
    data_dir: Path = DATA_DIR
    documents_dir: Path = DATA_DIR / "documents"
    index_dir: Path = DATA_DIR / "index"
    metadata_dir: Path = DATA_DIR / "metadata"
    results_dir: Path = DATA_DIR / "results"

    sqlite_path: Path = DATA_DIR / "metadata" / "biolit.db"

    # Ollama
    ollama_host: str = _env("OLLAMA_HOST", "http://localhost:11434")
    ollama_model: str = _env("OLLAMA_MODEL", "qwen2.5:3b")

    # Raw config blobs (validated lazily by the modules that use them)
    models_config: dict[str, Any] = {}
    retrieval_config: dict[str, Any] = {}

    # Runtime
    debug: bool = _env("BIOLIT_DEBUG", "false").lower() == "true"

    class Config:
        arbitrary_types_allowed = True


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    models_config = _load_yaml(CONFIGS_DIR / "models.yaml")
    retrieval_config = _load_yaml(CONFIGS_DIR / "retrieval.yaml")

    ollama_cfg = models_config.get("ollama", {})
    settings = Settings(
        models_config=models_config,
        retrieval_config=retrieval_config,
        ollama_host=_env("OLLAMA_HOST", ollama_cfg.get("host", "http://localhost:11434")),
        ollama_model=_env(
            "OLLAMA_MODEL",
            ollama_cfg.get("serving_model_name") or ollama_cfg.get("model_name", "qwen2.5:3b"),
        ),
    )

    for d in [settings.data_dir, settings.documents_dir, settings.index_dir,
              settings.metadata_dir, settings.results_dir]:
        d.mkdir(parents=True, exist_ok=True)

    return settings
