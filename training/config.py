"""Tiny YAML config loading helper shared by training/ scripts.

Kept separate from the application config (backend/app/config/settings.py)
so training/ has no import dependency on backend/.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML file into a plain dict. Raises FileNotFoundError with a
    clear message if the path doesn't exist, since training scripts should
    never silently fall back to hard-coded defaults for these files."""
    import yaml  # imported lazily so modules that only need other helpers

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}. "
            "Expected e.g. configs/training.yaml or configs/models.yaml at the repo root."
        )
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a top-level mapping in {config_path}, got {type(data)}")
    return data


def repo_root() -> Path:
    """Best-effort repo root: the parent of this training/ directory."""
    return Path(__file__).resolve().parent.parent
