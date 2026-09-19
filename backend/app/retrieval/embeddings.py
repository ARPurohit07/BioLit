"""Local sentence-embedding wrapper around sentence-transformers (BAAI/bge-small-en-v1.5)."""
from __future__ import annotations

import numpy as np


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


class EmbeddingModel:
    def __init__(self, model_name: str, device: str = "auto", normalize: bool = True):
        from sentence_transformers import SentenceTransformer  # lazy: heavy/optional import

        self.model_name = model_name
        self.normalize = normalize
        self.device = _resolve_device(device)
        self._model = SentenceTransformer(model_name, device=self.device)

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
        )
        return np.asarray(embeddings, dtype=np.float32)
