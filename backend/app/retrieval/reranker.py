"""Cross-encoder reranker.

Uses sentence_transformers.CrossEncoder (simpler than a manual
AutoModelForSequenceClassification load) to score (query, chunk_text) pairs
with BAAI/bge-reranker-base.
"""
from __future__ import annotations

from backend.app.models.schemas import Chunk


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


class Reranker:
    def __init__(self, model_name: str, device: str = "auto", max_length: int = 512):
        from sentence_transformers import CrossEncoder  # lazy: heavy/optional import

        self.device = _resolve_device(device)
        self.model = CrossEncoder(model_name, device=self.device, max_length=max_length)

    def rerank(self, query: str, candidates: list[tuple[Chunk, float]], top_k: int) -> list[tuple[Chunk, float]]:
        if not candidates:
            return []
        pairs = [(query, chunk.text) for chunk, _ in candidates]
        scores = self.model.predict(pairs)
        scored = [(chunk, float(score)) for (chunk, _), score in zip(candidates, scores)]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]
