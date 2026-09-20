"""Cross-encoder reranker.

Uses sentence_transformers.CrossEncoder (simpler than a manual
AutoModelForSequenceClassification load) to score (query, chunk_text) pairs
with BAAI/bge-reranker-base.

GPU handling
------------
On a small laptop GPU the reranker shares the card with the Ollama-served LLM, so it must not assume memory is free:

* CrossEncoder only *records* the device it is given and moves the weights inside predict(), i.e. lazily, on the
  first query. That is where a CUDA out-of-memory error used to surface (as a 500 on the user's first Balanced or
  High-Faithfulness request). We load on the CPU and place the model ourselves at startup instead.
* On the GPU the model runs in half precision (about 0.6 GB instead of about 1.1 GB).
* If the GPU cannot hold it (at startup or later), we retry with a smaller batch, then fall back to the CPU
  permanently, log a warning, and keep answering. Slower, but it never fails a request over GPU memory.
"""
from __future__ import annotations

import logging

from backend.app.models.schemas import Chunk

logger = logging.getLogger("biolit")


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _is_oom(exc: BaseException) -> bool:
    try:
        import torch

        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
    except Exception:
        pass
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


class Reranker:
    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        max_length: int = 512,
        batch_size: int = 16,
        half_precision: bool = True,
        fallback_to_cpu: bool = True,
    ):
        from sentence_transformers import CrossEncoder  # lazy: heavy/optional import

        self.batch_size = max(1, int(batch_size))
        self.fallback_to_cpu = fallback_to_cpu
        self.fallback_reason: str | None = None
        self.device = "cpu"
        self.dtype = "fp32"
        self.model = CrossEncoder(model_name, device="cpu", max_length=max_length)

        target = _resolve_device(device)
        if target.startswith("cuda"):
            self._place_on_gpu(target, half_precision)

    # ------------------------------------------------------------------ placement

    def _place_on_gpu(self, target: str, half_precision: bool) -> None:
        try:
            import torch

            if half_precision:
                self.model.model.half()  # convert on the CPU first, then move only half the bytes
            self.model.model.to(target)
            self.model._target_device = torch.device(target)  # predict() re-moves to this device on every call
            self.device, self.dtype = target, ("fp16" if half_precision else "fp32")
        except Exception as exc:
            if not (self.fallback_to_cpu and _is_oom(exc)):
                raise
            self._fall_back_to_cpu(f"out of GPU memory placing the model on {target}")

    def _fall_back_to_cpu(self, reason: str) -> None:
        import torch

        logger.warning("Reranker: %s; scoring on the CPU instead (slower, but it will not fail).", reason)
        self.model.model.to("cpu").float()
        self.model._target_device = torch.device("cpu")
        self.device, self.dtype, self.fallback_reason = "cpu", "fp32", reason
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def describe(self) -> str:
        """Human-readable placement, e.g. 'cuda (fp16)', 'cpu', or 'cpu (fell back from GPU: ...)'."""
        if self.fallback_reason:
            return f"cpu (fell back from GPU: {self.fallback_reason})"
        return self.device if self.device == "cpu" else f"{self.device} ({self.dtype})"

    # ------------------------------------------------------------------ scoring

    def _predict(self, pairs: list[tuple[str, str]], batch_size: int):
        return self.model.predict(pairs, batch_size=batch_size, show_progress_bar=False)

    def rerank(self, query: str, candidates: list[tuple[Chunk, float]], top_k: int) -> list[tuple[Chunk, float]]:
        if not candidates:
            return []
        pairs = [(query, chunk.text) for chunk, _ in candidates]
        try:
            scores = self._predict(pairs, self.batch_size)
        except Exception as exc:
            if self.device == "cpu" or not (self.fallback_to_cpu and _is_oom(exc)):
                raise
            scores = self._recover_from_oom(pairs)
        scored = [(chunk, float(score)) for (chunk, _), score in zip(candidates, scores)]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def _recover_from_oom(self, pairs: list[tuple[str, str]]):
        """Out of GPU memory mid-request: free the cache, retry with a smaller batch, then give up on the GPU."""
        import torch

        torch.cuda.empty_cache()
        smaller = max(1, self.batch_size // 4)
        try:
            return self._predict(pairs, smaller)
        except Exception as exc:
            if not _is_oom(exc):
                raise
        self._fall_back_to_cpu("out of GPU memory while scoring")
        return self._predict(pairs, self.batch_size)
