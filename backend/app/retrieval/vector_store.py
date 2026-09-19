"""FAISS-backed dense vector store with a JSONL sidecar for chunk metadata.

FAISS itself has no notion of document-id filtering, so `search` over-fetches
candidates from the flat index and post-filters by document_id, widening the
fetch window if too few results survive the filter.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from backend.app.models.schemas import Chunk

try:
    import faiss
except ImportError:  # pragma: no cover - exercised only when faiss is missing
    faiss = None


class FAISSVectorStore:
    def __init__(self, dim: int, index_path: str, metadata_path: str):
        if faiss is None:
            raise ImportError("faiss is required for FAISSVectorStore — pip install faiss-cpu")
        self.dim = dim
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)
        self.index = faiss.IndexFlatIP(dim)
        self._chunks: list[Chunk] = []

    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        if not chunks:
            return
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        if embeddings.shape[0] != len(chunks):
            raise ValueError("embeddings and chunks must have the same length")
        if embeddings.shape[1] != self.dim:
            raise ValueError(f"expected embedding dim {self.dim}, got {embeddings.shape[1]}")
        self.index.add(embeddings)
        self._chunks.extend(chunks)

    def search(self, query_embedding: np.ndarray, top_k: int,
               document_ids: list[str] | None = None) -> list[tuple[Chunk, float]]:
        if len(self) == 0 or top_k <= 0:
            return []
        query = np.ascontiguousarray(query_embedding, dtype=np.float32).reshape(1, -1)

        if document_ids is None:
            k = min(top_k, len(self))
            scores, idxs = self.index.search(query, k)
            return self._to_results(idxs[0], scores[0])

        allowed = set(document_ids)
        fetch_k = min(len(self), max(top_k * 4, top_k + 50))
        results: list[tuple[Chunk, float]] = []
        while True:
            scores, idxs = self.index.search(query, fetch_k)
            results = [
                (self._chunks[idx], float(score))
                for idx, score in zip(idxs[0], scores[0])
                if idx >= 0 and self._chunks[idx].document_id in allowed
            ]
            if len(results) >= top_k or fetch_k >= len(self):
                break
            fetch_k = min(len(self), fetch_k * 2)
        return results[:top_k]

    def _to_results(self, idxs: np.ndarray, scores: np.ndarray) -> list[tuple[Chunk, float]]:
        return [(self._chunks[idx], float(score)) for idx, score in zip(idxs, scores) if idx >= 0]

    def save(self) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            for chunk in self._chunks:
                f.write(json.dumps(chunk.model_dump()) + "\n")

    def load(self) -> None:
        if self.index_path.exists():
            self.index = faiss.read_index(str(self.index_path))
        else:
            self.index = faiss.IndexFlatIP(self.dim)

        self._chunks = []
        if self.metadata_path.exists():
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self._chunks.append(Chunk(**json.loads(line)))

    def remove_document(self, document_id: str) -> None:
        if len(self) == 0:
            return
        vectors = self.index.reconstruct_n(0, len(self))
        keep_mask = np.array([c.document_id != document_id for c in self._chunks], dtype=bool)
        remaining_chunks = [c for c, keep in zip(self._chunks, keep_mask) if keep]

        self.index = faiss.IndexFlatIP(self.dim)
        if remaining_chunks:
            self.index.add(np.ascontiguousarray(vectors[keep_mask], dtype=np.float32))
        self._chunks = remaining_chunks

    def __len__(self) -> int:
        return self.index.ntotal
