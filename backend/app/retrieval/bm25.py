"""BM25 sparse index over chunk text using rank_bm25.BM25Okapi.

document_id filtering is done post-hoc (score against the full corpus, then
filter) since BM25Okapi has no native subset-search support.
"""
from __future__ import annotations

import pickle
import re
from pathlib import Path

from backend.app.models.schemas import Chunk

try:
    from rank_bm25 import BM25Okapi
except ImportError:  # pragma: no cover - exercised only when rank_bm25 is missing
    BM25Okapi = None

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    def __init__(self, index_path: str, k1: float = 1.5, b: float = 0.75):
        if BM25Okapi is None:
            raise ImportError("rank_bm25 is required for BM25Index — pip install rank_bm25")
        self.index_path = Path(index_path)
        self.k1 = k1
        self.b = b
        self._chunks: list[Chunk] = []
        self._bm25: BM25Okapi | None = None

    def add(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        self._chunks.extend(chunks)
        self._rebuild()

    def _rebuild(self) -> None:
        if not self._chunks:
            self._bm25 = None
            return
        tokenized = [_tokenize(c.text) for c in self._chunks]
        self._bm25 = BM25Okapi(tokenized, k1=self.k1, b=self.b)

    def search(self, query: str, top_k: int,
               document_ids: list[str] | None = None) -> list[tuple[Chunk, float]]:
        if self._bm25 is None or top_k <= 0:
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        pairs = list(zip(self._chunks, scores))
        if document_ids is not None:
            allowed = set(document_ids)
            pairs = [(c, s) for c, s in pairs if c.document_id in allowed]
        pairs.sort(key=lambda x: x[1], reverse=True)
        return [(c, float(s)) for c, s in pairs[:top_k]]

    def save(self) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.index_path, "wb") as f:
            pickle.dump({"chunks": self._chunks, "k1": self.k1, "b": self.b}, f)

    def load(self) -> None:
        if self.index_path.exists():
            with open(self.index_path, "rb") as f:
                data = pickle.load(f)
            self._chunks = data.get("chunks", [])
            self.k1 = data.get("k1", self.k1)
            self.b = data.get("b", self.b)
        else:
            self._chunks = []
        self._rebuild()

    def remove_document(self, document_id: str) -> None:
        self._chunks = [c for c in self._chunks if c.document_id != document_id]
        self._rebuild()

    def __len__(self) -> int:
        return len(self._chunks)
