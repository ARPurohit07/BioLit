"""Reciprocal Rank Fusion and the mode-aware hybrid retriever.

Reranking is intentionally NOT performed here — callers apply
backend.app.retrieval.reranker.Reranker on top of retrieve()'s output.
"""
from __future__ import annotations

from backend.app.models.schemas import Chunk


def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[Chunk, float]]], k: int = 60
) -> list[tuple[Chunk, float]]:
    fused_scores: dict[str, float] = {}
    chunk_by_id: dict[str, Chunk] = {}

    for ranked_list in ranked_lists:
        for rank, (chunk, _score) in enumerate(ranked_list, start=1):
            fused_scores[chunk.chunk_id] = fused_scores.get(chunk.chunk_id, 0.0) + 1.0 / (k + rank)
            chunk_by_id.setdefault(chunk.chunk_id, chunk)

    ordered = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    return [(chunk_by_id[cid], score) for cid, score in ordered]


class HybridRetriever:
    def __init__(self, vector_store, bm25_index, embedding_model, config: dict):
        self.vector_store = vector_store
        self.bm25_index = bm25_index
        self.embedding_model = embedding_model
        self.config = config

    def retrieve(self, query: str, mode: str,
                  document_ids: list[str] | None = None) -> list[tuple[Chunk, float]]:
        query_embedding = self.embedding_model.encode([query])[0]

        if mode == "fast":
            top_k = self.config.get("modes", {}).get("fast", {}).get("top_k", 5)
            return self.vector_store.search(query_embedding, top_k, document_ids=document_ids)

        hybrid_cfg = self.config.get("hybrid", {})
        dense_top_k = hybrid_cfg.get("dense_top_k", 20)
        bm25_top_k = hybrid_cfg.get("bm25_top_k", 20)
        rrf_k = hybrid_cfg.get("rrf_k", 60)
        final_top_k = hybrid_cfg.get("final_top_k", 10)

        dense_results = self.vector_store.search(query_embedding, dense_top_k, document_ids=document_ids)
        bm25_results = self.bm25_index.search(query, bm25_top_k, document_ids=document_ids)

        fused = reciprocal_rank_fusion([dense_results, bm25_results], k=rrf_k)
        return fused[:final_top_k]
