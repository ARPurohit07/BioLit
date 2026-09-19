"""Standard retrieval-quality metrics (Recall@k, MRR, nDCG@10) with binary relevance.

Each result dict supports two ways of expressing relevance, matched with a
chunk_id-first, (document_id, page) fallback strategy:
    {
        "retrieved_chunk_ids": ["c1", "c2", ...],           # ranked, best first
        "relevant_chunk_ids": ["c2", "c5"],                  # optional
        "retrieved_doc_pages": [("d1", 3), ("d1", 4), ...],  # optional, ranked
        "relevant_doc_pages": [("d1", 4)],                   # optional
    }
"""
from __future__ import annotations

import math

from backend.app.models.schemas import RetrievalMetrics


def _hits(retrieved: list, relevant: set) -> list[bool]:
    return [item in relevant for item in retrieved]


def _result_hits(result: dict) -> list[bool]:
    retrieved_ids = result.get("retrieved_chunk_ids") or []
    relevant_ids = set(result.get("relevant_chunk_ids") or [])

    retrieved_pages = result.get("retrieved_doc_pages") or []
    relevant_pages = set(tuple(p) for p in (result.get("relevant_doc_pages") or []))

    if retrieved_ids and relevant_ids:
        return _hits(retrieved_ids, relevant_ids)
    if retrieved_pages and relevant_pages:
        return _hits([tuple(p) for p in retrieved_pages], relevant_pages)
    # No usable relevance judgments for this query.
    retrieved = retrieved_ids or retrieved_pages
    return [False] * len(retrieved)


def _recall_at_k(hits: list[bool], k: int, num_relevant: int) -> float:
    if num_relevant == 0:
        return 0.0
    return sum(hits[:k]) / num_relevant


def _mrr(hits: list[bool]) -> float:
    for rank, hit in enumerate(hits, start=1):
        if hit:
            return 1.0 / rank
    return 0.0


def _ndcg_at_k(hits: list[bool], k: int) -> float:
    dcg = sum((1.0 / math.log2(i + 2)) for i, hit in enumerate(hits[:k]) if hit)
    num_relevant_in_topk = min(sum(hits), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(num_relevant_in_topk))
    if idcg == 0:
        return 0.0
    return dcg / idcg


def compute_retrieval_metrics(results: list[dict]) -> RetrievalMetrics:
    if not results:
        return RetrievalMetrics(
            recall_at_1=0.0, recall_at_3=0.0, recall_at_5=0.0, recall_at_10=0.0,
            mrr=0.0, ndcg_at_10=0.0, num_queries=0,
        )

    r1s, r3s, r5s, r10s, mrrs, ndcgs = [], [], [], [], [], []

    for result in results:
        hits = _result_hits(result)
        num_relevant = len(set(result.get("relevant_chunk_ids") or [])) or \
            len(set(tuple(p) for p in (result.get("relevant_doc_pages") or [])))

        if num_relevant == 0:
            continue

        r1s.append(_recall_at_k(hits, 1, num_relevant))
        r3s.append(_recall_at_k(hits, 3, num_relevant))
        r5s.append(_recall_at_k(hits, 5, num_relevant))
        r10s.append(_recall_at_k(hits, 10, num_relevant))
        mrrs.append(_mrr(hits))
        ndcgs.append(_ndcg_at_k(hits, 10))

    n = len(r1s)
    if n == 0:
        return RetrievalMetrics(
            recall_at_1=0.0, recall_at_3=0.0, recall_at_5=0.0, recall_at_10=0.0,
            mrr=0.0, ndcg_at_10=0.0, num_queries=len(results),
        )

    return RetrievalMetrics(
        recall_at_1=sum(r1s) / n,
        recall_at_3=sum(r3s) / n,
        recall_at_5=sum(r5s) / n,
        recall_at_10=sum(r10s) / n,
        mrr=sum(mrrs) / n,
        ndcg_at_10=sum(ndcgs) / n,
        num_queries=n,
    )
