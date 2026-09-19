"""Aggregates per-query CitationMetrics into a GenerationMetrics summary for an eval set."""
from __future__ import annotations

from backend.app.evaluation.citation_metrics import average_citation_metrics
from backend.app.models.schemas import GenerationMetrics, QueryResponse

NOTE = (
    "These faithfulness/citation numbers come from an automated LLM-based claim verifier "
    "and simple citation bookkeeping; they are approximate and not a substitute for human "
    "review of the underlying answers."
)


def compute_generation_metrics(responses: list[QueryResponse]) -> GenerationMetrics:
    citation_metrics = average_citation_metrics(responses)
    return GenerationMetrics(
        citation_precision=citation_metrics.citation_precision,
        citation_coverage=citation_metrics.citation_coverage,
        faithfulness=citation_metrics.faithfulness,
        unsupported_claim_rate=citation_metrics.unsupported_claim_rate,
        answer_relevance_approx=None,
        note=NOTE,
    )
