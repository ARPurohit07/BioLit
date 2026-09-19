"""Aggregation helpers on top of citation_validator, for the evaluation API and scripts/benchmark.py."""
from __future__ import annotations

from backend.app.models.schemas import Claim, CitationMetrics, QueryResponse
from backend.app.verification.citation_validator import compute_citation_metrics


def citation_metrics_for_response(response: QueryResponse) -> CitationMetrics:
    return compute_citation_metrics(response.claims)


def average_citation_metrics(responses: list[QueryResponse]) -> CitationMetrics:
    """Average CitationMetrics across many QueryResponses (e.g. a benchmark run)."""
    all_claims: list[Claim] = []
    for r in responses:
        all_claims.extend(r.claims)
    return compute_citation_metrics(all_claims)


def average_metrics_list(metrics: list[CitationMetrics]) -> CitationMetrics:
    """Average a list of already-computed CitationMetrics (simple mean, not claim-weighted)."""
    if not metrics:
        return CitationMetrics(
            citation_precision=0.0, citation_coverage=0.0, faithfulness=0.0,
            unsupported_claim_rate=0.0, total_claims=0, total_citations=0,
        )
    n = len(metrics)
    return CitationMetrics(
        citation_precision=sum(m.citation_precision for m in metrics) / n,
        citation_coverage=sum(m.citation_coverage for m in metrics) / n,
        faithfulness=sum(m.faithfulness for m in metrics) / n,
        unsupported_claim_rate=sum(m.unsupported_claim_rate for m in metrics) / n,
        total_claims=sum(m.total_claims for m in metrics),
        total_citations=sum(m.total_citations for m in metrics),
    )
