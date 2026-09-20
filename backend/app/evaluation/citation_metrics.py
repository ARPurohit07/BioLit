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


def _mean_of_measured(values: list) -> float | None:
    measured = [v for v in values if v is not None]
    return sum(measured) / len(measured) if measured else None


def average_metrics_list(metrics: list[CitationMetrics]) -> CitationMetrics:
    """Average already-computed CitationMetrics (simple mean, not claim-weighted).

    Precision / faithfulness / unsupported rate are averaged over the responses that measured them; if none did, the
    result is None (not measured) rather than a misleading 0.
    """
    if not metrics:
        return CitationMetrics(
            citation_precision=0.0, citation_coverage=0.0, faithfulness=0.0,
            unsupported_claim_rate=0.0, verified_claims=0, total_claims=0, total_citations=0,
        )
    return CitationMetrics(
        citation_precision=_mean_of_measured([m.citation_precision for m in metrics]),
        citation_coverage=sum(m.citation_coverage for m in metrics) / len(metrics),
        faithfulness=_mean_of_measured([m.faithfulness for m in metrics]),
        unsupported_claim_rate=_mean_of_measured([m.unsupported_claim_rate for m in metrics]),
        verified_claims=sum(m.verified_claims for m in metrics),
        total_claims=sum(m.total_claims for m in metrics),
        total_citations=sum(m.total_citations for m in metrics),
    )
