"""Aggregate citation/faithfulness metrics from a list of verified Claims."""
from __future__ import annotations

from backend.app.models.schemas import Claim, ClaimStatus, CitationMetrics

_SUPPORTED_LIKE = {ClaimStatus.SUPPORTED, ClaimStatus.PARTIALLY_SUPPORTED}
_UNSUPPORTED_LIKE = {ClaimStatus.UNSUPPORTED, ClaimStatus.CONTRADICTED}


def compute_citation_metrics(claims: list[Claim]) -> CitationMetrics:
    total_claims = len(claims)
    cited_claims = [c for c in claims if c.citation_ids]
    total_citations = sum(len(c.citation_ids) for c in claims)

    if total_claims == 0:
        return CitationMetrics(
            citation_precision=0.0,
            citation_coverage=0.0,
            faithfulness=0.0,
            unsupported_claim_rate=0.0,
            total_claims=0,
            total_citations=0,
        )

    citation_coverage = len(cited_claims) / total_claims

    if cited_claims:
        supported_cited = [c for c in cited_claims if c.status in _SUPPORTED_LIKE]
        citation_precision = len(supported_cited) / len(cited_claims)
    else:
        citation_precision = 0.0

    faithfulness = sum(1 for c in claims if c.status == ClaimStatus.SUPPORTED) / total_claims
    unsupported_claim_rate = sum(1 for c in claims if c.status in _UNSUPPORTED_LIKE) / total_claims

    return CitationMetrics(
        citation_precision=citation_precision,
        citation_coverage=citation_coverage,
        faithfulness=faithfulness,
        unsupported_claim_rate=unsupported_claim_rate,
        total_claims=total_claims,
        total_citations=total_citations,
    )
