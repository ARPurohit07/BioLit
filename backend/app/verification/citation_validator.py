"""Aggregate citation/faithfulness metrics from a list of Claims.

Claims the verifier never looked at (`NOT_VERIFIED`, which is every claim in Fast and Balanced modes) must not be
scored as unsupported: that would read as "wrong" when the truth is "not checked". Coverage (does the claim carry a
citation?) needs no verifier, so it always uses every claim. Precision, faithfulness and the unsupported rate use
only the verified claims, and are None (not measured) when there are none.
"""
from __future__ import annotations

from backend.app.models.schemas import Claim, ClaimStatus, CitationMetrics
from backend.app.verification.grounding import count_flagged

_SUPPORTED_LIKE = {ClaimStatus.SUPPORTED, ClaimStatus.PARTIALLY_SUPPORTED}
_UNSUPPORTED_LIKE = {ClaimStatus.UNSUPPORTED, ClaimStatus.CONTRADICTED}


def compute_citation_metrics(claims: list[Claim]) -> CitationMetrics:
    total_claims = len(claims)
    total_citations = sum(len(c.citation_ids) for c in claims)

    if total_claims == 0:
        return CitationMetrics(
            citation_precision=0.0,
            citation_coverage=0.0,
            faithfulness=0.0,
            unsupported_claim_rate=0.0,
            verified_claims=0,
            total_claims=0,
            total_citations=0,
        )

    citation_coverage = sum(1 for c in claims if c.citation_ids) / total_claims
    flagged_claims = count_flagged(claims)

    verified = [c for c in claims if c.status != ClaimStatus.NOT_VERIFIED]
    if not verified:
        return CitationMetrics(
            citation_precision=None,
            citation_coverage=citation_coverage,
            faithfulness=None,
            unsupported_claim_rate=None,
            verified_claims=0,
            flagged_claims=flagged_claims,
            total_claims=total_claims,
            total_citations=total_citations,
        )

    cited_verified = [c for c in verified if c.citation_ids]
    citation_precision = (
        sum(1 for c in cited_verified if c.status in _SUPPORTED_LIKE) / len(cited_verified) if cited_verified else 0.0
    )
    faithfulness = sum(1 for c in verified if c.status == ClaimStatus.SUPPORTED) / len(verified)
    unsupported_claim_rate = sum(1 for c in verified if c.status in _UNSUPPORTED_LIKE) / len(verified)

    return CitationMetrics(
        citation_precision=citation_precision,
        citation_coverage=citation_coverage,
        faithfulness=faithfulness,
        unsupported_claim_rate=unsupported_claim_rate,
        verified_claims=len(verified),
        flagged_claims=flagged_claims,
        total_claims=total_claims,
        total_citations=total_citations,
    )
