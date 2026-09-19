"""Unit tests for citation/faithfulness metric math (no LLM calls)."""
from __future__ import annotations

from backend.app.models.schemas import Claim, ClaimStatus
from backend.app.verification.citation_validator import compute_citation_metrics


def _claim(status: ClaimStatus, citation_ids: list[int]) -> Claim:
    return Claim(claim_id="x", text="t", citation_ids=citation_ids, status=status)


def test_empty_claims_returns_zeros_not_crash():
    metrics = compute_citation_metrics([])
    assert metrics.citation_precision == 0.0
    assert metrics.citation_coverage == 0.0
    assert metrics.faithfulness == 0.0
    assert metrics.unsupported_claim_rate == 0.0
    assert metrics.total_claims == 0
    assert metrics.total_citations == 0


def test_all_supported_and_cited():
    claims = [_claim(ClaimStatus.SUPPORTED, [1]) for _ in range(4)]
    metrics = compute_citation_metrics(claims)
    assert metrics.citation_precision == 1.0
    assert metrics.citation_coverage == 1.0
    assert metrics.faithfulness == 1.0
    assert metrics.unsupported_claim_rate == 0.0
    assert metrics.total_claims == 4
    assert metrics.total_citations == 4


def test_mixed_statuses():
    claims = [
        _claim(ClaimStatus.SUPPORTED, [1]),
        _claim(ClaimStatus.PARTIALLY_SUPPORTED, [2]),
        _claim(ClaimStatus.UNSUPPORTED, []),
        _claim(ClaimStatus.CONTRADICTED, [3]),
    ]
    metrics = compute_citation_metrics(claims)
    # cited claims: 3 (all but the uncited UNSUPPORTED one)
    assert metrics.citation_coverage == 3 / 4
    # of the 3 cited claims, 2 are SUPPORTED/PARTIALLY_SUPPORTED
    assert metrics.citation_precision == 2 / 3
    # faithfulness = fraction strictly SUPPORTED
    assert metrics.faithfulness == 1 / 4
    # unsupported/contradicted = 2 of 4
    assert metrics.unsupported_claim_rate == 2 / 4
    assert metrics.total_claims == 4
    assert metrics.total_citations == 3


def test_cited_but_unsupported_hurts_precision_not_coverage():
    claims = [_claim(ClaimStatus.UNSUPPORTED, [1]), _claim(ClaimStatus.SUPPORTED, [2])]
    metrics = compute_citation_metrics(claims)
    assert metrics.citation_coverage == 1.0  # both have citations
    assert metrics.citation_precision == 0.5  # only one of them is actually supported


def test_no_citations_at_all_gives_zero_precision_and_coverage():
    claims = [_claim(ClaimStatus.UNSUPPORTED, []) for _ in range(3)]
    metrics = compute_citation_metrics(claims)
    assert metrics.citation_coverage == 0.0
    assert metrics.citation_precision == 0.0
