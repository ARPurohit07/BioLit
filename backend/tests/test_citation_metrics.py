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


# --- NOT_VERIFIED: "not checked" must never read as "wrong" -------------------------------------------------------


def test_extracted_claims_start_out_not_verified():
    from backend.app.models.schemas import EvidenceItem
    from backend.app.verification.claims import ClaimExtractor

    ev = [EvidenceItem(citation_id=1, chunk_id="c", document_id="d", document_title="P", page_number=1,
                       section="Results", text="t")]
    claim = ClaimExtractor().extract("Drug X reduced tumor size in mice [1].", ev)[0]
    assert claim.status == ClaimStatus.NOT_VERIFIED


def test_unverified_claims_report_coverage_but_no_faithfulness_numbers():
    claims = [_claim(ClaimStatus.NOT_VERIFIED, [1]), _claim(ClaimStatus.NOT_VERIFIED, [2]), _claim(ClaimStatus.NOT_VERIFIED, [])]
    metrics = compute_citation_metrics(claims)
    assert metrics.citation_coverage == 2 / 3        # needs no verifier
    assert metrics.citation_precision is None        # not 0%: nothing was checked
    assert metrics.faithfulness is None
    assert metrics.unsupported_claim_rate is None
    assert metrics.verified_claims == 0 and metrics.total_claims == 3 and metrics.total_citations == 2


def test_only_verified_claims_count_toward_faithfulness():
    claims = [
        _claim(ClaimStatus.SUPPORTED, [1]),
        _claim(ClaimStatus.UNSUPPORTED, [2]),
        _claim(ClaimStatus.NOT_VERIFIED, [3]),   # must not count against the answer
        _claim(ClaimStatus.NOT_VERIFIED, []),
    ]
    metrics = compute_citation_metrics(claims)
    assert metrics.verified_claims == 2
    assert metrics.faithfulness == 1 / 2
    assert metrics.unsupported_claim_rate == 1 / 2
    assert metrics.citation_precision == 1 / 2       # of the 2 verified cited claims, 1 is supported
    assert metrics.citation_coverage == 3 / 4        # coverage still uses every claim


def test_a_verified_answer_with_no_citations_has_zero_precision_not_none():
    metrics = compute_citation_metrics([_claim(ClaimStatus.UNSUPPORTED, []) for _ in range(2)])
    assert metrics.citation_precision == 0.0         # verified, and nothing cited: a real 0
    assert metrics.verified_claims == 2


def test_averaging_skips_responses_that_measured_nothing():
    from backend.app.evaluation.citation_metrics import average_metrics_list

    unverified = compute_citation_metrics([_claim(ClaimStatus.NOT_VERIFIED, [1])])
    verified = compute_citation_metrics([_claim(ClaimStatus.SUPPORTED, [1]), _claim(ClaimStatus.UNSUPPORTED, [2])])
    only_unverified = average_metrics_list([unverified, unverified])
    assert only_unverified.faithfulness is None and only_unverified.citation_coverage == 1.0
    mixed = average_metrics_list([unverified, verified])
    assert mixed.faithfulness == 0.5                 # averaged over the one response that measured it
    assert mixed.citation_coverage == 1.0
    assert mixed.verified_claims == 2
