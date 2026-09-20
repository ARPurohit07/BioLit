"""The LLM-free grounding estimate: what it rewards, what it caps, and that it is wired into claims and metrics."""
from __future__ import annotations

from backend.app.models.schemas import Claim, ClaimStatus, EvidenceItem
from backend.app.verification.claims import ClaimExtractor
from backend.app.verification.citation_validator import compute_citation_metrics
from backend.app.verification.grounding import NONE, STRONG, WEAK, count_flagged, estimate

EVIDENCE = (
    "The MSDA plug-in improves zero-shot drug response prediction on the GDSCv2 dataset by adapting multiple "
    "target domain branches, reaching an AUC of 0.83 across 120 cell lines in the inductive setting."
)
OTHER = "Serendipitous discovery of repurposed drugs is inconsistent, and databases of drug attributes are large."


def test_an_uncited_claim_is_never_grounded():
    g = estimate("The MSDA plug-in improves zero-shot drug response prediction.", [])
    assert (g.label, g.score) == (NONE, 0.0)
    assert "cites no evidence" in g.reason


def test_a_claim_copied_from_the_evidence_is_strongly_grounded():
    g = estimate("SUPPORTED CLAIM: The MSDA plug-in improves zero-shot drug response prediction on GDSCv2 [1].", [EVIDENCE])
    assert g.label == STRONG and g.score >= 0.7


def test_an_unrelated_claim_is_not_grounded():
    g = estimate("Blockchain ledgers guarantee reproducible randomization for every participant.", [EVIDENCE])
    assert g.label == NONE


def test_a_number_missing_from_the_cited_evidence_caps_the_label_at_none():
    g = estimate("The MSDA plug-in reaches an AUC of 0.97 across 120 cell lines on the GDSCv2 dataset.", [EVIDENCE])
    assert g.label == NONE and "0.97" in g.reason


def test_a_number_in_the_wrong_setting_is_downgraded_from_strong():
    # 0.83 appears in the evidence, but the claim attaches it to a setting the evidence never mentions near it.
    g = estimate("Transductive benchmark ranking reached 0.83 accuracy for pediatric patients receiving therapy.", [EVIDENCE])
    assert g.label != STRONG


def test_partial_overlap_is_weak():
    g = estimate("The MSDA plug-in improves prediction while also reducing hospital costs and clinical trial durations.", [EVIDENCE])
    assert g.label in (WEAK, NONE) and g.label != STRONG


def test_an_unrelated_extra_citation_downgrades_an_otherwise_strong_claim():
    claim = "The MSDA plug-in improves zero-shot drug response prediction on the GDSCv2 dataset."
    assert estimate(claim, [EVIDENCE]).label == STRONG
    padded = estimate(claim, [EVIDENCE, OTHER])
    assert padded.label == WEAK and "one of the cited blocks" in padded.reason


def test_citation_markers_and_labels_do_not_count_as_wording():
    plain = estimate("The MSDA plug-in improves zero-shot drug response prediction on the GDSCv2 dataset.", [EVIDENCE])
    dressed = estimate("SUPPORTED CLAIM: The MSDA plug-in improves zero-shot drug response prediction on the GDSCv2 dataset [1][2].", [EVIDENCE])
    assert plain.score == dressed.score


def test_the_extractor_annotates_every_claim_and_metrics_count_the_flagged_ones_without_a_verifier():
    ev = [EvidenceItem(citation_id=1, chunk_id="c1", document_id="d", document_title="P", page_number=1,
                       section="Results", text=EVIDENCE)]
    answer = ("The MSDA plug-in improves zero-shot drug response prediction on the GDSCv2 dataset [1].\n"
              "Quantum annealing hardware dramatically accelerates protein folding simulations everywhere [1].")
    claims = ClaimExtractor().extract(answer, ev)
    assert [c.grounding for c in claims] == [STRONG, NONE]
    assert all(c.status == ClaimStatus.NOT_VERIFIED for c in claims)      # the estimate is not a verdict
    metrics = compute_citation_metrics(claims)
    assert metrics.flagged_claims == 1                                    # only the unrelated claim is flagged
    assert metrics.faithfulness is None                                   # still "not measured"


def test_count_flagged_ignores_claims_that_were_never_estimated():
    assert count_flagged([]) == 0
    assert count_flagged([Claim(claim_id="x", text="t")]) == 0
