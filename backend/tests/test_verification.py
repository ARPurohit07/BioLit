"""Unit tests for claim extraction and citation-marker parsing. No LLM calls."""
from __future__ import annotations

from backend.app.models.schemas import ClaimStatus, EvidenceItem
from backend.app.verification.claims import ClaimExtractor
from backend.app.verification.verifier import ClaimVerifier


def _evidence(n: int) -> list[EvidenceItem]:
    return [
        EvidenceItem(
            citation_id=i,
            chunk_id=f"c{i}",
            document_id="doc1",
            document_title="Some Paper",
            page_number=i,
            section="Results",
            text=f"Evidence text {i}.",
        )
        for i in range(1, n + 1)
    ]


def test_extract_single_citation_marker():
    evidence = _evidence(2)
    answer = "Drug X reduced tumor size in mice [1]. The mechanism remains unclear [2]."
    claims = ClaimExtractor().extract(answer, evidence)
    assert len(claims) == 2
    assert claims[0].citation_ids == [1]
    assert claims[1].citation_ids == [2]


def test_extract_multi_citation_marker():
    evidence = _evidence(3)
    answer = "The effect was consistent across cohorts [1][3]."
    claims = ClaimExtractor().extract(answer, evidence)
    assert len(claims) == 1
    assert claims[0].citation_ids == [1, 3]


def test_extract_comma_style_marker():
    evidence = _evidence(3)
    answer = "The result was replicated [1, 3]."
    claims = ClaimExtractor().extract(answer, evidence)
    assert claims[0].citation_ids == [1, 3]


def test_extract_statement_without_citation_still_becomes_claim():
    evidence = _evidence(1)
    answer = "This is an uncited assertion with no marker at all."
    claims = ClaimExtractor().extract(answer, evidence)
    assert len(claims) == 1
    assert claims[0].citation_ids == []


def test_extract_ignores_citation_ids_not_in_evidence():
    evidence = _evidence(1)
    answer = "This references a nonexistent source [9]."
    claims = ClaimExtractor().extract(answer, evidence)
    assert claims[0].citation_ids == []


def test_extract_deterministic_claim_ids():
    evidence = _evidence(2)
    answer = "First sentence [1]. Second sentence [2]."
    claims = ClaimExtractor().extract(answer, evidence)
    assert [c.claim_id for c in claims] == ["claim_0", "claim_1"]


# --- Regressions from real model output: scaffolding must not become "claims" -----------------------


def test_citation_after_the_period_stays_with_its_sentence():
    # Real shape: "...is 19.32%. [1][4] SUPPORTED CLAIM" used to become two claims, the second being "[1][4]".
    claims = ClaimExtractor().extract("MSN-DDI improves accuracy by 19.32%. [1][4] SUPPORTED CLAIM", _evidence(4))
    assert len(claims) == 1
    assert claims[0].citation_ids == [1, 4]
    assert "SUPPORTED CLAIM" not in claims[0].text


def test_citation_only_line_attaches_to_the_claim_above():
    answer = "The two strategies are S1 and S2 partitions.\n\nSUPPORTED CLAIM: [3][4]"
    claims = ClaimExtractor().extract(answer, _evidence(4))
    assert len(claims) == 1
    assert claims[0].citation_ids == [3, 4]


def test_bare_marker_is_not_a_claim():
    assert ClaimExtractor().extract("[3]", _evidence(3)) == []


def test_headings_and_lead_ins_are_not_claims():
    answer = "## Summary\nThe main findings are:\n**Key results**\n- Drug X reduced tumor size in mice [1]."
    claims = ClaimExtractor().extract(answer, _evidence(1))
    assert [c.citation_ids for c in claims] == [[1]]


def test_statement_starting_with_a_number_is_preserved():
    claims = ClaimExtractor().extract("19.32% relative improvement was reported [1].", _evidence(1))
    assert claims[0].text.startswith("19.32%")


def test_labels_are_stripped_from_claim_text():
    claims = ClaimExtractor().extract("SUPPORTED CLAIM: Drug X reduced tumor size in mice [1].", _evidence(1))
    assert claims[0].text == "Drug X reduced tumor size in mice [1]."


def test_uncited_abstention_is_a_caveat_not_a_claim_but_cited_one_is():
    assert ClaimExtractor().extract("LIMITATION: The evidence does not specify the dosing regimen.", _evidence(1)) == []
    cited = ClaimExtractor().extract("The evidence does not specify the dosing regimen [1].", _evidence(1))
    assert len(cited) == 1


def test_short_cited_statement_is_still_a_claim():
    assert len(ClaimExtractor().extract("First sentence [1].", _evidence(1))) == 1


def test_inline_unsupported_note_is_removed_but_neighbouring_claims_survive():
    answer = (
        "Drug X reduced tumor size in mice [1]. *(unsupported — could not be verified against "
        "retrieved evidence)* Drug Y improved survival in trials [2]."
    )
    claims = ClaimExtractor().extract(answer, _evidence(2))
    assert [c.citation_ids for c in claims] == [[1], [2]]
    assert all("unsupported" not in c.text for c in claims)


def test_regeneration_echo_is_stripped_before_extraction():
    from backend.app.verification.claims import strip_regeneration_scaffold

    echoed = (
        "The following statements from that answer could NOT be verified against the evidence "
        "(they are unsupported or contradicted):\n"
        "- Drug X cures every known disease\n"
        "- Drug Y has no side effects at all\n\n"
        "Drug X reduced tumor size in mice [1]."
    )
    cleaned = strip_regeneration_scaffold(echoed)
    assert cleaned == "Drug X reduced tumor size in mice [1]."
    assert [c.citation_ids for c in ClaimExtractor().extract(cleaned, _evidence(1))] == [[1]]


def test_verifier_short_circuits_when_no_citations():
    claim = ClaimExtractor().extract("An uncited claim with no marker.", _evidence(1))[0]
    verified = ClaimVerifier(ollama_client=None).verify(claim, _evidence(1))
    assert verified.status == ClaimStatus.UNSUPPORTED
    assert verified.verifier_rationale is not None
