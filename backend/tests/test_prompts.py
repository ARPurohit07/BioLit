"""Every task prompt must carry the same citation discipline (they share CITATION_RULES)."""
from __future__ import annotations

import pytest

from backend.app.generation import prompts
from backend.app.models.schemas import EvidenceItem


def _evidence() -> list[EvidenceItem]:
    return [
        EvidenceItem(citation_id=i, chunk_id=f"c{i}", document_id="d", document_title="Paper",
                     page_number=i, section="Results", text=f"Evidence text {i}.")
        for i in (1, 2)
    ]


# Synthesis tasks keep the labelled rules (SUPPORTED CLAIM / INTERPRETATION / LIMITATION); question answering has plainer ones.
BUILDERS = [
    ("summarize", lambda e: prompts.build_summarization_prompt(e)),
    ("compare", lambda e: prompts.build_comparison_prompt(e, "methodology")),
    ("literature_review", lambda e: prompts.build_literature_review_section_prompt(e, "drug repurposing")),
    ("research_gaps", lambda e: prompts.build_research_gap_prompt(e)),
    ("conflicts", lambda e: prompts.build_conflict_detection_prompt(e)),
]
QA_BUILDERS = [("qa", lambda e: prompts.build_qa_prompt("What did the study find?", e))]


@pytest.mark.parametrize("name,build", BUILDERS, ids=[b[0] for b in BUILDERS])
def test_every_prompt_carries_the_full_citation_rules(name, build):
    system, user = build(_evidence())
    for rule in (
        "inline [n] markers",
        "immediately before its final period",
        "Every factual sentence must carry at least one marker",
        "Never invent a citation number",
        "SUPPORTED CLAIM",
        "Do not repeat the question's wording",
        "insufficient evidence",
    ):
        assert rule in system, f"{name}: missing rule {rule!r}"
    assert "[1]" in user and "[2]" in user  # numbered evidence blocks are still formatted into the user turn


@pytest.mark.parametrize("name,build", QA_BUILDERS, ids=[b[0] for b in QA_BUILDERS])
def test_question_answering_keeps_citation_discipline_but_drops_the_labels(name, build):
    system, user = build(_evidence())
    for rule in (
        "inline [n] markers",
        "immediately before its final period",
        "Every factual sentence must carry at least one marker",
        "Never invent a citation number",
        "Do not repeat the question's wording",
        "insufficient evidence",
        "Stay close to its wording",
    ):
        assert rule in system, f"{name}: missing rule {rule!r}"
    assert "Do not label them" in system
    assert "only a citation marker is not an answer" in system     # a marker alone must never count as an answer
    assert "Explicitly label each claim" not in system       # the mandatory-label rule is gone for QA
    assert "[1]" in user and "[2]" in user


def test_synthesis_tasks_still_use_the_labelled_rules():
    assert "Explicitly label each claim" in prompts.CITATION_RULES
    assert prompts.QA_RULES != prompts.CITATION_RULES


def test_rules_do_not_cap_answer_length():
    # The dataset builder's STYLE_HINT limits answers to 1-5 sentences; that is QA-shaped and must NOT leak
    # into summaries, comparisons or literature-review sections.
    assert "1 to 5 sentences" not in prompts.CITATION_RULES


def test_qa_rules_tell_the_model_to_pick_the_block_that_matches_the_question():
    # Measured failure: with the right chunk in context the model still answered with a related number from another
    # block. These two rules target that; see README section 9.3.
    rules = prompts.QA_RULES
    assert "names the exact subject of the question" in rules
    assert "do not answer with a related number from a different block" in rules
    assert "copied exactly as the evidence writes it" in rules
