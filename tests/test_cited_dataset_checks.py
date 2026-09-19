"""The answer filter in training/build_cited_dataset.py decides what enters the SFT set — pin its behavior."""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for p in (str(REPO_ROOT), str(REPO_ROOT / "training")):
    if p not in sys.path:
        sys.path.insert(0, p)

from build_cited_dataset import check_answer, content_sentences, normalize_answer  # noqa: E402

EVIDENCE = {
    1: "The MSDA plug-in improves zero-shot drug response prediction on the GDSCv2 dataset by adapting "
       "multiple target domain branches, reaching an AUC of 0.83 across 120 cell lines.",
    2: "Serendipitous discovery of repurposed drugs is inconsistent, and in silico approaches depend on "
       "large accessible databases of drug and disease attributes.",
}

GOOD = (
    "SUPPORTED CLAIM: The MSDA plug-in improves zero-shot drug response prediction on the GDSCv2 dataset [1].\n"
    "SUPPORTED CLAIM: It reaches an AUC of 0.83 across 120 cell lines using multiple target domain branches [1].\n"
    "LIMITATION: In silico approaches depend on large accessible databases of drug and disease attributes [2]."
)


def test_accepts_grounded_cited_answer():
    ok, reason, metrics = check_answer(GOOD, EVIDENCE)
    assert ok, reason
    assert metrics["coverage"] == 1.0


def test_single_cited_sentence_is_a_complete_answer_only_when_allowed():
    answer = "The MSDA plug-in reaches an AUC of 0.83 across 120 cell lines on the GDSCv2 dataset. [1] SUPPORTED CLAIM"
    assert check_answer(answer, EVIDENCE, min_sentences=1)[0] is True
    assert check_answer(answer, EVIDENCE) == (False, "too_few_sentences", {})


def test_rejects_meta_commentary_padding():
    answer = (
        "According to the table, MSDA outperformed the baselines on all four reported metrics [1].\n"
        "This is stated directly in the evidence provided by the authors of the paper.\n"
        "Therefore, this can be classified as a supported claim based on that evidence."
    )
    ok, reason, _ = check_answer(answer, EVIDENCE)
    assert (ok, reason) == (False, "meta_commentary")


def test_rejects_cited_meta_commentary():
    answer = (
        "The MSDA plug-in improves zero-shot drug response prediction on the GDSCv2 dataset [1].\n"
        "This is supported by [1] which states that the plug-in adapts multiple target domain branches."
    )
    assert check_answer(answer, EVIDENCE)[1] == "meta_commentary"


def test_normalize_folds_citation_only_line_into_previous_sentence():
    raw = ("The MSDA plug-in improves zero-shot drug response prediction on the GDSCv2 dataset.\n\n"
           "SUPPORTED CLAIM: [1]")
    fixed = normalize_answer(raw)
    assert fixed == "The MSDA plug-in improves zero-shot drug response prediction on the GDSCv2 dataset [1]."
    assert check_answer(fixed, EVIDENCE, min_sentences=1)[0] is True


def test_normalize_leaves_well_formed_answers_untouched():
    assert normalize_answer(GOOD) == GOOD


def test_rejects_right_number_in_the_wrong_context():
    # 0.83 exists in evidence [1], but attached to a setting the evidence never mentions there.
    answer = "Transductive benchmark ranking reached 0.83 accuracy for pediatric patients [1]."
    ok, reason, _ = check_answer(answer, EVIDENCE, min_sentences=1)
    assert (ok, reason) == (False, "number_context_mismatch")


def test_rejects_answer_that_only_restates_the_question():
    q = "How does the MSDA plug-in improve zero-shot drug response prediction?"
    restated = "SUPPORTED CLAIM: The MSDA plug-in improves zero-shot drug response prediction [1]."
    informative = "SUPPORTED CLAIM: It adapts multiple target domain branches on the GDSCv2 dataset [1]."
    assert check_answer(restated, EVIDENCE, min_sentences=1, question=q)[1] == "restates_question"
    assert check_answer(informative, EVIDENCE, min_sentences=1, question=q)[0] is True


def test_revalidate_applies_current_checks_to_a_stored_example():
    from build_cited_dataset import revalidate

    ctx = "\n\n".join(f"[{i}] {t}" for i, t in EVIDENCE.items())
    good = {"task_type": "qa", "instruction": "What are the limits of in silico approaches?",
            "context": ctx, "response": "LIMITATION: In silico approaches depend on large accessible databases of drug and disease attributes [2]."}
    assert revalidate(good) == (True, "ok")
    assert revalidate({**good, "response": good["response"].replace("[2]", "[9]")}) == (False, "invalid_citation_id")


def test_uncited_abstention_beside_a_cited_fact_is_accepted():
    answer = (
        "SUPPORTED CLAIM: The MSDA plug-in reaches an AUC of 0.83 across 120 cell lines [1].\n"
        "LIMITATION: The evidence does not specify how the plug-in handles missing cell line data."
    )
    ok, reason, metrics = check_answer(answer, EVIDENCE, min_sentences=1)
    assert ok, reason
    assert metrics["coverage"] == 1.0


def test_pure_abstention_is_rejected_because_it_cannot_be_verified():
    answer = "The evidence does not specify how the plug-in handles missing cell line data."
    assert check_answer(answer, EVIDENCE, min_sentences=1)[1] == "no_citations"


def test_abstention_never_excuses_an_uncited_factual_claim():
    answer = (
        "SUPPORTED CLAIM: The MSDA plug-in reaches an AUC of 0.83 across 120 cell lines [1].\n"
        "Researchers have long argued that machine learning transforms modern clinical practice everywhere.\n"
        "Nobody has yet established whether regulators will accept such predictions in real trials.\n"
        "LIMITATION: The evidence does not specify how the plug-in handles missing cell line data."
    )
    assert check_answer(answer, EVIDENCE, min_sentences=1)[1] == "low_citation_coverage"


def test_rejects_a_relevant_citation_padded_with_an_unrelated_one():
    # [1] supports the claim; [2] (in silico databases) has nothing to do with AUC / cell lines.
    padded = "SUPPORTED CLAIM: The MSDA plug-in reaches an AUC of 0.83 across 120 cell lines [1][2]."
    assert check_answer(padded, EVIDENCE, min_sentences=1)[1] == "irrelevant_citation"
    assert check_answer(padded.replace("[1][2]", "[1]"), EVIDENCE, min_sentences=1)[0] is True


def test_partial_coverage_by_two_blocks_is_still_accepted():
    both = ("SUPPORTED CLAIM: The MSDA plug-in adapts multiple target domain branches, while in silico "
            "approaches depend on large accessible databases of drug attributes [1][2].")
    assert check_answer(both, EVIDENCE, min_sentences=1)[0] is True


def test_rejects_answer_without_citations():
    ok, reason, _ = check_answer(GOOD.replace("[1]", "").replace("[2]", ""), EVIDENCE)
    assert (ok, reason) == (False, "no_citations")


def test_rejects_invented_citation_id():
    ok, reason, _ = check_answer(GOOD.replace("[2]", "[7]"), EVIDENCE)
    assert (ok, reason) == (False, "invalid_citation_id")


def test_rejects_number_not_in_cited_evidence():
    bad = GOOD.replace("0.83", "0.97")
    ok, reason, _ = check_answer(bad, EVIDENCE)
    assert not ok and reason.startswith("number_not_in_evidence")


def test_rejects_mostly_uncited_answer():
    answer = (
        "The MSDA plug-in improves zero-shot drug response prediction on the GDSCv2 dataset [1].\n"
        "Researchers have long argued that machine learning transforms modern clinical practice everywhere.\n"
        "Nobody has yet established whether regulators will accept such predictions in real trials.\n"
        "Further validation across many independent laboratories would certainly be extremely valuable."
    )
    ok, reason, _ = check_answer(answer, EVIDENCE)
    assert (ok, reason) == (False, "low_citation_coverage")


def test_rejects_cited_but_ungrounded_sentences():
    answer = (
        "Quantum annealing hardware accelerates protein folding simulations dramatically in practice [1].\n"
        "Blockchain ledgers guarantee reproducible clinical trial randomization for every participant [2]."
    )
    ok, reason, _ = check_answer(answer, EVIDENCE)
    # Unrelated blocks are caught either by the per-citation check or by pooled grounding.
    assert ok is False and reason in {"irrelevant_citation", "weak_grounding"}


def test_rejects_previous_unsupported_annotation():
    ok, reason, _ = check_answer(GOOD + " *(unsupported — could not be verified against retrieved evidence)*", EVIDENCE)
    assert (ok, reason) == (False, "contains_unsupported_note")


def test_content_sentences_drops_headings_and_short_fragments():
    text = "## Summary\nObjective:\n- The proposed plug-in adapts multiple target domain branches [1].\n[3]"
    assert content_sentences(text) == ["The proposed plug-in adapts multiple target domain branches [1]."]
