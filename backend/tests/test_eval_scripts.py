"""The evaluation code itself: retrieval scoring, the eval-set filters and the RAGAS runner's bookkeeping.

These check the arithmetic and the guards with tiny hand-made cases, so a number in the evaluation report can be
trusted to mean what its label says.
"""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def er():
    return _load("eval_retrieval")


@pytest.fixture(scope="module")
def bes():
    return _load("build_eval_set")


def chunk(cid, doc):
    return SimpleNamespace(chunk_id=cid, document_id=doc)


ITEM = {"chunk_id": "c3", "document_id": "docA"}


# ------------------------------------------------------------------ retrieval scoring
def test_chunk_level_hits_mark_only_the_labelled_chunk(er):
    ranked = [chunk("c1", "docA"), chunk("c2", "docB"), chunk("c3", "docA")]
    assert er.ranked_hits(ranked, ITEM, "chunk") == [False, False, True]


def test_paper_level_counts_a_paper_once_at_its_best_rank(er):
    ranked = [chunk("c1", "docB"), chunk("c2", "docB"), chunk("c3", "docA"), chunk("c4", "docA")]
    hits = er.ranked_hits(ranked, ITEM, "paper")
    assert hits == [False, True]                          # docB once, then docA once: no repeats


def test_per_query_metrics_for_a_hit_at_rank_three(er):
    row = er.per_query([False, False, True, False, False])
    assert (row["recall@1"], row["recall@3"], row["recall@5"]) == (0.0, 1.0, 1.0)
    assert row["mrr"] == pytest.approx(1 / 3)
    assert row["ndcg@10"] == pytest.approx(1 / math.log2(4))


def test_a_miss_scores_zero_everywhere(er):
    row = er.per_query([False] * 10)
    assert all(v == 0.0 for v in row.values())


def test_summary_reports_mean_and_an_interval_that_contains_it(er):
    rows = [{"recall@1": float(i % 2), "mrr": 0.5} for i in range(40)]
    s = er.summarise(rows)
    assert s["n"] == 40 and s["recall@1"]["mean"] == 0.5
    lo, hi = s["recall@1"]["ci95"]
    assert lo <= 0.5 <= hi and lo < hi
    assert s["mrr"]["ci95"] == [0.5, 0.5]                # a constant metric has a zero-width interval


def test_summary_of_nothing_is_explicit(er):
    assert er.summarise([]) == {"n": 0}


# ------------------------------------------------------------------ eval-set filters
PASSAGE = {
    "chunk_type": "text",
    "text": "The MSDA plug-in improves zero-shot drug response prediction on the GDSCv2 dataset by adapting "
            "multiple target domain branches, reaching an AUC of 0.83 across 120 cell lines.",
}


def good(**over):
    return {"question": "Which module raises zero-shot response accuracy on GDSCv2 through several target branches?",
            "answer": "The MSDA plug-in improves zero-shot drug response prediction on GDSCv2, reaching an AUC of 0.83.", **over}


def test_a_self_contained_grounded_question_passes(bes):
    assert bes.check(good(), PASSAGE) is None


@pytest.mark.parametrize("q", [
    "What does this paper say about the MSDA plug-in and zero-shot drug response?",
    "According to the text, which plug-in improves zero-shot drug response prediction?",
])
def test_questions_that_refer_to_the_paper_are_rejected(bes, q):
    assert "refers to the paper" in bes.check(good(question=q), PASSAGE)


def test_a_question_that_copies_a_run_of_words_is_rejected(bes):
    q = "Does the MSDA plug-in improves zero-shot drug response prediction on the GDSCv2 dataset always?"
    assert bes.check(good(question=q), PASSAGE) == "copies the passage"


def test_an_answer_with_a_number_not_in_the_passage_is_rejected(bes):
    why = bes.check(good(answer="The MSDA plug-in improves zero-shot drug response prediction, reaching an AUC of 0.97."), PASSAGE)
    assert why and why.startswith("answer not grounded")


def test_malformed_items_are_rejected(bes):
    assert bes.check({"question": "", "answer": "x"}, PASSAGE) == "empty"
    assert bes.check(good(question="No question mark here at all in this sentence"), PASSAGE) == "bad question shape"
    assert bes.check({"question": 5, "answer": None}, PASSAGE) == "empty"


def test_only_prose_like_text_chunks_and_real_tables_are_eligible(bes):
    prose = {"chunk_type": "text", "section": "Results", "text": "word " * 150}
    assert bes.eligible(prose)
    assert not bes.eligible({**prose, "section": "References"})
    assert not bes.eligible({**prose, "text": "word " * 20})                       # too short to ask about
    assert not bes.eligible({**prose, "text": "1 2 3 4 5 6 " * 40})                # numeric residue of a table
    table = {"chunk_type": "table", "section": "Results", "token_count": 200, "text": "| a | b |\n" * 8}
    assert bes.eligible(table) and not bes.eligible({**table, "text": "| a |\n"})


def test_sampling_is_deterministic_and_spreads_over_papers(bes):
    docs = {f"d{i}": [{"chunk_id": f"d{i}_{j}", "document_id": f"d{i}", "chunk_type": "text", "section": "Results",
                       "text": "word " * 150, "token_count": 150} for j in range(6)] for i in range(10)}
    a, b = bes.sample(docs, 20, seed=1), bes.sample(docs, 20, seed=1)
    assert [c["chunk_id"] for c in a] == [c["chunk_id"] for c in b]
    first_round = {c["document_id"] for c in a[:10]}
    assert len(first_round) >= 5                                                    # not one paper repeated


# ------------------------------------------------------------------ ragas runner bookkeeping
@pytest.fixture()
def rg(tmp_path, monkeypatch):
    # The runner is imported without ragas (it is only needed inside the judge phase).
    mod = _load("eval_ragas")
    monkeypatch.setattr(mod, "EVAL_DIR", tmp_path)
    monkeypatch.setattr(mod, "EVAL_SET", tmp_path / "eval_set.jsonl")
    return mod


def _write_set(rg, n_text=6, n_table=3, n_fig=1):
    rows = []
    for kind, n in (("text", n_text), ("table", n_table), ("figure", n_fig)):
        rows += [{"qid": f"{kind}{i}", "type": kind, "question": "q?", "reference_answer": "a", "chunk_id": f"c{kind}{i}",
                  "document_id": "d", "question_chunk_overlap": 0.3} for i in range(n)]
    rg.EVAL_SET.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def test_the_sample_keeps_the_sets_type_proportions_and_is_repeatable(rg):
    _write_set(rg)
    a, b = rg.sample_questions(10), rg.sample_questions(10)
    assert [q["qid"] for q in a] == [q["qid"] for q in b]
    kinds = [q["type"] for q in a]
    assert kinds.count("text") == 6 and kinds.count("table") == 3 and kinds.count("figure") == 1


def test_summary_averages_only_scored_samples_and_counts_failures(rg):
    _write_set(rg, 2, 0, 0)
    run = {"qid": "text0", "type": "text", "source_chunk_retrieved": True, "source_paper_retrieved": True, "claims": [],
           "citation_metrics": {"citation_coverage": 1.0, "total_claims": 4, "flagged_claims": 1},
           "latency": {k: 100.0 for k in ("retrieval_latency_ms", "generation_latency_ms", "verification_latency_ms", "total_latency_ms")}}
    (rg.EVAL_DIR / "runs_balanced.jsonl").write_text(json.dumps(run) + "\n" + json.dumps({**run, "qid": "text1", "source_chunk_retrieved": False}) + "\n")
    scores = [{"qid": "text0", "type": "text", "faithfulness": 0.8, "answer_relevancy": 0.9, "context_precision": None,
               "context_recall": 1.0, "factual_correctness": None},
              {"qid": "text1", "type": "text", "faithfulness": None, "answer_relevancy": 0.7, "context_precision": None,
               "context_recall": 0.0, "factual_correctness": None}]
    (rg.EVAL_DIR / "ragas_scores_balanced.jsonl").write_text("\n".join(json.dumps(s) for s in scores) + "\n")
    rg.summary()
    out = json.loads((rg.EVAL_DIR / "ragas_summary.json").read_text())["modes"]["balanced"]
    assert out["ragas"]["faithfulness"] == {"mean": 0.8, "n_scored": 1, "n_failed": 1}     # the failed one is not counted as 0
    assert out["ragas"]["answer_relevancy"]["mean"] == 0.8
    assert out["ragas"]["context_precision"]["mean"] is None                               # nothing scored: unmeasured, not 0
    assert out["retrieval"]["source_chunk_in_context"] == 0.5
    assert out["citations"]["flagged_share"] == 0.25


def test_generic_questions_that_name_nothing_are_rejected(bes):
    generic = "How do researchers typically evaluate new techniques in this area of study?"
    assert bes.check(good(question=generic.replace("this area", "that area")), PASSAGE) is not None
    assert bes.check(good(question="How do people usually evaluate new techniques for making predictions?"), PASSAGE) \
        == "generic question (names nothing specific)"
    for named in ("Which module lifts zero-shot accuracy on GDSCv2?", "What does ProtTrans learn from sequences?",
                  "How many cell lines does the benchmark cover in the 2023 release?"):
        assert bes._SPECIFIC.search(named)


def test_a_bare_number_is_a_valid_table_answer_only_if_it_is_in_the_table(bes):
    table = {"chunk_type": "table", "text": "Table 3: AUC\n| Method | AUC |\n| --- | --- |\n| GraphDTA | 0.912 |"}
    q = "What AUC does GraphDTA reach in the benchmark comparison?"
    assert bes.check({"question": q, "answer": "0.912"}, table) is None
    assert bes.check({"question": q, "answer": "0.987"}, table) == "answer numbers not in the table"


def test_each_control_breaks_exactly_one_input(rg):
    items = [{"qid": f"q{i}", "question": f"question {i}?", "reference_answer": f"answer {i}", "chunk_id": f"c{i}", "document_id": f"d{i}"}
             for i in range(10)]
    chunks = {f"c{i}": {"chunk_id": f"c{i}", "document_id": f"d{i}", "chunk_type": "text", "text": f"context {i} " * 60} for i in range(10)}
    import random

    ctx = rg._control_rows("context", items, chunks, random.Random(1))
    assert all(p["contexts"] == [chunks[it["chunk_id"]]["text"]] for p, it in zip(ctx["positive"], items))
    assert all(n["contexts"] != p["contexts"] and n["question"] == p["question"] for n, p in zip(ctx["negative"], ctx["positive"]))

    q = rg._control_rows("question", items, chunks, random.Random(1))
    assert all(n["question"] != p["question"] and n["contexts"] == p["contexts"] for n, p in zip(q["negative"], q["positive"]))

    ref = rg._control_rows("reference", items, chunks, random.Random(1))
    assert all(n["reference_answer"] != p["reference_answer"] and n["answer"] == p["answer"] for n, p in zip(ref["negative"], ref["positive"]))


def test_a_metric_separates_only_when_right_and_wrong_cases_are_far_apart(rg):
    assert rg.MIN_GAP == 0.4
    assert set(rg.CONTROL_KIND) == set(rg.METRICS)            # every reported metric has a validity control


# ------------------------------------------------------------------ prompt A/B
@pytest.fixture()
def ab(tmp_path, monkeypatch):
    mod = _load("eval_prompt_ab")
    monkeypatch.setattr(mod.er, "EVAL_DIR", tmp_path)
    monkeypatch.setattr(mod.er, "EVAL_SET", tmp_path / "eval_set.jsonl")
    monkeypatch.setattr(mod, "SAMPLE", tmp_path / "ab_sample.json")
    return mod


def test_the_fresh_sample_excludes_questions_already_judged_and_is_frozen(ab):
    rows = [{"qid": f"q{i}", "type": "text", "question": "q?", "reference_answer": "a", "chunk_id": f"c{i}", "document_id": "d",
             "question_chunk_overlap": 0.3} for i in range(20)]
    ab.er.EVAL_SET.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    judged = [{"qid": f"q{i}"} for i in range(8)]
    (ab.er.EVAL_DIR / "runs_balanced.jsonl").write_text("\n".join(json.dumps(r) for r in judged) + "\n", encoding="utf-8")

    first = ab.fresh_sample(6)
    assert len(first) == 6 and not ({q["qid"] for q in first} & {f"q{i}" for i in range(8)})
    assert {q["qid"] for q in ab.fresh_sample(6)} == {q["qid"] for q in first}      # frozen: both arms see the same questions


def test_paired_difference_uses_only_questions_scored_in_both_arms(ab):
    before = {"a": {"m": 0.2}, "b": {"m": 0.4}, "c": {"m": None}, "d": {"m": 0.5}}
    after = {"a": {"m": 0.6}, "b": {"m": 0.4}, "c": {"m": 0.9}, "e": {"m": 1.0}}
    r = ab.paired(before, after, "m")
    assert r["n_pairs"] == 2                        # c (missing before) and d/e (missing in one arm) are not paired
    assert r["mean_diff"] == pytest.approx(0.2)
    assert (r["improved"], r["worse"], r["same"]) == (1, 0, 1)
    assert r["ci95"][0] <= 0.2 <= r["ci95"][1]
    assert ab.paired({}, {}, "m") == {"n_pairs": 0}


def test_the_adapted_judge_instruction_keeps_numbers_and_names_strict(rg):
    text = rg.ADAPTED_NLI
    assert "paraphrase" in text and "must match the context" in text and "contradicts" in text
