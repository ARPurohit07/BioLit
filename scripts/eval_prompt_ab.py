"""Before/after test of the question-answering prompt change, on questions that were not used to design it.

The change (QA_RULES in backend/app/generation/prompts.py) drops the mandatory SUPPORTED CLAIM / INTERPRETATION labels and
asks the model to stay close to the evidence's wording. It was motivated by looking at 39 already-judged answers, so it is
evaluated here on a DIFFERENT set of questions: the same fresh questions are answered by both prompts (arm "before" = the
old prompt, arm "after" = the new one) and compared pair by pair.

Measured per arm
    faithfulness          RAGAS, scored twice: with the stock judge prompt and with the adapted one (see eval_ragas.ADAPTED_NLI)
    factual_correctness   RAGAS F1 against the reference answer (unaffected by the faithfulness adaptation)
    answer length, citation coverage, share of answers with labels, "check source" rate
The difference is reported with a paired bootstrap 95% interval, whichever way it points.

    python scripts/eval_prompt_ab.py generate --arm before     # backend running the OLD prompt
    python scripts/eval_prompt_ab.py generate --arm after      # backend running the NEW prompt
    python scripts/eval_prompt_ab.py judge --arm before        # and --arm after; runs in .venv-eval
    python scripts/eval_prompt_ab.py summary
"""
from __future__ import annotations

import argparse
import json
import random
import re
import statistics as st
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_ragas as er  # noqa: E402

EVAL_DIR = er.EVAL_DIR
SAMPLE = EVAL_DIR / "ab_sample.json"
OUT = EVAL_DIR / "prompt_ab.json"
LABELS = re.compile(r"SUPPORTED CLAIM|INTERPRETATION|LIMITATION")


def runs(arm: str) -> Path:
    if arm == "original":          # the 39 answers judged earlier (old prompt), re-scored here with the adapted judge
        return er.runs_path("balanced")
    return EVAL_DIR / f"ab_runs_{arm}.jsonl"


def scores(arm: str, judge: str) -> Path:
    return EVAL_DIR / f"ab_scores_{arm}_{judge}.jsonl"


def fresh_sample(n: int, seed: int = 33) -> list[dict]:
    """Questions never answered/judged before, stratified by type; fixed on first use so both arms see the same ones."""
    if SAMPLE.exists():
        chosen = set(json.loads(SAMPLE.read_text(encoding="utf-8")))
        return [q for q in er.read_jsonl(er.EVAL_SET) if q["qid"] in chosen]
    used = {r["qid"] for r in er.read_jsonl(er.runs_path("balanced"))}
    pool = [q for q in er.read_jsonl(er.EVAL_SET) if q["qid"] not in used]
    rng = random.Random(seed)
    by_type: dict[str, list[dict]] = {}
    for q in pool:
        by_type.setdefault(q["type"], []).append(q)
    picked: list[dict] = []
    for kind, items in by_type.items():
        rng.shuffle(items)
        picked += items[: max(1, round(n * len(items) / len(pool)))]
    picked = picked[:n]
    SAMPLE.write_text(json.dumps([q["qid"] for q in picked]), encoding="utf-8")
    return picked


def generate(arm: str, n: int) -> None:
    done = {r["qid"] for r in er.read_jsonl(runs(arm))}
    todo = [q for q in fresh_sample(n) if q["qid"] not in done]
    print(f"generate[{arm}]: {len(done)} done, {len(todo)} to go", flush=True)
    t0 = time.time()
    for i, q in enumerate(todo, 1):
        try:
            r = requests.post(f"{er.API}/api/query", json={"question": q["question"], "mode": "balanced"}, timeout=900)
            r.raise_for_status()
            d = r.json()
        except Exception as exc:
            print(f"  [{q['qid']}] failed: {exc}", flush=True)
            continue
        row = {**{k: q[k] for k in ("qid", "type", "question", "reference_answer", "chunk_id", "document_id")},
               "arm": arm, "answer": d["answer_markdown"], "contexts": [e["text"] for e in d["evidence"]],
               "source_chunk_retrieved": q["chunk_id"] in {e["chunk_id"] for e in d["evidence"]},
               "citation_metrics": d.get("citation_metrics"), "latency": d["latency"]}
        with open(runs(arm), "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"  [{i}/{len(todo)}] {q['qid']} {q['type']:<6} {(time.time() - t0) / 60:.1f} min", flush=True)


def judge(arm: str, judge_kind: str, batch: int = 3) -> None:
    """default judge -> faithfulness only; adapted judge -> faithfulness + factual correctness (FC does not use the adapted prompt)."""
    names = ("faithfulness",) if (judge_kind == "default" or arm == "original") else ("faithfulness", "factual_correctness")
    rows, path = er.read_jsonl(runs(arm)), scores(arm, judge_kind)
    done = {r["qid"] for r in er.read_jsonl(path)}
    todo = [r for r in rows if r["qid"] not in done]
    print(f"judge[{arm}/{judge_kind}]: {len(done)} scored, {len(todo)} to go", flush=True)
    _, _, metrics = er.make_judge(adapted=(judge_kind == "adapted"))
    t0 = time.time()
    for s in range(0, len(todo), batch):
        chunk = todo[s: s + batch]
        out = er._score(chunk, metrics, names)
        with open(path, "a", encoding="utf-8") as f:
            for r, sc in zip(chunk, out):
                f.write(json.dumps({"qid": r["qid"], "type": r["type"], **sc}) + "\n")
        print(f"  [{min(s + batch, len(todo))}/{len(todo)}] {(time.time() - t0) / 60:.1f} min | {out[-1]}", flush=True)


def _mean(v):
    v = [x for x in v if x is not None]
    return round(st.fmean(v), 3) if v else None


def paired(a: dict, b: dict, metric: str, seed: int = 9) -> dict:
    """b - a over questions where both arms have a score, with a paired bootstrap interval."""
    diffs = [b[q][metric] - a[q][metric] for q in a if q in b and a[q].get(metric) is not None and b[q].get(metric) is not None]
    if not diffs:
        return {"n_pairs": 0}
    rng = random.Random(seed)
    boots = sorted(st.fmean(rng.choices(diffs, k=len(diffs))) for _ in range(2000))
    return {"n_pairs": len(diffs), "mean_diff": round(st.fmean(diffs), 3), "ci95": [round(boots[50], 3), round(boots[1949], 3)],
            "improved": sum(d > 0 for d in diffs), "worse": sum(d < 0 for d in diffs), "same": sum(d == 0 for d in diffs)}


def summary() -> None:
    result = {"n_questions": {}, "arms": {}, "paired_change_after_minus_before": {}}
    per: dict[str, dict[str, dict]] = {}
    for arm in ("original", "before", "after"):
        rs = er.read_jsonl(runs(arm))
        if not rs:
            continue
        per[arm] = {}
        for jk in ("default", "adapted"):
            per[arm][jk] = {s["qid"]: s for s in er.read_jsonl(scores(arm, jk))}
        cms = [r["citation_metrics"] for r in rs if r.get("citation_metrics")]
        words = [len(re.findall(r"\w+", r["answer"])) for r in rs]
        result["n_questions"][arm] = len(rs)
        result["arms"][arm] = {
            "faithfulness_stock_judge": _mean([s.get("faithfulness") for s in per[arm]["default"].values()]),
            "faithfulness_adapted_judge": _mean([s.get("faithfulness") for s in per[arm]["adapted"].values()]),
            "factual_correctness_f1": _mean([s.get("factual_correctness") for s in per[arm]["adapted"].values()]),
            "answer_words_median": st.median(words), "reference_words_median": st.median(len(re.findall(r"\w+", r["reference_answer"])) for r in rs),
            "share_with_labels": round(sum(bool(LABELS.search(r["answer"])) for r in rs) / len(rs), 3),
            "citation_coverage": _mean([c.get("citation_coverage") for c in cms]),
            "check_source_share": _mean([(c.get("flagged_claims") or 0) / c["total_claims"] for c in cms if c.get("total_claims")]),
            "source_chunk_in_context": _mean([float(r["source_chunk_retrieved"]) for r in rs]),
        }
    if "before" in per and "after" in per:
        for name, jk, metric in (("faithfulness_stock_judge", "default", "faithfulness"), ("faithfulness_adapted_judge", "adapted", "faithfulness"),
                                 ("factual_correctness_f1", "adapted", "factual_correctness")):
            result["paired_change_after_minus_before"][name] = paired(per["before"][jk], per["after"][jk], metric)
    result["note"] = ("Same fresh questions in both arms; a paired bootstrap interval that includes 0 means the change is not distinguishable from "
                      "noise at this sample size. The adapted judge is only meaningful alongside judge_controls_adapted.json.")
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("phase", choices=["generate", "judge", "summary"])
    ap.add_argument("--arm", choices=["original", "before", "after"])
    ap.add_argument("--judge", choices=["default", "adapted"], default="default")
    ap.add_argument("--n", type=int, default=30)
    a = ap.parse_args()
    if a.phase == "generate":
        generate(a.arm, a.n)
    elif a.phase == "judge":
        judge(a.arm, a.judge)
    else:
        summary()
