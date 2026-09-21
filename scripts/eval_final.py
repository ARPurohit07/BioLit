"""Generate answers for the audited questions through the running backend and judge them with a large model.

Rebuilt from the failures of earlier evaluations: only audited references (verify_references.py), citation markers stripped
before judging, empty answers scored 0, one generator, deduplicated by question, and every number reported with how many
answers it covers. The judge has NOT been validated with right-vs-wrong controls in this run; treat its scores as unvalidated.

    python scripts/eval_final.py --generate --judge
"""
from __future__ import annotations

import argparse
import json
import re
import statistics as st
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_ragas as er  # noqa: E402

SET = er.EVAL_DIR / "eval_set_audited.jsonl"
ANSWERS = er.EVAL_DIR / "answers_final.jsonl"
SCORES = er.EVAL_DIR / "scores_final.jsonl"
OUT = er.EVAL_DIR / "final_result.json"
SUBSET_SEED = 5
NUM = re.compile(r"\d+(?:\.\d+)?")


def ask(q: dict) -> dict | None:
    try:
        r = requests.post(f"{er.API}/api/query", json={"question": q["question"], "mode": "balanced"}, timeout=300)
        r.raise_for_status()
        d = r.json()
    except Exception as exc:
        print(f"  [{q['qid']}] failed: {exc}", flush=True)
        return None
    return {**{k: q[k] for k in ("qid", "type", "question", "reference_answer", "chunk_id", "document_id")},
            "answer": d["answer_markdown"], "contexts": [e["text"] for e in d["evidence"]],
            "source_chunk_retrieved": q["chunk_id"] in {e["chunk_id"] for e in d["evidence"]},
            "citation_metrics": d.get("citation_metrics"), "latency": d["latency"]}


def generate() -> None:
    done = {r["qid"] for r in er.read_jsonl(ANSWERS)}
    todo = [q for q in er.read_jsonl(SET) if q["qid"] not in done]
    print(f"generate: {len(done)} done, {len(todo)} to go", flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(8) as pool, open(ANSWERS, "a", encoding="utf-8") as f:
        for row in pool.map(ask, todo):
            if row:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
    unique = {r["qid"] for r in er.read_jsonl(ANSWERS)}
    print(f"generate: {len(unique)} unique answers in {(time.time() - t0) / 60:.1f} min", flush=True)


def subset(rows: list[dict], n: int) -> list[dict]:
    """A fixed sample stratified by question type, so the same answers are judged on every run."""
    import random
    by = defaultdict(list)
    for r in rows:
        by[r["type"]].append(r)
    picked = []
    for kind, items in sorted(by.items()):
        random.Random(SUBSET_SEED).shuffle(items)
        picked += items[: max(1, round(n * len(items) / len(rows)))]
    return picked[:n]


def judge(provider: str, adapted: bool, limit: int) -> None:
    global SCORES, OUT
    er.JUDGE_PROVIDER, er.WORKERS = provider, (3 if provider == "openrouter" else 1)
    batch = 16 if provider == "openrouter" else 4
    if provider != "openrouter":
        SCORES, OUT = SCORES.with_name("scores_final_local.jsonl"), OUT.with_name("final_result_local.json")
    done = {r["qid"] for r in er.read_jsonl(SCORES)}
    seen, rows = set(), []
    for r in er.read_jsonl(ANSWERS):
        if r["qid"] not in seen:
            seen.add(r["qid"]); rows.append(r)
    if limit:
        rows = subset(rows, limit)
    rows = [r for r in rows if r["qid"] not in done]
    print(f"judge: {len(done)} scored, {len(rows)} to go", flush=True)
    _, _, metrics = er.make_judge(adapted)
    t0 = time.time()
    for i in range(0, len(rows), batch):
        chunk = rows[i:i + batch]
        out = er._score(chunk, metrics, ("faithfulness", "factual_correctness"))
        with open(SCORES, "a", encoding="utf-8") as f:
            for r, s in zip(chunk, out):
                f.write(json.dumps({"qid": r["qid"], "type": r["type"], **s}) + "\n")
        print(f"  judged {min(i + batch, len(rows))}/{len(rows)} | {(time.time() - t0) / 60:.1f} min", flush=True)


def _mean(v):
    v = [x for x in v if x is not None]
    return (round(st.fmean(v), 3), len(v)) if v else (None, 0)


def summary() -> None:
    answers = {r["qid"]: r for r in er.read_jsonl(ANSWERS)}
    scores = {s["qid"]: s for s in er.read_jsonl(SCORES)}
    res = {"n_answers": len(answers), "n_judged": len(scores), "judge_validated": False}
    for m in ("faithfulness", "factual_correctness"):
        mean, n = _mean([s[m] for s in scores.values()])
        res[m] = {"mean": mean, "scored": n, "failed": len(scores) - n}
    by = defaultdict(list)
    for q, s in scores.items():
        by[answers[q]["type"]].append(s)
    res["by_type"] = {t: {m: _mean([s[m] for s in v])[0] for m in ("faithfulness", "factual_correctness")} | {"n": len(v)} for t, v in by.items()}
    res["by_source_retrieved"] = {str(k): {m: _mean([scores[q][m] for q in scores if answers[q]["source_chunk_retrieved"] == k])[0]
                                           for m in ("faithfulness", "factual_correctness")} | {"n": sum(answers[q]["source_chunk_retrieved"] == k for q in scores)}
                                  for k in (True, False)}
    hits = []
    for a in answers.values():
        nums = NUM.findall(a["reference_answer"])
        if nums:
            text = er.clean_answer(a["answer"])
            hits.append(all(n in text for n in nums))
    res["reference_numbers_in_answer"] = {"share": round(st.fmean(hits), 3), "of": len(hits)}
    res["empty_answers"] = sum(er.is_empty_answer(a["answer"]) for a in answers.values())
    res["no_answer_fallback"] = sum("does not state the answer" in a["answer"] for a in answers.values())
    res["median_latency_s"] = round(st.median(a["latency"]["total_latency_ms"] for a in answers.values()) / 1000, 1)
    OUT.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--generate", action="store_true")
    ap.add_argument("--judge", action="store_true")
    ap.add_argument("--provider", choices=["openrouter", "ollama"], default="openrouter")
    ap.add_argument("--adapted", action="store_true", help="use the calibrated faithfulness instruction")
    ap.add_argument("--limit", type=int, default=0, help="judge a stratified sample of this many answers")
    a = ap.parse_args()
    if a.generate:
        generate()
    if a.judge:
        judge(a.provider, a.adapted, a.limit)
    summary()
