"""A fast, deterministic check of how much of the reference answer an answer actually contains. No LLM, runs in a second.

RAGAS factual correctness needs a judge and about 30 minutes per 29 answers, which is too slow to iterate against. These
two proxies are computed straight from the text, so a prompt or retrieval change can be scored immediately:

    number_match    every number in the reference answer appears in the answer (only for references that contain one)
    term_recall     share of the reference's content words that appear in the answer

Neither is factual correctness: they reward wording overlap with a model-written reference and cannot see whether an
answer is right in different words. They are for ranking variants during iteration; the RAGAS judge decides the number
that gets reported.

    python scripts/score_answers_fast.py experiments/eval/ab_runs_after.jsonl [more.jsonl ...]
"""
from __future__ import annotations

import json
import re
import statistics as st
import sys
from pathlib import Path

NUMBER = re.compile(r"\d+(?:\.\d+)?")
MARKER = re.compile(r"\[\d+(?:\s*,\s*\d+)*\]")
STOP = frozenset(
    "the and for with that this from are was were has have been used using uses use its their they them which when "
    "what where while such into over both each other more most only some than then also can may not but does did "
    "based paper study results method methods model models approach across between within".split()
)


def words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z][a-z0-9\-]{2,}", text.lower()) if w not in STOP}


def score(row: dict) -> dict:
    answer = MARKER.sub("", row["answer"])
    ref = row["reference_answer"]
    ref_nums = NUMBER.findall(ref)
    ref_words = words(ref)
    return {
        "number_match": all(n in answer for n in ref_nums) if ref_nums else None,
        "term_recall": len(ref_words & words(answer)) / len(ref_words) if ref_words else None,
        "has_numbers": bool(ref_nums),
        "type": row["type"],
        "retrieved": row["source_chunk_retrieved"],
    }


def report(path: Path) -> None:
    rows = [score(json.loads(l)) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    nums = [r["number_match"] for r in rows if r["number_match"] is not None]
    rec = [r["term_recall"] for r in rows if r["term_recall"] is not None]
    print(f"\n{path.name}  (n={len(rows)})")
    print(f"  number_match  {sum(nums)}/{len(nums)} = {st.fmean(nums):.0%}" if nums else "  number_match  n/a")
    print(f"  term_recall   {st.fmean(rec):.3f}" if rec else "  term_recall   n/a")
    for key, label in (("type", "by type"), ("retrieved", "source chunk retrieved")):
        groups: dict = {}
        for r in rows:
            groups.setdefault(r[key], []).append(r)
        parts = []
        for g, rs in sorted(groups.items(), key=lambda kv: str(kv[0])):
            n = [r["number_match"] for r in rs if r["number_match"] is not None]
            t = [r["term_recall"] for r in rs if r["term_recall"] is not None]
            num = f"num {st.fmean(n):.0%} ({len(n)})" if n else "num n/a"
            parts.append(f"{g}: {num} recall {st.fmean(t):.2f}" if t else f"{g}: {num}")
        print(f"    {label:<24} " + " | ".join(parts))


if __name__ == "__main__":
    for arg in sys.argv[1:] or ["experiments/eval/ab_runs_after.jsonl"]:
        report(Path(arg))
