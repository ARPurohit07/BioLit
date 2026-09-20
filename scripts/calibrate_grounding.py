"""How well does the fast, LLM-free grounding estimate agree with the LLM claim verifier?

    python scripts/calibrate_grounding.py collect [--only qa|compare|all]   # needs the backend running; ~4 s per claim (resumable)
    python scripts/calibrate_grounding.py analyze    # offline: tune thresholds on half the questions, score the other half

collect: asks a set of questions in Balanced mode over the arXiv papers, then sends every CITED claim to /api/verify
(against the evidence blocks it cites) and stores the verifier's verdict next to the cited text.
analyze: splits by QUESTION (so no answer straddles the split), grid-searches the two label thresholds on the
calibration half, and reports agreement with the verifier on the held-out half, against simple baselines.

The reference here is the LLM verifier, itself a 3B model: this measures AGREEMENT WITH THE VERIFIER, not truth.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from backend.app.verification import grounding as G  # noqa: E402

DATA = REPO_ROOT / "data" / "results" / "grounding_calibration.jsonl"
OUT = REPO_ROOT / "experiments" / "grounding_calibration.json"
BASE = "http://127.0.0.1:8000"

EXTRA_QUESTIONS = [
    ("What limitations do these drug repurposing papers mention?", "question_answering"),
    ("How does the MSDA plug-in improve zero-shot drug response prediction?", "question_answering"),
    ("What is the role of the bilinear attention network in SiamDTI?", "question_answering"),
    ("What machine learning algorithms does NeuroCADR use?", "question_answering"),
    ("How is hyperbolic space used for drug repositioning?", "question_answering"),
    ("What evaluation metrics do these papers report?", "question_answering"),
    ("What is the substructure extraction module in MSN-DDI?", "question_answering"),
    ("How do graph neural networks represent drugs?", "question_answering"),
    ("What datasets are used for drug-drug interaction prediction?", "question_answering"),
    ("drug response prediction with domain adaptation", "summarize"),
    ("drug-target interaction prediction", "summarize"),
    ("drug repurposing with machine learning", "limitations"),
    ("graph neural networks for drug discovery", "limitations"),
    ("drug synergy and interaction prediction", "research_gaps"),
]


def _questions() -> list[tuple[str, str]]:
    qs = list(EXTRA_QUESTIONS)
    for path in ("data/validation/val.jsonl", "data/test/test.jsonl"):
        for line in (REPO_ROOT / path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                e = json.loads(line)
                if e["task_type"] == "qa":
                    qs.append((e["instruction"], "question_answering"))
    seen, out = set(), []
    for q in qs:
        if q[0] not in seen:
            seen.add(q[0])
            out.append(q)
    return out


def _jobs(docs: list[str], only: str) -> list[tuple[str, str, callable]]:
    """(label, kind, request) triples. `request()` returns the API response for the job."""
    import requests

    jobs = []
    if only in ("qa", "all"):
        for q, qtype in _questions():
            jobs.append((q, qtype, lambda q=q, qtype=qtype: requests.post(
                f"{BASE}/api/query", timeout=600,
                json={"question": q, "mode": "balanced", "query_type": qtype, "document_ids": docs}).json()))
    if only in ("compare", "all"):
        pairs = [(0, 1), (2, 3), (4, 5), (0, 3), (1, 4), (2, 5), (0, 5), (1, 2)]
        aspects = ["compare_methodology", "compare_results", "compare_papers"]
        for k, (a, b) in enumerate(pairs):
            for aspect in (aspects[k % 3], aspects[(k + 1) % 3]):
                jobs.append((f"[{aspect}] {docs[a][:8]} vs {docs[b][:8]}", aspect, lambda a=a, b=b, aspect=aspect: requests.post(
                    f"{BASE}/api/compare", timeout=900,
                    json={"document_ids": [docs[a], docs[b]], "aspect": aspect, "mode": "balanced"}).json()))
        jobs.append(("[literature-review] drug repurposing with machine learning", "literature_review", lambda: requests.post(
            f"{BASE}/api/literature-review", timeout=900,
            json={"topic": "drug repurposing with machine learning", "mode": "balanced", "document_ids": docs, "max_papers": 6}).json()))
    return jobs


def collect(only: str = "all") -> None:
    import requests

    DATA.parent.mkdir(parents=True, exist_ok=True)
    done = {(r["question"], r["claim_text"]) for r in map(json.loads, DATA.read_text(encoding="utf-8").splitlines())} if DATA.exists() else set()
    docs = [d["document_id"] for d in requests.get(f"{BASE}/api/documents/", timeout=30).json()["documents"]
            if d["filename"].startswith("arxiv_")]
    jobs = _jobs(docs, only)
    print(f"[collect] {len(jobs)} jobs ({only}) over {len(docs)} papers | {len(done)} claims already collected", flush=True)
    t0, n_new = time.time(), 0
    for i, (label, qtype, request) in enumerate(jobs, 1):
        d = request()
        chunk_of = {e["citation_id"]: e for e in d["evidence"]}
        cited_claims = [c for c in d["claims"] if any(x in chunk_of for x in c["citation_ids"])]
        for c in cited_claims:
            if (label, c["text"]) in done:
                continue
            cited = [chunk_of[x] for x in c["citation_ids"] if x in chunk_of]
            v = requests.post(f"{BASE}/api/verify", timeout=300,
                              json={"claim_text": c["text"], "evidence_chunk_ids": [e["chunk_id"] for e in cited]}).json()
            rec = {"question": label, "query_type": qtype, "claim_text": c["text"], "citation_ids": c["citation_ids"],
                   "cited_texts": [e["text"] for e in cited], "verdict": v["status"], "rationale": v["rationale"][:300]}
            with open(DATA, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_new += 1
        print(f"[collect] {i:>2}/{len(jobs)} {qtype:<19} {len(cited_claims):>2} cited claims | +{n_new} verified so far | "
              f"{(time.time() - t0) / 60:.1f} min | {label[:60]}", flush=True)
    print("[collect] DONE", flush=True)


def _load() -> list[dict]:
    return [json.loads(l) for l in DATA.read_text(encoding="utf-8").splitlines() if l.strip()]


def _predict(rec: dict, strong_min: float, weak_min: float) -> str:
    return G.label_from(G.signals(rec["claim_text"], rec["cited_texts"]), strong_min, weak_min).label


SUPPORTED_LIKE = {"SUPPORTED", "PARTIALLY_SUPPORTED"}


def _score(rows: list[dict], strong_min: float, weak_min: float) -> dict:
    """Agreement of the estimate with the verifier. Two views:
    * grounded-vs-not: predicted 'strong' against verifier SUPPORTED (the strict reference);
    * flagging: predicted 'none' against verifier UNSUPPORTED/CONTRADICTED (do we catch the bad claims?)."""
    pred = [_predict(r, strong_min, weak_min) for r in rows]
    strict = [r["verdict"] == "SUPPORTED" for r in rows]
    bad = [r["verdict"] not in SUPPORTED_LIKE for r in rows]
    tp = sum(p == G.STRONG and s for p, s in zip(pred, strict))
    fp = sum(p == G.STRONG and not s for p, s in zip(pred, strict))
    fn = sum(p != G.STRONG and s for p, s in zip(pred, strict))
    tn = sum(p != G.STRONG and not s for p, s in zip(pred, strict))
    flagged = [p == G.NONE for p in pred]
    return {
        "n": len(rows), "labels": dict(Counter(pred)),
        "accuracy_strong_vs_supported": (tp + tn) / len(rows),
        "balanced_accuracy": 0.5 * (tp / max(tp + fn, 1) + tn / max(tn + fp, 1)),
        "precision_strong": tp / max(tp + fp, 1), "recall_strong": tp / max(tp + fn, 1),
        "none_catches_unsupported": sum(f and b for f, b in zip(flagged, bad)) / max(sum(flagged), 1),
        "unsupported_recall": sum(f and b for f, b in zip(flagged, bad)) / max(sum(bad), 1),
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }


def analyze() -> None:
    rows = _load()
    questions = sorted({r["question"] for r in rows})
    cal_q = set(questions[0::2])
    cal, test = [r for r in rows if r["question"] in cal_q], [r for r in rows if r["question"] not in cal_q]
    verdicts = Counter(r["verdict"] for r in rows)
    print(f"{len(rows)} cited claims from {len(questions)} questions | calibration {len(cal)} claims, held-out {len(test)}")
    print("verifier verdicts:", dict(verdicts))
    base_rate = verdicts["SUPPORTED"] / len(rows)
    bad_rate = (verdicts["UNSUPPORTED"] + verdicts["CONTRADICTED"]) / len(rows)
    print(f"base rate: {base_rate:.0%} of cited claims are SUPPORTED, so 'always say grounded' scores {base_rate:.0%} accuracy")
    print(f"base rate: {bad_rate:.0%} are UNSUPPORTED/CONTRADICTED, so a random flag would be right {bad_rate:.0%} of the time\n")

    best = max(((s, w) for s in (0.5, 0.6, 0.7, 0.8, 0.9) for w in (0.2, 0.3, 0.4, 0.5, 0.6) if w < s),
               key=lambda sw: _score(cal, *sw)["balanced_accuracy"])
    print(f"thresholds tuned on the calibration half: strong >= {best[0]}, weak >= {best[1]} "
          f"(defaults in code: {G.STRONG_MIN}, {G.WEAK_MIN})")
    shipped = (G.STRONG_MIN, G.WEAK_MIN)
    for name, rs, th in (("calibration half (tuned)", cal, best), ("HELD-OUT half (tuned)", test, best),
                         ("SHIPPED defaults, calibration half", cal, shipped), ("SHIPPED defaults, HELD-OUT half", test, shipped)):
        sc = _score(rs, *th)
        print(f"\n{name}: n={sc['n']} labels={sc['labels']}")
        print(f"   accuracy (strong vs SUPPORTED) {sc['accuracy_strong_vs_supported']:.0%} | balanced accuracy {sc['balanced_accuracy']:.0%} "
              f"| precision(strong) {sc['precision_strong']:.0%} recall(strong) {sc['recall_strong']:.0%}")
        print(f"   'none' flags: {sc['none_catches_unsupported']:.0%} of them are truly unsupported/contradicted, "
              f"catching {sc['unsupported_recall']:.0%} of all such claims | confusion {sc['confusion']}")

    held = _score(test, *best)
    shipped_held, shipped_cal = _score(test, *shipped), _score(cal, *shipped)
    # answer-level: does the estimated share of strong claims track the verifier's share of SUPPORTED claims?
    per_q = {}
    for r in test:
        per_q.setdefault(r["question"], []).append(r)
    est = [sum(_predict(r, *shipped) == G.STRONG for r in rs) / len(rs) for rs in per_q.values()]
    ref = [sum(r["verdict"] == "SUPPORTED" for r in rs) / len(rs) for rs in per_q.values()]
    mae = sum(abs(a - b) for a, b in zip(est, ref)) / len(est)
    print(f"\nanswer level, shipped defaults (held-out, {len(est)} answers): mean |estimated share - verified share| = {mae:.2f}")

    print("\nsample disagreements on the held-out half (estimate says strong but verifier disagrees, and vice versa):")
    shown = 0
    for r in test:
        p = _predict(r, *shipped)
        if (p == G.STRONG) != (r["verdict"] == "SUPPORTED") and shown < 8:
            shown += 1
            print(f"  est={p:<6} verifier={r['verdict']:<19} | {r['claim_text'][:110]!r}\n      verifier said: {r['rationale'][:150]!r}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "n_claims": len(rows), "n_questions": len(questions), "verifier_verdicts": dict(verdicts),
        "base_rate_supported": round(base_rate, 3), "base_rate_unsupported_or_contradicted": round(bad_rate, 3),
        "shipped_thresholds": {"strong_min": shipped[0], "weak_min": shipped[1]},
        "shipped_calibration_half": shipped_cal, "shipped_held_out": shipped_held,
        "tuned_alternative": {"strong_min": best[0], "weak_min": best[1], "held_out": held},
        "answer_level_mean_abs_error": round(mae, 3),
        "note": "Agreement with the LLM verifier (a 3B model), not with ground truth. Shipped thresholds were fixed before collecting data; the tuned alternative is fit on half the questions and scored on the other half.",
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT.relative_to(REPO_ROOT).as_posix()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("phase", choices=["collect", "analyze"])
    ap.add_argument("--only", choices=["qa", "compare", "all"], default="all",
                    help="collect: which job set to run (compare = paper-pair comparisons + one literature review)")
    args = ap.parse_args()
    collect(args.only) if args.phase == "collect" else analyze()
