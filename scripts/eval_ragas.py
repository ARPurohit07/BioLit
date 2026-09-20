"""End-to-end answer quality with RAGAS, judged by a local Ollama model, plus a check that the judge can be trusted.

Runs in the evaluation environment (.venv-eval: ragas + langchain-ollama), not the main one, and talks to a running
backend over HTTP, so it evaluates exactly what the app serves.

Phases
    generate   ask the backend each sampled question, keep answer, retrieved contexts, citation metrics, latency
    judge      score every answer with RAGAS metrics (appends per batch, resumable)
    controls   judge validity: does Faithfulness give ~1 for a reference answer with its true context and ~0 with a
               wrong context? A judge that cannot tell these apart would make every number below meaningless
    summary    aggregate everything into experiments/eval/ragas_summary.json

RAGAS metrics used
    Faithfulness                       share of the answer's statements that the retrieved contexts support
    ResponseRelevancy                  does the answer address the question (regenerated-question similarity)
    LLMContextPrecisionWithReference   are the relevant contexts ranked at the top of what was retrieved
    LLMContextRecall                   does the retrieved context contain what the reference answer needs
    FactualCorrectness                 do the answer's claims agree with the reference answer (correctness proxy)

    python scripts/eval_ragas.py generate --mode balanced --n 50
    python scripts/eval_ragas.py judge --mode balanced
    python scripts/eval_ragas.py controls
    python scripts/eval_ragas.py summary
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = REPO_ROOT / "experiments" / "eval"
EVAL_SET = EVAL_DIR / "eval_set.jsonl"
PROCESSED = REPO_ROOT / "data" / "processed"
API = "http://127.0.0.1:8000"
JUDGE_MODEL = "qwen2.5:3b"
EMBED_MODEL = "nomic-embed-text"
METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall", "factual_correctness")


def runs_path(mode: str) -> Path:
    return EVAL_DIR / f"runs_{mode}.jsonl"


def scores_path(mode: str) -> Path:
    return EVAL_DIR / f"ragas_scores_{mode}.jsonl"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()] if path.exists() else []


def sample_questions(n: int, seed: int = 21) -> list[dict]:
    """Deterministic, stratified by question type in the eval set's own proportions."""
    items = read_jsonl(EVAL_SET)
    rng = random.Random(seed)
    by_type: dict[str, list[dict]] = defaultdict(list)
    for it in items:
        by_type[it["type"]].append(it)
    picked: list[dict] = []
    for kind, pool in by_type.items():
        rng.shuffle(pool)
        picked += pool[: max(1, round(n * len(pool) / len(items)))]
    rng.shuffle(picked)
    return picked[:n]


# ---------------------------------------------------------------------------------------------------- generate
def generate(mode: str, n: int) -> None:
    out = runs_path(mode)
    done = {r["qid"] for r in read_jsonl(out)}
    todo = [q for q in sample_questions(n) if q["qid"] not in done]
    print(f"generate[{mode}]: {len(done)} done, {len(todo)} to go", flush=True)
    t0 = time.time()
    for i, q in enumerate(todo, 1):
        try:
            r = requests.post(f"{API}/api/query", json={"question": q["question"], "mode": mode}, timeout=900)
            r.raise_for_status()
            d = r.json()
        except Exception as exc:
            print(f"  [{q['qid']}] failed: {exc}", flush=True)
            continue
        row = {
            **{k: q[k] for k in ("qid", "type", "question", "reference_answer", "chunk_id", "document_id", "question_chunk_overlap")},
            "mode": mode,
            "answer": d["answer_markdown"],
            "contexts": [e["text"] for e in d["evidence"]],
            "context_chunk_ids": [e["chunk_id"] for e in d["evidence"]],
            "source_chunk_retrieved": q["chunk_id"] in {e["chunk_id"] for e in d["evidence"]},
            "source_paper_retrieved": q["document_id"] in {e["document_id"] for e in d["evidence"]},
            "citation_metrics": d.get("citation_metrics"),
            "claims": [{"status": c["status"], "grounding": c.get("grounding"), "cited": bool(c["citation_ids"])} for c in d["claims"]],
            "latency": d["latency"], "tokens_per_second": d.get("tokens_per_second"),
        }
        with open(out, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"  [{i}/{len(todo)}] {q['qid']} {q['type']:<6} {d['latency']['total_latency_ms'] / 1000:.1f}s | "
              f"source chunk retrieved: {row['source_chunk_retrieved']} | {(time.time() - t0) / 60:.1f} min", flush=True)


# ---------------------------------------------------------------------------------------------------- judge
def make_judge():
    from langchain_ollama import ChatOllama, OllamaEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (Faithfulness, FactualCorrectness, LLMContextPrecisionWithReference,
                               LLMContextRecall, ResponseRelevancy)
    llm = LangchainLLMWrapper(ChatOllama(model=JUDGE_MODEL, temperature=0, num_ctx=6144, num_predict=700, keep_alive="30m"))
    emb = LangchainEmbeddingsWrapper(OllamaEmbeddings(model=EMBED_MODEL))
    return llm, emb, {
        "faithfulness": Faithfulness(llm=llm),
        "answer_relevancy": ResponseRelevancy(llm=llm, embeddings=emb),
        "context_precision": LLMContextPrecisionWithReference(llm=llm),
        "context_recall": LLMContextRecall(llm=llm),
        "factual_correctness": FactualCorrectness(llm=llm),
    }


def _score(rows: list[dict], metrics: dict, names: tuple[str, ...]) -> list[dict]:
    from ragas import EvaluationDataset, evaluate
    from ragas.run_config import RunConfig
    ds = EvaluationDataset.from_list([
        {"user_input": r["question"], "response": r["answer"], "retrieved_contexts": r["contexts"], "reference": r["reference_answer"]}
        for r in rows
    ])
    res = evaluate(ds, metrics=[metrics[m] for m in names], show_progress=False, raise_exceptions=False,
                   run_config=RunConfig(timeout=600, max_retries=2, max_workers=1))
    df = res.to_pandas()
    out = []
    for i in range(len(rows)):
        entry = {}
        for name in names:
            # RAGAS may suffix the column with the metric's settings, e.g. "factual_correctness(mode=f1)".
            col = next((c for c in df.columns if c == metrics[name].name or c.startswith(metrics[name].name + "(")), None)
            v = df.iloc[i][col] if col is not None else float("nan")
            entry[name] = None if v is None or (isinstance(v, float) and math.isnan(v)) else float(v)
        out.append(entry)
    return out


def judge(mode: str, batch: int = 4) -> None:
    runs, path = read_jsonl(runs_path(mode)), scores_path(mode)
    done = {r["qid"] for r in read_jsonl(path)}
    todo = [r for r in runs if r["qid"] not in done]
    print(f"judge[{mode}]: {len(done)} scored, {len(todo)} to go (judge {JUDGE_MODEL})", flush=True)
    _, _, metrics = make_judge()
    t0 = time.time()
    for s in range(0, len(todo), batch):
        chunk = todo[s: s + batch]
        scored = _score(chunk, metrics, METRICS)
        with open(path, "a", encoding="utf-8") as f:
            for r, sc in zip(chunk, scored):
                f.write(json.dumps({"qid": r["qid"], "type": r["type"], "mode": mode, **sc}) + "\n")
        print(f"  [{min(s + batch, len(todo))}/{len(todo)}] {(time.time() - t0) / 60:.1f} min | last: "
              f"{ {k: (round(v, 2) if v is not None else None) for k, v in scored[-1].items()} }", flush=True)


# ---------------------------------------------------------------------------------------------------- controls
CONTROL_KIND = {          # how each metric's negative control is built
    "faithfulness": "context", "context_recall": "context", "context_precision": "context",
    "answer_relevancy": "question", "factual_correctness": "reference",
}
MIN_GAP = 0.4             # a metric "separates" true from wrong inputs if the control means differ by at least this


def _control_rows(kind: str, items: list[dict], chunks: dict, rng: random.Random) -> dict[str, list[dict]]:
    """Positive: the reference answer with everything correct. Negative: exactly one input made wrong.
    context   -> the context is an unrelated paper's passage (Faithfulness, Context Recall, Context Precision)
    question  -> the answer is paired with a different question (Answer Relevancy)
    reference -> the answer is compared with a different question's reference answer (Factual Correctness)"""
    pos, neg = [], []
    for i, it in enumerate(items):
        base = {"question": it["question"], "answer": it["reference_answer"], "reference_answer": it["reference_answer"],
                "contexts": [chunks[it["chunk_id"]]["text"]]}
        other = items[(i + 7) % len(items)]
        pos.append(base)
        if kind == "context":
            wrong = rng.choice([c for c in chunks.values() if c["document_id"] != it["document_id"]
                                and c.get("chunk_type", "text") == "text" and len(c["text"]) > 300])
            neg.append({**base, "contexts": [wrong["text"]]})
        elif kind == "question":
            neg.append({**base, "question": other["question"]})
        else:
            neg.append({**base, "reference_answer": other["reference_answer"]})
    return {"positive": pos, "negative": neg}


def controls(n: int, only: list[str] | None = None) -> None:
    """Judge validity. For each metric, score a case where everything is right and a case where one input is wrong;
    a metric the judge cannot tell apart is not measuring what its name says. Results merge into judge_controls.json."""
    chunks = {}
    for f in PROCESSED.glob("*.json"):
        rec = json.loads(f.read_text(encoding="utf-8"))
        for c in rec["chunks"]:
            chunks[c["chunk_id"]] = c
    items = [i for i in read_jsonl(EVAL_SET) if i["type"] == "text"]
    random.Random(5).shuffle(items)
    items = items[:n]
    path = EVAL_DIR / "judge_controls.json"
    saved = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"summary": {}, "raw": {}}
    _, _, metrics = make_judge()
    for name in (only or list(CONTROL_KIND)):
        rows = _control_rows(CONTROL_KIND[name], items, chunks, random.Random(5))
        raw = {}
        for label, rs in rows.items():
            scored = []
            for s in range(0, len(rs), 4):
                scored += [x[name] for x in _score(rs[s: s + 4], metrics, (name,))]
                print(f"  controls {name} {label}: {len(scored)}/{len(rs)}", flush=True)
            raw[label] = scored
        pos = [v for v in raw["positive"] if v is not None]
        neg = [v for v in raw["negative"] if v is not None]
        pairs = [(p, q) for p, q in zip(raw["positive"], raw["negative"]) if p is not None and q is not None]
        pos_mean = round(statistics.fmean(pos), 3) if pos else None
        neg_mean = round(statistics.fmean(neg), 3) if neg else None
        saved["summary"][name] = {
            "negative_control": CONTROL_KIND[name],
            "positive_mean": pos_mean, "negative_mean": neg_mean,
            "gap": round(pos_mean - neg_mean, 3) if pos_mean is not None and neg_mean is not None else None,
            "separates": bool(pos_mean is not None and neg_mean is not None and pos_mean - neg_mean >= MIN_GAP),
            "n": len(items), "n_positive_scored": len(pos), "n_negative_scored": len(neg),
            # share of (right, wrong) pairs where the judge scored the right case strictly higher
            "pairs_ranked_correctly": round(sum(p > q for p, q in pairs) / max(1, len(pairs)), 3),
        }
        saved["raw"][name] = raw
        saved["judge"] = JUDGE_MODEL
        path.write_text(json.dumps(saved, indent=2), encoding="utf-8")     # after every metric: a kill loses at most one
    print(json.dumps(saved["summary"], indent=2))


# ---------------------------------------------------------------------------------------------------- summary
def _mean(vals):
    vals = [v for v in vals if v is not None]
    return (round(statistics.fmean(vals), 3), len(vals)) if vals else (None, 0)


def summary() -> None:
    out = {"judge": JUDGE_MODEL, "modes": {}}
    for path in sorted(EVAL_DIR.glob("runs_*.jsonl")):
        mode = path.stem.removeprefix("runs_")
        runs, scores = read_jsonl(path), {s["qid"]: s for s in read_jsonl(scores_path(mode))}
        if not runs:
            continue
        m: dict = {"n_questions": len(runs), "n_judged": len(scores), "ragas": {}, "by_type": {}, "citations": {}, "retrieval": {}, "latency_ms": {}}
        for name in METRICS:
            mean, k = _mean([s.get(name) for s in scores.values()])
            m["ragas"][name] = {"mean": mean, "n_scored": k, "n_failed": len(scores) - k}
        for kind in ("text", "table", "figure"):
            sub = [s for s in scores.values() if s["type"] == kind]
            if sub:
                m["by_type"][kind] = {"n": len(sub), **{name: _mean([s.get(name) for s in sub])[0] for name in METRICS}}
        cms = [r["citation_metrics"] for r in runs if r.get("citation_metrics")]
        for k in ("citation_coverage", "citation_precision", "faithfulness", "unsupported_claim_rate"):
            m["citations"][k] = _mean([c.get(k) for c in cms])[0]
        m["citations"]["flagged_share"] = _mean([(c.get("flagged_claims") or 0) / c["total_claims"] for c in cms if c.get("total_claims")])[0]
        m["retrieval"] = {"source_chunk_in_context": _mean([float(r["source_chunk_retrieved"]) for r in runs])[0],
                          "source_paper_in_context": _mean([float(r["source_paper_retrieved"]) for r in runs])[0]}
        for k in ("retrieval_latency_ms", "generation_latency_ms", "verification_latency_ms", "total_latency_ms"):
            m["latency_ms"][k] = round(statistics.median(r["latency"][k] for r in runs), 0)
        out["modes"][mode] = m
    controls_path = EVAL_DIR / "judge_controls.json"
    if controls_path.exists():
        out["judge_validity"] = json.loads(controls_path.read_text(encoding="utf-8"))["summary"]
    out["caveats"] = [
        "The judge is qwen2.5:3b, a small model: RAGAS scores are noisy and some samples fail to parse (n_failed). "
        "judge_validity shows whether it separates a true context from a wrong one.",
        "Reference answers were written by the same 3B model from the source chunk; FactualCorrectness measures agreement "
        "with that reference, not with ground truth.",
        "Small samples: treat differences of a few points as noise.",
    ]
    (EVAL_DIR / "ragas_summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "caveats"}, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("phase", choices=["generate", "judge", "controls", "summary"])
    ap.add_argument("--mode", default="balanced", choices=["fast", "balanced", "high_faithfulness"])
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--metrics", nargs="*", help="controls: only these metrics (default: all five)")
    a = ap.parse_args()
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    if a.phase == "generate":
        generate(a.mode, a.n)
    elif a.phase == "judge":
        judge(a.mode)
    elif a.phase == "controls":
        controls(a.n if a.n != 50 else 20, a.metrics)
    else:
        summary()
