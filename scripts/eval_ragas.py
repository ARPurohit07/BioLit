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
import os
import random
import re
import statistics
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
JUDGE_PROVIDER = "ollama"            # "ollama" (local 3B) or "openrouter" (large model); set by --provider
OPENROUTER_JUDGE = "openai/gpt-oss-120b"
WORKERS = 1                          # concurrent judge calls: 1 for the local GPU, more for a remote API
METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall", "factual_correctness")


_MARKERS = re.compile(r"\[\d+(?:\s*[,–-]\s*\d+)*\]")
_LABELS = re.compile(r"(?:\*\*)?(?:SUPPORTED CLAIM|INTERPRETATION|LIMITATION)(?:\*\*)?\s*:?", re.IGNORECASE)
_PIPELINE_NOTES = re.compile(r"\s*\*\(unsupported — could not be verified against retrieved evidence\)\*|\*\*Unverified statements:\*\*")


def clean_answer(text: str) -> str:
    """The answer as prose, for the judge: without citation markers, claim labels, markdown emphasis or pipeline notes.

    The app's answers are cited on purpose ("... minimises Lbce [2]."). RAGAS turns a marker into a statement of its own
    ("[2] refers to a source") and the judge marks it unsupported because the paper does not contain the text "[2]", so
    every cited answer was being penalised for its own citations: a plainly supported sentence scored 0. Markers and labels
    are formatting, not claims about the papers (the app's own verifier parses them out too), so they are removed before
    judging. The stored answers are untouched, and a flagged-unverified claim stays in the text and is still scored."""
    text = _PIPELINE_NOTES.sub("", text)
    text = _LABELS.sub("", _MARKERS.sub("", text)).replace("**", "")
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\s+([.,;:])", r"\1", text)
    return text.strip()


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
# The stock RAGAS instruction asks whether a statement "can be directly inferred" from the context, which a 3B judge reads
# very strictly (an answer copied from its source only scores ~0.45). The adapted instruction accepts faithful paraphrases
# but still requires numbers and names to match. It is only trusted because it passed the right-vs-wrong controls
# (judge_controls_adapted.json): the true context must score higher AND an unrelated context must still score near zero.
ADAPTED_NLI = (
    "Your task is to judge the faithfulness of a series of statements based on a given context. For each statement "
    "return verdict 1 if the context states it, or states it in different words (a paraphrase or a summary of what "
    "the context says). Return verdict 0 if the context does not mention it or contradicts it. Numbers, names and "
    "results in the statement must match the context. Ignore citation markers such as [1] and label words such as "
    "'SUPPORTED CLAIM'."
)


def _openrouter_key() -> str:
    key = os.environ.get("OPENROUTER_KEY", "")
    env = REPO_ROOT / ".env"
    if not key and env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("OPENROUTER_KEY"):
                key = line.partition("=")[2].strip().strip("\"'")
    if not key:
        raise SystemExit("OPENROUTER_KEY is not set (environment or .env)")
    return key


def make_judge(adapted: bool = False):
    from langchain_ollama import ChatOllama, OllamaEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (Faithfulness, FactualCorrectness, LLMContextPrecisionWithReference,
                               LLMContextRecall, ResponseRelevancy)
    if JUDGE_PROVIDER == "openrouter":
        from langchain_openai import ChatOpenAI
        llm = LangchainLLMWrapper(ChatOpenAI(
            model=OPENROUTER_JUDGE, base_url="https://openrouter.ai/api/v1", api_key=_openrouter_key(), temperature=0,
            max_tokens=1200, timeout=120, max_retries=3, extra_body={"reasoning": {"effort": "low"}}))
    else:
        llm = LangchainLLMWrapper(ChatOllama(model=JUDGE_MODEL, temperature=0, num_ctx=6144, num_predict=2000 if "cloud" in JUDGE_MODEL else 700, keep_alive="30m"))
    emb = LangchainEmbeddingsWrapper(OllamaEmbeddings(model=EMBED_MODEL))
    faithfulness = Faithfulness(llm=llm)
    if adapted:
        nli = faithfulness.get_prompts()["n_l_i_statement_prompt"]
        nli.instruction = ADAPTED_NLI
        faithfulness.set_prompts(n_l_i_statement_prompt=nli)
    return llm, emb, {
        "faithfulness": faithfulness,
        "answer_relevancy": ResponseRelevancy(llm=llm, embeddings=emb),
        "context_precision": LLMContextPrecisionWithReference(llm=llm),
        "context_recall": LLMContextRecall(llm=llm),
        "factual_correctness": FactualCorrectness(llm=llm),
    }


def is_empty_answer(answer: str) -> bool:
    """A reply with no words once markers and labels are removed (e.g. just "[4]") does not answer anything."""
    return not re.search(r"[A-Za-z0-9]{2,}", clean_answer(answer))


def _score(rows: list[dict], metrics: dict, names: tuple[str, ...]) -> list[dict]:
    """Score rows with RAGAS. An empty answer has no statements, which RAGAS reports as "no score" and which would then be
    silently left out of every average, flattering a system that answers with nothing. It is scored 0 instead."""
    empty = [i for i, r in enumerate(rows) if is_empty_answer(r["answer"])]
    if empty:
        real = [r for i, r in enumerate(rows) if i not in empty]
        scored = iter(_score(real, metrics, names) if real else [])
        return [{n: 0.0 for n in names} if i in empty else next(scored) for i in range(len(rows))]
    from ragas import EvaluationDataset, evaluate
    from ragas.run_config import RunConfig
    ds = EvaluationDataset.from_list([
        {"user_input": r["question"], "response": clean_answer(r["answer"]), "retrieved_contexts": r["contexts"], "reference": r["reference_answer"]}
        for r in rows
    ])
    res = evaluate(ds, metrics=[metrics[m] for m in names], show_progress=False, raise_exceptions=False,
                   run_config=RunConfig(timeout=600, max_retries=2, max_workers=WORKERS))
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


def controls(n: int, only: list[str] | None = None, adapted: bool = False) -> None:
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
    path = EVAL_DIR / ("judge_controls" + ("_" + JUDGE_PROVIDER if JUDGE_PROVIDER != "ollama" else "_cloud" if "cloud" in JUDGE_MODEL else "") + ("_adapted" if adapted else "") + ".json")
    saved = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"summary": {}, "raw": {}}
    _, _, metrics = make_judge(adapted)
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
    ap.add_argument("--judge", choices=["default", "adapted"], default="default", help="controls: use the adapted faithfulness prompt")
    ap.add_argument("--provider", choices=["ollama", "openrouter"], default="ollama", help="which model judges")
    ap.add_argument("--eval-set", help="override the question set")
    ap.add_argument("--judge-model", help="Ollama model that judges, e.g. gpt-oss:120b-cloud")
    a = ap.parse_args()
    JUDGE_PROVIDER = a.provider
    if a.judge_model:
        JUDGE_MODEL = a.judge_model
    WORKERS = 8 if a.provider == "openrouter" else 4 if "cloud" in JUDGE_MODEL else 1
    if a.eval_set:
        EVAL_SET = Path(a.eval_set)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    if a.phase == "generate":
        generate(a.mode, a.n)
    elif a.phase == "judge":
        judge(a.mode)
    elif a.phase == "controls":
        controls(a.n if a.n != 50 else 20, a.metrics, a.judge == "adapted")
    else:
        summary()
