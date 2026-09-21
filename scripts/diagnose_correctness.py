"""Why do answers lose factual-correctness points? Precision and recall reported separately, then a cause for each low score.

RAGAS factual correctness is an F1 over claims: precision penalises claims in the answer that the reference does not make,
recall penalises reference claims the answer misses. Reporting them apart shows whether answers are wrong or merely longer
than a one-sentence reference. For low-scoring answers whose source chunk WAS retrieved, a large model then assigns a cause
from a fixed list, so the next change targets what is actually going wrong.

    python scripts/diagnose_correctness.py --half dev
"""
from __future__ import annotations

import argparse
import json
import re
import statistics as st
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_ragas as er  # noqa: E402

MODEL = "gpt-oss:120b-cloud"
LABELS = {
    "equivalent": "the answer says the same thing as the reference; the metric under-rated it",
    "extra_detail": "the answer contains the reference's content plus additional true detail from the source",
    "incomplete": "the answer is correct but omits part of what the reference states",
    "different_aspect": "the answer is true per the source but addresses a different part of the question than the reference",
    "wrong_fact": "the answer states something the source contradicts or does not support",
    "reference_defect": "the reference is wrong, unsupported or garbled and the answer is better",
    "abstained": "the answer declines to answer",
}


def is_value(ref: str) -> bool:
    return len(re.findall(r"[A-Za-z]{3,}", ref)) <= 2 and bool(re.search(r"\d", ref))


def precision_recall(rows: list[dict], path: Path) -> dict[str, dict]:
    from ragas import EvaluationDataset, evaluate
    from ragas.metrics import FactualCorrectness
    from ragas.run_config import RunConfig
    er.JUDGE_PROVIDER, er.JUDGE_MODEL, er.WORKERS = "ollama", MODEL, 4
    llm = er.make_judge()[2]["factual_correctness"].llm
    mp, mr = FactualCorrectness(llm=llm, mode="precision"), FactualCorrectness(llm=llm, mode="recall")
    done = {r["qid"]: r for r in er.read_jsonl(path)}
    todo = [r for r in rows if r["qid"] not in done]
    for i in range(0, len(todo), 16):
        chunk = todo[i:i + 16]
        ds = EvaluationDataset.from_list([{"user_input": r["question"], "response": er.clean_answer(r["answer"]),
                                           "retrieved_contexts": r["contexts"], "reference": r["reference_answer"]} for r in chunk])
        df = evaluate(ds, metrics=[mp, mr], show_progress=False, raise_exceptions=False,
                      run_config=RunConfig(timeout=600, max_retries=2, max_workers=4)).to_pandas()
        with open(path, "a", encoding="utf-8") as f:
            for j, r in enumerate(chunk):
                p, rc = df.iloc[j]["factual_correctness(mode=precision)"], df.iloc[j]["factual_correctness(mode=recall)"]
                p, rc = (None if p != p else float(p)), (None if rc != rc else float(rc))
                f.write(json.dumps({"qid": r["qid"], "precision": p, "recall": rc}) + "\n")
        print(f"  precision/recall {min(i + 16, len(todo))}/{len(todo)}", flush=True)
    return {r["qid"]: r for r in er.read_jsonl(path)}


def classify(row: dict, chunk_text: str) -> dict:
    menu = "\n".join(f"- {k}: {v}" for k, v in LABELS.items())
    prompt = (f"A question was answered from a source passage. Decide why the answer scored low against the reference.\n\n"
              f"QUESTION: {row['question']}\nREFERENCE ANSWER: {row['reference_answer']}\nSYSTEM ANSWER: {er.clean_answer(row['answer'])}\n"
              f"SOURCE PASSAGE:\n{chunk_text[:2500]}\n\nChoose exactly one cause:\n{menu}\n\n"
              'Reply as JSON: {"cause": "<one of the names above>", "reason": "<one short sentence>"}')
    try:
        r = requests.post("http://127.0.0.1:11434/api/generate", timeout=300,
                          json={"model": MODEL, "prompt": prompt, "stream": False, "format": "json", "options": {"num_predict": 2000}})
        d = json.loads(r.json()["response"])
        return {"qid": row["qid"], "cause": d.get("cause") if d.get("cause") in LABELS else "unparsed", "reason": str(d.get("reason", ""))[:200]}
    except Exception as exc:
        return {"qid": row["qid"], "cause": "error", "reason": str(exc)[:100]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--half", default="dev")
    a = ap.parse_args()
    qids = {q["qid"] for q in er.read_jsonl(er.EVAL_DIR / f"eval_set_{a.half}.jsonl")}
    answers = [r for r in er.read_jsonl(er.EVAL_DIR / "answers_final.jsonl") if r["qid"] in qids]
    sentence = [r for r in answers if not is_value(r["reference_answer"])]
    print(f"{a.half}: {len(answers)} answers, {len(sentence)} with sentence references", flush=True)

    pr = precision_recall(sentence, er.EVAL_DIR / f"pr_{a.half}.jsonl")
    rows = []
    for r in sentence:
        p, rc = pr[r["qid"]]["precision"], pr[r["qid"]]["recall"]
        f1 = (2 * p * rc / (p + rc) if p is not None and rc is not None and p + rc else 0.0) if p is not None and rc is not None else None
        rows.append({**r, "p": p, "r": rc, "f1": f1})
    mean = lambda k: round(st.fmean(x[k] for x in rows if x[k] is not None), 3)
    print(f"\nprecision {mean('p')} | recall {mean('r')} | F1 {mean('f1')}   (n={len(rows)})", flush=True)

    meta = {c["chunk_id"]: c["text"] for c in map(json.loads, open(er.REPO_ROOT / "data/index/chunk_metadata.jsonl", encoding="utf-8"))}
    low = [x for x in rows if x["f1"] is not None and x["f1"] < 0.5 and x["source_chunk_retrieved"]]
    print(f"classifying {len(low)} low-scoring answers whose source chunk was retrieved", flush=True)
    with ThreadPoolExecutor(4) as pool:
        verdicts = list(pool.map(lambda x: classify(x, meta.get(x["chunk_id"], "")), low))
    by = defaultdict(list)
    for v in verdicts:
        by[v["cause"]].append(v)
    print("\nCAUSE OF LOSS (source chunk retrieved, F1 < 0.5):")
    for cause, vs in sorted(by.items(), key=lambda kv: -len(kv[1])):
        print(f"  {cause:<17} {len(vs):>3}  e.g. {vs[0]['qid']}: {vs[0]['reason'][:110]}")
    (er.EVAL_DIR / f"diagnosis_{a.half}.json").write_text(json.dumps(
        {"n": len(rows), "precision": mean("p"), "recall": mean("r"), "f1": mean("f1"),
         "causes": dict(Counter(v["cause"] for v in verdicts)), "verdicts": verdicts}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
