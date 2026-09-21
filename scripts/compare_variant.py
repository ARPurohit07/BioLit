"""Answer the questions of one half of the audited set with the CURRENT prompts and compare with the saved baseline answers.

Generates through the running backend, scores the new answers with the same validated judge (faithfulness; factual-correctness
precision/recall/F1 on sentence references), checks value-only references directly, and reports paired differences with a
bootstrap interval. The baseline is experiments/eval/answers_final.jsonl, so the two arms answer identical questions.

    python scripts/compare_variant.py --half dev --tag v2
"""
from __future__ import annotations

import argparse
import json
import re
import statistics as st
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_ragas as er  # noqa: E402
import diagnose_correctness as dc  # noqa: E402
from eval_final import ask  # noqa: E402
from eval_prompt_ab import paired  # noqa: E402

NUM = re.compile(r"\d+(?:\.\d+)?")


def norm(s: str) -> str:
    s = re.sub(r"(?<=\d),(?=\d{3})", "", s)
    return s.replace("\u2011", "-").replace("\u2212", "-").replace("\u00a0", " ").replace("\u202f", " ")


def value_present(ref: str, ans: str) -> bool:
    nums = NUM.findall(ref.split("\u00b1")[0])
    return bool(nums) and all(n in norm(er.clean_answer(ans)) for n in nums)


def generate(qs: list[dict], path: Path) -> None:
    done = {r["qid"] for r in er.read_jsonl(path)}
    todo = [q for q in qs if q["qid"] not in done]
    with ThreadPoolExecutor(3) as pool, open(path, "a", encoding="utf-8") as f:
        for row in pool.map(ask, todo):
            if row:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()


def faithfulness(rows: list[dict], path: Path) -> dict[str, float | None]:
    er.JUDGE_PROVIDER, er.JUDGE_MODEL, er.WORKERS = "ollama", dc.MODEL, 4
    done = {r["qid"] for r in er.read_jsonl(path)}
    todo = [r for r in rows if r["qid"] not in done]
    metrics = er.make_judge()[2]
    for i in range(0, len(todo), 16):
        chunk = todo[i:i + 16]
        out = er._score(chunk, metrics, ("faithfulness",))
        with open(path, "a", encoding="utf-8") as f:
            for r, s in zip(chunk, out):
                f.write(json.dumps({"qid": r["qid"], **s}) + "\n")
    return {r["qid"]: r["faithfulness"] for r in er.read_jsonl(path)}


def f1_by_qid(pr: dict[str, dict]) -> dict[str, float | None]:
    out = {}
    for q, v in pr.items():
        p, r = v["precision"], v["recall"]
        out[q] = None if p is None or r is None else (2 * p * r / (p + r) if p + r else 0.0)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--half", default="dev")
    ap.add_argument("--tag", default="v2")
    a = ap.parse_args()
    import requests
    label = requests.get(f"{er.API}/api/health", timeout=10).json().get("ollama_model", "")
    if "gpt-oss" not in label:
        raise SystemExit(f"backend generator is {label!r}, not gpt-oss-120b; a stale server may hold the port")
    qs = er.read_jsonl(er.EVAL_DIR / f"eval_set_{a.half}.jsonl")
    qids = {q["qid"] for q in qs}
    base = [r for r in er.read_jsonl(er.EVAL_DIR / "answers_final.jsonl") if r["qid"] in qids]
    newp = er.EVAL_DIR / f"answers_{a.tag}_{a.half}.jsonl"
    generate(qs, newp)
    new = [r for r in er.read_jsonl(newp) if r["qid"] in qids]
    print(f"{a.half}: {len(base)} baseline answers, {len(new)} new answers", flush=True)

    arms = {}
    for name, rows, suffix in (("baseline", base, ""), (a.tag, new, f"_{a.tag}")):
        sent = [r for r in rows if not dc.is_value(r["reference_answer"])]
        pr = dc.precision_recall(sent, er.EVAL_DIR / f"pr_{a.half}{suffix}.jsonl")
        fa = faithfulness(rows, er.EVAL_DIR / f"faith_{a.half}{suffix}.jsonl")
        vals = {r["qid"]: value_present(r["reference_answer"], r["answer"]) for r in rows if dc.is_value(r["reference_answer"])}
        arms[name] = {"pr": pr, "f1": f1_by_qid(pr), "faith": fa, "vals": vals, "rows": {r["qid"]: r for r in rows}}

    def mean(d):
        v = [x for x in d.values() if x is not None]
        return round(st.fmean(v), 3) if v else None

    print(f"\n{'':<10}{'faith':>7}{'prec':>7}{'recall':>8}{'F1':>7}{'values':>9}{'abstain':>9}{'words':>7}")
    for name, x in arms.items():
        ab = sum("insufficient evidence" in r["answer"].lower() for r in x["rows"].values())
        w = st.median(len(re.findall(r"\w+", r["answer"])) for r in x["rows"].values())
        print(f"{name:<10}{mean(x['faith']):>7}{mean({q: v['precision'] for q, v in x['pr'].items()}):>7}"
              f"{mean({q: v['recall'] for q, v in x['pr'].items()}):>8}{mean(x['f1']):>7}"
              f"{sum(x['vals'].values())}/{len(x['vals']):>4}{ab:>9}{w:>7}")
    b, n = arms["baseline"], arms[a.tag]
    for label, key in (("F1", "f1"), ("faithfulness", "faith")):
        d = paired({q: {"m": v} for q, v in b[key].items()}, {q: {"m": v} for q, v in n[key].items()}, "m")
        print(f"paired {label}: {d}")
    won = sum(n["vals"].get(q, False) and not v for q, v in b["vals"].items())
    lost = sum(v and not n["vals"].get(q, False) for q, v in b["vals"].items())
    print(f"value questions: gained {won}, lost {lost}")


if __name__ == "__main__":
    main()
