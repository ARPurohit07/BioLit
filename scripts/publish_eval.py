"""Assemble experiments/eval/ragas_summary.json (what the Evaluation page serves) from the committed result files.

Three columns are published:
    balanced            all 130 audited questions, answered by Balanced mode
    held-out, before    the 62 held-out questions with the answer prompt as it was before the scope-matching change
    held-out, after     the same 62 questions with the current prompt (the prompt was tuned on the other half only)

    python scripts/publish_eval.py
"""
from __future__ import annotations

import json
import shutil
import statistics
from pathlib import Path

EVAL = Path(__file__).resolve().parents[1] / "experiments" / "eval"


def _load(name: str):
    path = EVAL / name
    return json.loads(path.read_text(encoding="utf-8")) if name.endswith(".json") else [
        json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _f1(p: float, r: float) -> float:
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)


def _metric(values: list[float | None], total: int) -> dict:
    scored = [v for v in values if v is not None]
    return {"mean": round(statistics.fmean(scored), 3) if scored else None, "n_scored": len(scored), "n_failed": total - len(scored)}


def _heldout_column(faith_file: str, pr_file: str) -> dict:
    faith, pr = _load(faith_file), _load(pr_file)
    return {
        "n_questions": len(faith),
        "n_judged": len(faith),
        "ragas": {
            "faithfulness": _metric([x["faithfulness"] for x in faith], len(faith)),
            "factual_correctness": _metric([_f1(x["precision"], x["recall"]) for x in pr], len(pr)),
            "factual_correctness_precision": _metric([x["precision"] for x in pr], len(pr)),
            "factual_correctness_recall": _metric([x["recall"] for x in pr], len(pr)),
        },
        "by_type": {}, "citations": {}, "retrieval": {}, "latency_ms": {},
    }


def main() -> None:
    final = _load("final_result_cloud.json")
    controls = _load("judge_controls_cloud.json")
    summary_path = EVAL / "ragas_summary.json"
    archive = EVAL / "ragas_summary_3b_judge.json"
    if summary_path.exists() and not archive.exists():
        shutil.copy(summary_path, archive)

    sentence = final["factual_correctness_sentence_references"]
    balanced = {
        "n_questions": final["n_answers"],
        "n_judged": final["n_judged"],
        "ragas": {
            "faithfulness": {"mean": final["faithfulness"]["mean"], "n_scored": final["faithfulness"]["scored"],
                             "n_failed": final["faithfulness"]["failed"]},
            "factual_correctness": {"mean": sentence["mean"], "n_scored": sentence["scored"], "n_failed": 0},
        },
        "by_type": {t: {"n": v["n"], "faithfulness": v["faithfulness"]} for t, v in final["by_type"].items()},
        "citations": {},
        "retrieval": {"value_in_answer": final["value_questions"]["value_present"]},
        "latency_ms": {"total_latency_ms": round(final["median_latency_s"] * 1000)},
    }
    out = {
        "judge": controls["judge"],
        "modes": {
            "balanced": balanced,
            "heldout_before": _heldout_column("faith_test.jsonl", "pr_test.jsonl"),
            "heldout_after": _heldout_column("faith_test_v2.jsonl", "pr_test_v2.jsonl"),
        },
        "judge_validity": controls["summary"],
        "caveats": [
            "The answers and the judge are both gpt-oss-120b (the judge runs as an Ollama cloud model, so answers and "
            "evidence were sent off this machine for judging). One model family grading its own output tends to flatter it.",
            "Reference answers were written by a language model from the source passage and checked against it; they are "
            "narrow, so a correct answer with extra true detail scores below 1. Factual correctness is agreement with that "
            "reference, not with ground truth.",
            f"Factual correctness covers the {sentence['scored']} questions whose reference is a sentence. The "
            f"{final['value_questions']['n']} whose reference is a bare value (mostly table cells) cannot be split into claims, so "
            f"the value is checked directly instead: it appears in {final['value_questions']['value_present']:.0%} of those answers.",
            "Held-out columns: 62 questions the answer prompt was never tuned on; factual correctness there is the F1 of "
            "claim precision and recall over the 50 sentence references. A 62-question sample moves a few points by chance, "
            "and the two columns differ in more than the scope rule: the earlier answers were served through OpenRouter and "
            "the current ones through Ollama cloud, and the current prompt also carries an extractive-answer rule.",
            "Answer relevancy, context precision and context recall were measured only with the earlier 3B judge, which "
            "did not pass the judge-validity check, so they are not published here.",
        ],
    }
    summary_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {summary_path.name}; archived the 3B-judge file as {archive.name}")


if __name__ == "__main__":
    main()
