"""Merge the raw citation-evaluation result files into one labelled comparison table.

Inputs (all under experiments/finetuned/, produced by training/eval_citations.py and
training/eval_ollama_citations.py on the SAME held-out prompts and scored by the SAME answer checker):
    citation_eval_v1_v2.json        base 1.5B, v1 adapter, v2 adapter (checkpoint-21)
    citation_eval_ollama.json       the served qwen2.5:3b, prompts exactly as the pipeline sends them
    citation_eval_ollama_hint.json  the same model with the dataset builder's style hint added to the system prompt

Output: experiments/finetuned/citation_comparison.json, which GET /api/evaluation serves to the UI.

    python scripts/summarize_citation_evals.py
"""
from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "training"))

from build_cited_dataset import content_sentences  # noqa: E402

DIR = REPO_ROOT / "experiments" / "finetuned"

CAVEATS = [
    "Only 14 held-out prompts (6 validation + 8 test) from 2 papers: differences of a few examples are noise.",
    "'Passes checker' is the same automated check used to build the training data (valid [n] ids, cited sentences, "
    "numbers and wording grounded in the cited evidence, no filler). It measures citation form and grounding, "
    "not whether an answer is factually correct, and it favours models trained or prompted toward that style.",
    "Validation examples were also used to pick adapter checkpoints, so the test column is the cleaner number.",
    "Adapters were decoded greedily; the served model was sampled with the app's settings (temperature 0.1).",
    "The fine-tuned adapters are NOT deployed: the app serves qwen2.5:3b.",
]


def _sentence_coverage(answer: str) -> float:
    sentences = content_sentences(answer)
    return sum(bool(re.search(r"\[\d+\]", s)) for s in sentences) / len(sentences) if sentences else 0.0


def _config(name: str, description: str, rows: list[tuple[str, str, dict]]) -> dict:
    """rows: (source split, answer text, score dict) per held-out prompt."""
    n = len(rows)
    val = [r for r in rows if r[0] == "val"]
    test = [r for r in rows if r[0] == "test"]
    return {
        "name": name,
        "description": description,
        "n": n,
        "cites_any": sum(r[2]["cites_any"] for r in rows),
        "valid_ids": sum(r[2]["valid_ids"] for r in rows),
        "passes_checker": sum(r[2]["passes_checker"] for r in rows),
        "val_passes": sum(r[2]["passes_checker"] for r in val), "val_n": len(val),
        "test_passes": sum(r[2]["passes_checker"] for r in test), "test_n": len(test),
        "avg_sentence_coverage": round(sum(_sentence_coverage(r[1]) for r in rows) / n, 4),
        "avg_answer_words": round(sum(len(r[1].split()) for r in rows) / n, 1),
        "avg_citations": round(sum(r[2]["num_citations"] for r in rows) / n, 2),
    }


def main() -> None:
    hf = json.loads((DIR / "citation_eval_v1_v2.json").read_text(encoding="utf-8"))["results"]
    ol = json.loads((DIR / "citation_eval_ollama.json").read_text(encoding="utf-8"))["results"]
    hint = json.loads((DIR / "citation_eval_ollama_hint.json").read_text(encoding="utf-8"))["results"]

    configs = [
        _config("Base 1.5B", "Qwen2.5-1.5B-Instruct, no adapter, pipeline prompt",
                [(r["source"], r["base_answer"], r["base"]) for r in hf]),
        _config("Fine-tuned v1", "LoRA adapter trained on 66 cited examples",
                [(r["source"], r["tuned_answer"], r["tuned"]) for r in hf]),
        _config("Fine-tuned v2 (half-trained)", "LoRA adapter, 115 cited examples, checkpoint-21 of 42 steps",
                [(r["source"], r["v2_answer"], r["v2"]) for r in hf]),
        _config("qwen2.5:3b as served", "The model the app uses, prompts exactly as the pipeline sends them",
                [(r["source"], r["qwen2.5:3b"]["answer"], r["qwen2.5:3b"]) for r in ol]),
        _config("qwen2.5:3b + style hint", "Same model, citation-style hint added to the system prompt (no training)",
                [(r["source"], r["qwen2.5:3b"]["answer"], r["qwen2.5:3b"]) for r in hint]),
    ]
    out = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "n": configs[0]["n"],
        "prompts": "6 validation + 8 test held-out prompts (paper-level split, retrieval restricted to each split's papers)",
        "caveats": CAVEATS,
        "configs": configs,
    }
    path = DIR / "citation_comparison.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"{'config':<30}{'cites':>7}{'valid':>7}{'passes':>8}{'val':>6}{'test':>6}{'sent.cov':>10}{'words':>7}")
    for c in configs:
        print(f"{c['name']:<30}{c['cites_any']:>4}/{c['n']:<2}{c['valid_ids']:>4}/{c['n']:<2}{c['passes_checker']:>5}/{c['n']:<2}"
              f"{c['val_passes']:>3}/{c['val_n']}{c['test_passes']:>4}/{c['test_n']}{c['avg_sentence_coverage']:>9.0%}{c['avg_answer_words']:>7.0f}")
    print(f"\nwrote {path.relative_to(REPO_ROOT).as_posix()}")


if __name__ == "__main__":
    main()
