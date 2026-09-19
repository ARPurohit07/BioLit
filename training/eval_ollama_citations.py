"""Score Ollama-served models on the same held-out prompts and checker as training/eval_citations.py.

This is the baseline the fine-tuned adapters have to be compared against: the model actually served by the
app (qwen2.5:3b) receives the SAME pipeline-format system + user prompts (no style hint) and is scored by the
same answer checker. Sampling matches the pipeline's OllamaClient defaults (temperature 0.1, top_p 0.9,
num_ctx 8192), so this reflects what a user of the app gets — the adapter evaluation used greedy decoding.

    python training/eval_ollama_citations.py --models qwen2.5:3b
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import repo_root  # noqa: E402
from build_cited_dataset import STYLE_HINT  # noqa: E402
from eval_citations import score  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=["qwen2.5:3b"])
    ap.add_argument("--host", default=os.environ.get("OLLAMA_HOST", "http://localhost:11434"))
    ap.add_argument("--data", nargs="+", default=[str(repo_root() / "data" / "validation" / "val.jsonl"),
                                                   str(repo_root() / "data" / "test" / "test.jsonl")])
    ap.add_argument("--output", default=str(repo_root() / "experiments" / "finetuned" / "citation_eval_ollama.json"))
    ap.add_argument("--style_hint", action="store_true",
                    help="append the dataset builder's STYLE_HINT to the system prompt (prompt-only alternative "
                         "to fine-tuning: no training, no export)")
    args = ap.parse_args()

    examples = []
    for path in args.data:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                examples.append({**json.loads(line), "_source": Path(path).stem})
    print(f"[eval] {len(examples)} held-out examples | models {args.models} | host {args.host}", flush=True)

    def generate(model: str, ex: dict) -> tuple[str, float]:
        t0 = time.time()
        r = requests.post(
            f"{args.host}/api/generate",
            json={"model": model, "prompt": ex["user"],
                  "system": ex["system"] + (STYLE_HINT if args.style_hint else ""), "stream": False,
                  "options": {"temperature": 0.1, "top_p": 0.9, "num_ctx": 8192}},
            timeout=300,
        )
        r.raise_for_status()
        return r.json().get("response", "").strip(), time.time() - t0

    results = []
    for n, ex in enumerate(examples, start=1):
        row = {"source": ex["_source"], "task_type": ex["task_type"], "question": ex["instruction"]}
        parts = []
        for model in args.models:
            answer, secs = generate(model, ex)
            row[model] = {**score(answer, ex), "answer": answer, "seconds": round(secs, 1)}
            s = row[model]
            parts.append(f"{model}: cites={s['num_citations']} ok={s['passes_checker']!s:<5}({s['checker_reason']}) {secs:.0f}s")
        results.append(row)
        print(f"[{n:>2}/{len(examples)}] {ex['_source']:<4} {ex['task_type']:<11} " + " | ".join(parts), flush=True)

    summary = {m: {k: sum(r[m][k] for r in results) for k in ("cites_any", "valid_ids", "passes_checker")}
               for m in args.models}
    print(f"\n=== Summary over {len(results)} held-out examples (count of examples) ===")
    print(f"{'':<14}{'cites any [n]':>15}{'all ids valid':>15}{'passes checker':>16}{'avg sec':>9}")
    for m in args.models:
        s = summary[m]
        avg = sum(r[m]["seconds"] for r in results) / len(results)
        print(f"{m:<14}{s['cites_any']:>15}{s['valid_ids']:>15}{s['passes_checker']:>16}{avg:>9.1f}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"timestamp": datetime.now(timezone.utc).isoformat(), "models": args.models,
                               "n": len(results), "summary": summary, "results": results}, indent=2,
                              ensure_ascii=False), encoding="utf-8")
    print(f"\n[eval] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
