"""Does the fine-tune actually emit valid [n] citations? Base model vs. base + LoRA adapter, same prompts.

Runs each held-out example (pipeline-format system + user prompt from val/test JSONL) through the 4-bit
base model with the adapter DISABLED, then with it ENABLED, using greedy decoding, and scores both with the
same answer checker that built the dataset (training/build_cited_dataset.py::check_answer):
  - cites_any:        answer contains at least one [n]
  - valid_ids:        every [n] refers to a real evidence block (no invented citation numbers)
  - passes_checker:   passes ALL dataset checks (coverage, number grounding, per-citation relevance, ...)

Small-n caveat: val+test are only a few examples from held-out papers — read the counts, not the percentages.

    python training/eval_citations.py --adapter_path models/adapters/biolit-qwen-lora
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_cited_dataset import check_answer, evidence_from_context  # noqa: E402
from config import load_yaml_config, repo_root  # noqa: E402
from train import prompt_messages, resolve_compute_dtype  # noqa: E402

_CITE = re.compile(r"\[(\d+)\]")


def _relative_to_repo(path: str) -> str:
    """Store repo-relative paths in result files, not this machine's absolute ones."""
    try:
        return Path(path).resolve().relative_to(repo_root()).as_posix()
    except ValueError:
        return Path(path).name


def score(answer: str, example: dict) -> dict:
    evidence = evidence_from_context(example["context"])
    ids = [int(m) for m in _CITE.findall(answer)]
    is_qa = example["task_type"] == "qa"
    ok, reason, _ = check_answer(
        answer, evidence, min_sentences=1 if is_qa else 2, question=example["instruction"] if is_qa else None
    )
    return {
        "cites_any": bool(ids),
        "valid_ids": bool(ids) and all(i in evidence for i in ids),
        "passes_checker": ok,
        "checker_reason": reason,
        "num_citations": len(ids),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter_path", default=str(repo_root() / "models" / "adapters" / "biolit-qwen-lora"),
                    help="primary adapter, reported as 'tuned'")
    ap.add_argument("--extra_adapter", action="append", default=[], metavar="NAME=PATH",
                    help="additional adapter to score side by side (repeatable), "
                         "e.g. v2=models/adapters/biolit-qwen-lora-v2")
    ap.add_argument("--config", default=str(repo_root() / "configs" / "training.yaml"))
    ap.add_argument("--data", nargs="+", default=[str(repo_root() / "data" / "validation" / "val.jsonl"),
                                                   str(repo_root() / "data" / "test" / "test.jsonl")])
    ap.add_argument("--max_new_tokens", type=int, default=320)
    ap.add_argument("--output", default=str(repo_root() / "experiments" / "finetuned" / "citation_eval.json"))
    args = ap.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    config = load_yaml_config(args.config)
    examples = []
    for path in args.data:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                examples.append({**json.loads(line), "_source": Path(path).stem})
    print(f"[eval] {len(examples)} held-out examples from {[Path(p).name for p in args.data]}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    quant = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type=config.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_use_double_quant=config.get("bnb_4bit_use_double_quant", True),
        bnb_4bit_compute_dtype=resolve_compute_dtype(config),
    )
    base = AutoModelForCausalLM.from_pretrained(config["model_name"], quantization_config=quant, device_map="auto")
    model = PeftModel.from_pretrained(base, args.adapter_path, adapter_name="tuned")
    extras = dict(kv.split("=", 1) for kv in args.extra_adapter)
    for name, path in extras.items():
        model.load_adapter(path, adapter_name=name)
    names = ["tuned", *extras]
    model.eval()

    def generate(example: dict) -> str:
        prompt = tokenizer.apply_chat_template(prompt_messages(example), tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
        return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    results = []
    for n, ex in enumerate(examples, start=1):
        row = {"source": ex["_source"], "task_type": ex["task_type"], "question": ex["instruction"]}
        with model.disable_adapter():
            row["base_answer"] = generate(ex)
        row["base"] = score(row["base_answer"], ex)
        for name in names:
            model.set_adapter(name)
            row[f"{name}_answer"] = generate(ex)
            row[name] = score(row[f"{name}_answer"], ex)
        results.append(row)
        parts = " | ".join(
            f"{who}: cites={row[who]['num_citations']} ok={row[who]['passes_checker']!s:<5}({row[who]['checker_reason']})"
            for who in ("base", *names)
        )
        print(f"[{n:>2}/{len(examples)}] {ex['_source']:<4} {ex['task_type']:<11} {parts}", flush=True)

    summary = {}
    for who in ("base", *names):
        summary[who] = {k: sum(r[who][k] for r in results) for k in ("cites_any", "valid_ids", "passes_checker")}
    print(f"\n=== Summary over {len(results)} held-out examples (count of examples) ===")
    print(f"{'':<8}{'cites any [n]':>15}{'all ids valid':>15}{'passes checker':>16}")
    for who in ("base", *names):
        s = summary[who]
        print(f"{who:<8}{s['cites_any']:>15}{s['valid_ids']:>15}{s['passes_checker']:>16}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"timestamp": datetime.now(timezone.utc).isoformat(), "adapter": _relative_to_repo(args.adapter_path),
                               "n": len(results), "summary": summary, "results": results}, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    print(f"\n[eval] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
