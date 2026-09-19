"""LM-level eval-only pass for a BioLit LoRA adapter: eval loss / perplexity
on data/test/test.jsonl, plus a few qualitative sample generations printed
to stdout for manual inspection.

This is deliberately NOT the RAG-quality benchmark (citation precision,
recall, faithfulness) — that lives in the backend/experiments evaluation
code (e.g. backend/app/evaluation, experiments/rag_modes). This script only
checks "did fine-tuning make the base language model less surprised by
held-out BioLit-style examples", which is a necessary but not sufficient
signal on its own.

    python training/evaluate.py
    python training/evaluate.py --adapter_path models/adapters/biolit-qwen-lora --test_path data/test/test.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import load_yaml_config, repo_root  # noqa: E402
from train import format_example, prompt_messages, resolve_compute_dtype  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Eval-only pass + qualitative samples for a BioLit LoRA adapter")
    parser.add_argument("--adapter_path", default=str(repo_root() / "models" / "adapters" / "biolit-qwen-lora"))
    parser.add_argument("--config", default=str(repo_root() / "configs" / "training.yaml"))
    parser.add_argument("--test_path", default=str(repo_root() / "data" / "test" / "test.jsonl"))
    parser.add_argument("--output_path", default=str(repo_root() / "experiments" / "finetuned" / "eval_results.json"))
    parser.add_argument("--num_samples", type=int, default=3, help="Qualitative generations to print")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    return parser.parse_args()


def _count_jsonl_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def _write_result(output_path: Path, result: dict) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote {output_path}")


def main() -> None:
    args = parse_args()
    output_path = Path(args.output_path)
    test_path = Path(args.test_path)
    adapter_path = Path(args.adapter_path)

    num_examples = _count_jsonl_lines(test_path)
    if num_examples == 0:
        note = f"data/test/test.jsonl is empty or missing ({test_path}) — run training/prepare_dataset.py first."
        print(f"[evaluate] {note}")
        _write_result(output_path, {
            "evaluated": False,
            "eval_loss": None,
            "perplexity": None,
            "num_examples": 0,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "note": note,
        })
        return

    if not (adapter_path / "adapter_config.json").exists():
        note = f"No LoRA adapter found at {adapter_path} — run training/train.py first."
        print(f"[evaluate] {note}")
        _write_result(output_path, {
            "evaluated": False,
            "eval_loss": None,
            "perplexity": None,
            "num_examples": num_examples,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "note": note,
        })
        return

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, DataCollatorForLanguageModeling, Trainer, TrainingArguments
        from peft import PeftModel
        from datasets import load_dataset
    except ImportError as exc:
        note = f"Missing/broken ML dependency: {exc}"
        print(f"[evaluate] {note}")
        _write_result(output_path, {
            "evaluated": False,
            "eval_loss": None,
            "perplexity": None,
            "num_examples": num_examples,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "note": note,
        })
        return

    config = load_yaml_config(args.config)
    compute_dtype = resolve_compute_dtype(config)
    model_name = config["model_name"]
    max_seq_length = config["max_seq_length"]

    print(f"Loading base model {model_name} + adapter {adapter_path}")
    tokenizer = AutoTokenizer.from_pretrained(str(adapter_path))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if config.get("load_in_4bit", True):
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=config.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_use_double_quant=config.get("bnb_4bit_use_double_quant", True),
            bnb_4bit_compute_dtype=compute_dtype,
        )
        base_model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quant_config, device_map="auto")
    else:
        base_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=compute_dtype, device_map="auto")

    model = PeftModel.from_pretrained(base_model, str(adapter_path))
    model.eval()

    raw_test = load_dataset("json", data_files=str(test_path))["train"]
    raw_examples = list(raw_test)

    def _tokenize(example):
        formatted = format_example(example, tokenizer)
        tokenized = tokenizer(formatted["text"], truncation=True, max_length=max_seq_length)
        tokenized["labels"] = list(tokenized["input_ids"])
        return tokenized

    tokenized_test = raw_test.map(_tokenize, remove_columns=raw_test.column_names)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    eval_args = TrainingArguments(
        output_dir=str(repo_root() / "experiments" / "finetuned" / "_eval_tmp"),
        per_device_eval_batch_size=1,
        report_to="none",
    )
    trainer = Trainer(model=model, args=eval_args, eval_dataset=tokenized_test, data_collator=collator)
    metrics = trainer.evaluate()
    eval_loss = metrics.get("eval_loss")
    perplexity = math.exp(eval_loss) if eval_loss is not None else None

    print(f"eval_loss={eval_loss} perplexity={perplexity}")
    _write_result(output_path, {
        "evaluated": True,
        "eval_loss": eval_loss,
        "perplexity": perplexity,
        "num_examples": len(tokenized_test),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "note": None,
    })

    # Qualitative sample generations (manual inspection only, not scored).
    ollama_num_ctx = 8192
    try:
        models_config = load_yaml_config(repo_root() / "configs" / "models.yaml")
        ollama_num_ctx = models_config.get("ollama", {}).get("num_ctx", 8192)
    except FileNotFoundError:
        pass

    print(f"\n--- Qualitative samples (first {min(args.num_samples, len(raw_examples))}, num_ctx<={ollama_num_ctx}) ---")
    for example in raw_examples[: args.num_samples]:
        instruction = example.get("instruction", "")
        context = example.get("context", "")
        prompt_text = tokenizer.apply_chat_template(prompt_messages(example), tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=min(max_seq_length, ollama_num_ctx)).to(model.device)
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        generated = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        print(f"\nINSTRUCTION: {instruction}")
        print(f"CONTEXT (truncated): {context[:300]}{'...' if len(context) > 300 else ''}")
        print(f"GENERATED: {generated.strip()}")
    print("--- end qualitative samples ---")


if __name__ == "__main__":
    main()
