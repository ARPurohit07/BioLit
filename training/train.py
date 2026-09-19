"""QLoRA SFT training script for BioLit.

    python training/train.py --config configs/training.yaml
    python training/train.py --config configs/training.yaml --resume_from_checkpoint models/adapters/biolit-qwen-lora/checkpoint-100

All hyperparameters come from the YAML config (see configs/training.yaml) —
nothing here is hard-coded, so the same script scales from a 4GB laptop GPU
(4-bit QLoRA, batch=1, grad_accum=16) up to a larger GPU by editing the YAML.
"""
from __future__ import annotations

import argparse
import inspect
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import load_yaml_config, repo_root  # noqa: E402
import hardware_utils  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QLoRA SFT fine-tuning for BioLit")
    parser.add_argument(
        "--config",
        default=str(repo_root() / "configs" / "training.yaml"),
        help="Path to configs/training.yaml (default: repo_root/configs/training.yaml)",
    )
    parser.add_argument("--resume_from_checkpoint", default=None, help="Path to a checkpoint dir to resume from")
    return parser.parse_args()


def resolve_compute_dtype(config: dict[str, Any]):
    import torch

    requested_name = config.get("bnb_4bit_compute_dtype", "bfloat16")
    requested = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }.get(requested_name, torch.float16)

    if requested == torch.bfloat16 and torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
        print("[train] bf16 requested in config but not supported on this GPU — falling back to float16.")
        return torch.float16
    return requested


def validate_target_modules(model, target_modules: list[str]) -> list[str]:
    linear_class_names = {"Linear", "Linear4bit", "Linear8bitLt"}
    linear_module_names: set[str] = set()
    for name, module in model.named_modules():
        if module.__class__.__name__ in linear_class_names:
            linear_module_names.add(name.split(".")[-1])

    found = [t for t in target_modules if t in linear_module_names]
    missing = sorted(set(target_modules) - linear_module_names)
    if missing:
        print(f"[train] WARNING: lora_target_modules not found on this model: {missing}")
        print(f"[train] Actual linear-layer module names found on model: {sorted(linear_module_names)}")
    if not found:
        print(
            "[train] ERROR: none of the configured lora_target_modules exist on this model. "
            "Fix `lora_target_modules` in configs/training.yaml using the names printed above. Aborting."
        )
        sys.exit(1)
    return found


def prompt_messages(example: dict[str, Any]) -> list[dict[str, str]]:
    """The chat messages that precede the assistant's answer.

    Examples from training/build_cited_dataset.py carry the pipeline's own `system` + `user`
    prompts verbatim, so the model is trained on exactly what it sees at inference. Older
    examples (prepare_dataset.py) only have instruction/context.
    """
    if example.get("system") and example.get("user"):
        return [
            {"role": "system", "content": example["system"]},
            {"role": "user", "content": example["user"]},
        ]
    instruction = example.get("instruction", "")
    context = example.get("context", "")
    return [{"role": "user", "content": f"{instruction}\n\nContext:\n{context}" if context else instruction}]


def format_example(example: dict[str, Any], tokenizer) -> dict[str, str]:
    response = example.get("response", "")
    messages = prompt_messages(example) + [{"role": "assistant", "content": response}]
    if getattr(tokenizer, "chat_template", None):
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    else:
        # Manual ChatML fallback (Qwen-style) if the tokenizer has no chat template.
        text = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
    return {"text": text}


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)

    try:
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            EarlyStoppingCallback,
        )
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from datasets import load_dataset
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        print(
            f"[train] Missing/broken ML dependency: {exc}\n"
            "Install/repair: torch, transformers, peft, bitsandbytes, accelerate, datasets, trl."
        )
        sys.exit(1)
    except RuntimeError as exc:
        # trl/torchao/triton version mismatches surface as RuntimeError on import.
        print(f"[train] Failed to import training dependencies: {exc}")
        sys.exit(1)

    hardware_utils.print_gpu_info()
    compute_dtype = resolve_compute_dtype(config)
    model_name = config["model_name"]

    print(f"Loading tokenizer + base model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if config.get("load_in_4bit", True):
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=config.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_use_double_quant=config.get("bnb_4bit_use_double_quant", True),
            bnb_4bit_compute_dtype=compute_dtype,
        )
        model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quant_config, device_map="auto")
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=config.get("gradient_checkpointing", True)
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=compute_dtype, device_map="auto")
        if config.get("gradient_checkpointing", True):
            model.gradient_checkpointing_enable()

    target_modules = validate_target_modules(model, config.get("lora_target_modules", []))
    lora_config = LoraConfig(
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        target_modules=target_modules,
        bias=config.get("lora_bias", "none"),
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    param_stats = hardware_utils.print_model_stats(model)

    dataset_path = str(repo_root() / config["dataset_path"]) if not Path(config["dataset_path"]).is_absolute() else config["dataset_path"]
    validation_path = (
        str(repo_root() / config["validation_path"])
        if not Path(config["validation_path"]).is_absolute()
        else config["validation_path"]
    )

    data_files = {"train": dataset_path}
    if Path(validation_path).exists():
        data_files["validation"] = validation_path
    raw_datasets = load_dataset("json", data_files=data_files)

    train_dataset = raw_datasets["train"]
    if len(train_dataset) == 0:
        print("[train] ERROR: No training examples found — run training/prepare_dataset.py after ingesting papers.")
        sys.exit(1)

    eval_dataset = raw_datasets.get("validation")
    if eval_dataset is not None and len(eval_dataset) == 0:
        eval_dataset = None

    train_dataset = train_dataset.map(lambda ex: format_example(ex, tokenizer), remove_columns=train_dataset.column_names)
    if eval_dataset is not None:
        eval_dataset = eval_dataset.map(lambda ex: format_example(ex, tokenizer), remove_columns=eval_dataset.column_names)

    # Never let the trainer silently truncate: the answer sits at the END of each example, so an
    # over-length example loses exactly the part we're training on (this once erased every response).
    max_seq_length = config["max_seq_length"]

    def _fits(ex) -> bool:
        return len(tokenizer(ex["text"])["input_ids"]) <= max_seq_length

    n_before = len(train_dataset)
    train_dataset = train_dataset.filter(_fits)
    print(f"[train] {n_before - len(train_dataset)}/{n_before} train examples exceeded max_seq_length={max_seq_length} tokens and were dropped.")
    if eval_dataset is not None:
        n_before = len(eval_dataset)
        eval_dataset = eval_dataset.filter(_fits)
        print(f"[train] {n_before - len(eval_dataset)}/{n_before} val examples dropped for length.")
        if len(eval_dataset) == 0:
            eval_dataset = None
    if len(train_dataset) == 0:
        print("[train] ERROR: every training example is longer than max_seq_length — raise it or rebuild the dataset with a smaller prompt budget.")
        sys.exit(1)

    batch_size = config["per_device_train_batch_size"]
    grad_accum = config["gradient_accumulation_steps"]
    epochs = config["num_train_epochs"]
    steps_per_epoch = math.ceil(len(train_dataset) / (batch_size * grad_accum))
    estimated_total_steps = steps_per_epoch * epochs

    print("\n--- Training run summary ---")
    hardware_utils.print_gpu_info()
    print(f"Trainable params: {param_stats['trainable_params']:,} / {param_stats['total_params']:,} "
          f"({param_stats['trainable_pct']:.4f}% trainable)")
    print(f"Train examples: {len(train_dataset)} | Val examples: {len(eval_dataset) if eval_dataset is not None else 0}")
    print(f"Sequence length: {config['max_seq_length']} | Batch size: {batch_size} | Grad accumulation: {grad_accum}")
    print(f"Estimated steps: ~{estimated_total_steps} ({steps_per_epoch}/epoch x {epochs} epochs)")
    print("-----------------------------\n")

    has_eval = eval_dataset is not None
    # Save/eval strategy must always agree when load_best_model_at_end is on, regardless
    # of which extra kwargs below end up getting dropped for this trl version.
    eval_strategy = config["eval_strategy"] if has_eval else "no"
    load_best_model_at_end = config["load_best_model_at_end"] if has_eval else False
    save_strategy = eval_strategy if load_best_model_at_end else "steps"

    sft_kwargs = dict(
        output_dir=config["output_dir"],
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=config.get("per_device_eval_batch_size", batch_size),
        gradient_accumulation_steps=grad_accum,
        gradient_checkpointing=config.get("gradient_checkpointing", True),
        learning_rate=config["learning_rate"],
        lr_scheduler_type=config["lr_scheduler_type"],
        warmup_ratio=config["warmup_ratio"],
        num_train_epochs=epochs,
        weight_decay=config["weight_decay"],
        max_grad_norm=config["max_grad_norm"],
        optim=config["optim"],
        logging_steps=config["logging_steps"],
        save_strategy=save_strategy,
        save_steps=config["save_steps"],
        save_total_limit=config["save_total_limit"],
        eval_strategy=eval_strategy,
        eval_steps=config["eval_steps"] if has_eval else None,
        load_best_model_at_end=load_best_model_at_end,
        metric_for_best_model=config["metric_for_best_model"] if has_eval else None,
        seed=config["seed"],
        dataset_text_field="text",
        bf16=(compute_dtype == torch.bfloat16),
        fp16=(compute_dtype == torch.float16 and torch.cuda.is_available()),
        report_to="none",
    )
    # trl renamed SFTConfig's sequence-length cap from `max_seq_length` to `max_length`
    # at some point; support both by introspecting whichever this installed version accepts.
    accepted_params = set(inspect.signature(SFTConfig.__init__).parameters)
    for seq_len_key in ("max_length", "max_seq_length"):
        if seq_len_key in accepted_params:
            sft_kwargs[seq_len_key] = config["max_seq_length"]
            break

    # Drop anything this trl version's SFTConfig doesn't accept at all, rather than
    # constructing blind and reacting to a TypeError — that previously caused
    # eval_strategy/save_strategy to be stripped inconsistently and crash.
    unsupported = sorted(set(sft_kwargs) - accepted_params)
    if unsupported:
        print(f"[train] This trl version's SFTConfig doesn't accept {unsupported}; dropping them.")
        for key in unsupported:
            sft_kwargs.pop(key)
    sft_config = SFTConfig(**sft_kwargs)

    callbacks = []
    if has_eval and config.get("early_stopping_patience"):
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=config["early_stopping_patience"]))

    trainer_kwargs = dict(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=callbacks,
    )
    if config.get("completion_only_loss", False):
        # Loss only on the assistant's answer: ~85% of each example is evidence text the model
        # should read, not learn to reproduce.
        from trl import DataCollatorForCompletionOnlyLM

        response_template = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
        trainer_kwargs["data_collator"] = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)
        print("[train] completion_only_loss: on (loss computed on assistant tokens only)")
    try:
        trainer = SFTTrainer(processing_class=tokenizer, **trainer_kwargs)
    except TypeError:
        trainer = SFTTrainer(tokenizer=tokenizer, **trainer_kwargs)

    start_time = time.time()
    try:
        train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    except RuntimeError as exc:
        is_oom = "out of memory" in str(exc).lower() or isinstance(exc, getattr(torch.cuda, "OutOfMemoryError", ()))
        if is_oom:
            print(
                "\n[train] CUDA out of memory. This GPU cannot fit the current configuration.\n"
                "Try, in order of least to most disruptive:\n"
                "  1. Lower `max_seq_length` in configs/training.yaml (e.g. 2048 -> 1024).\n"
                "  2. Increase `gradient_accumulation_steps` further (batch size is already 1 minimum).\n"
                "  3. Ensure `load_in_4bit: true` and `gradient_checkpointing: true`.\n"
                "  4. Close other GPU-using applications (browsers, other model servers).\n"
                "  5. Switch to a smaller base model in configs/training.yaml.\n"
            )
            sys.exit(1)
        raise
    wall_clock_s = time.time() - start_time

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    eval_metrics = {}
    if has_eval:
        eval_metrics = trainer.evaluate()

    metadata = {
        "config": config,
        "config_path": str(args.config),
        "final_train_loss": train_result.metrics.get("train_loss"),
        "final_eval_loss": eval_metrics.get("eval_loss"),
        "wall_clock_seconds": wall_clock_s,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with open(output_dir / "training_run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)

    print(f"\nSaved LoRA adapter + tokenizer to {output_dir}")
    print(f"Wrote {output_dir / 'training_run_metadata.json'}")


if __name__ == "__main__":
    main()
