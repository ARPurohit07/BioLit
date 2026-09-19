"""CLI entry point: builds SFT examples from data/processed/*.json and writes
train/val/test JSONL splits.

    python training/prepare_dataset.py \\
        --processed-dir data/processed --output-dir data/ \\
        --val-ratio 0.1 --test-ratio 0.1 --seed 42

Split strategy — PAPER-LEVEL, not example-level
-------------------------------------------------
Comparison/synthesis examples reference chunks from 2+ papers at once, so
splitting individual *examples* into train/val/test would let a paper's
text leak across splits whenever it appears in an example assigned to a
different split than one of its sibling papers. To avoid this, papers are
partitioned into train/val/test FIRST (by document_id), and then
dataset_builder is invoked separately on each paper subset — so a
multi-paper example is only ever built from papers that already live in
the same split. This means with few processed papers, val/test may end up
with fewer (or zero) multi-paper examples; that's expected and reported in
the summary table, not an error.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset_builder import (  # noqa: E402
    DEFAULT_TARGET_DISTRIBUTION,
    build_examples,
    load_processed_papers,
    make_default_response_generator,
)
from config import load_yaml_config, repo_root  # noqa: E402


def split_papers(
    document_ids: list[str], val_ratio: float, test_ratio: float, seed: int
) -> tuple[list[str], list[str], list[str]]:
    """Pure function: deterministically partitions unique document_ids into
    (train_ids, val_ids, test_ids) by count, using `seed` for the shuffle.
    Safe for tiny inputs (rounds down to 0 rather than raising)."""
    ids = sorted(set(document_ids))
    rng = random.Random(seed)
    rng.shuffle(ids)

    n = len(ids)
    n_val = min(round(n * val_ratio), n)
    n_test = min(round(n * test_ratio), n - n_val)

    val_ids = ids[:n_val]
    test_ids = ids[n_val:n_val + n_test]
    train_ids = ids[n_val + n_test:]
    return train_ids, val_ids, test_ids


def _strip_internal_fields(example: dict[str, Any]) -> dict[str, Any]:
    return {
        "instruction": example["instruction"],
        "context": example["context"],
        "response": example["response"],
        "provenance": example["provenance"],
    }


def write_jsonl(path: Path, examples: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(_strip_internal_fields(ex), ensure_ascii=False) + "\n")


def _print_summary_table(splits: dict[str, list[dict[str, Any]]]) -> None:
    task_types = sorted(DEFAULT_TARGET_DISTRIBUTION.keys())
    split_names = ["train", "val", "test"]

    header = f"{'task_type':<30}" + "".join(f"{s:>10}" for s in split_names) + f"{'total':>10}"
    print("\nExample counts per task type per split")
    print("-" * len(header))
    print(header)
    print("-" * len(header))

    grand_totals = {s: 0 for s in split_names}
    for task_type in task_types:
        row_counts = {}
        for s in split_names:
            count = sum(1 for ex in splits[s] if ex["task_type"] == task_type)
            row_counts[s] = count
            grand_totals[s] += count
        total = sum(row_counts.values())
        print(f"{task_type:<30}" + "".join(f"{row_counts[s]:>10}" for s in split_names) + f"{total:>10}")

    print("-" * len(header))
    overall_total = sum(grand_totals.values())
    print(f"{'TOTAL':<30}" + "".join(f"{grand_totals[s]:>10}" for s in split_names) + f"{overall_total:>10}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build BioLit fine-tuning dataset from data/processed/*.json")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--output-dir", default="data/")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    train_path = output_dir / "training" / "train.jsonl"
    val_path = output_dir / "validation" / "val.jsonl"
    test_path = output_dir / "test" / "test.jsonl"

    papers = load_processed_papers(args.processed_dir)

    if not papers:
        write_jsonl(train_path, [])
        write_jsonl(val_path, [])
        write_jsonl(test_path, [])
        _print_summary_table({"train": [], "val": [], "test": []})
        return

    document_ids = [p["document_id"] for p in papers]
    train_ids, val_ids, test_ids = split_papers(document_ids, args.val_ratio, args.test_ratio, args.seed)
    print(f"Paper-level split: {len(train_ids)} train / {len(val_ids)} val / {len(test_ids)} test papers "
          f"(of {len(document_ids)} total)")

    papers_by_id = {p["document_id"]: p for p in papers}

    # Load models.yaml (optional) so the default Ollama generator uses the
    # configured host/base model instead of hard-coded fallbacks.
    models_config = None
    try:
        models_config = load_yaml_config(repo_root() / "configs" / "models.yaml")
    except FileNotFoundError:
        print("[prepare_dataset] configs/models.yaml not found; using built-in Ollama defaults "
              "(http://localhost:11434, qwen2.5:1.5b-instruct).")

    response_generator = make_default_response_generator(models_config)

    splits: dict[str, list[dict[str, Any]]] = {}
    for split_name, ids, seed_offset in (
        ("train", train_ids, 0),
        ("val", val_ids, 1),
        ("test", test_ids, 2),
    ):
        subset = [papers_by_id[i] for i in ids]
        examples, stats = build_examples(
            subset,
            response_generator=response_generator,
            rng=random.Random(args.seed + seed_offset),
        )
        splits[split_name] = examples
        for warning in stats["warnings"]:
            print(f"[prepare_dataset:{split_name}] WARNING: {warning}")

    write_jsonl(train_path, splits["train"])
    write_jsonl(val_path, splits["val"])
    write_jsonl(test_path, splits["test"])

    _print_summary_table(splits)
    print(f"\nWrote {train_path}, {val_path}, {test_path}")


if __name__ == "__main__":
    main()
