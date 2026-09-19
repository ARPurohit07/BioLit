"""Pytest suite for training/dataset_builder.py + training/prepare_dataset.py's
paper-level splitter. No GPU and no real Ollama call required — a canned
fake response_generator stands in for the local Ollama call.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pytest

_TRAINING_DIR = Path(__file__).resolve().parent.parent / "training"
sys.path.insert(0, str(_TRAINING_DIR))

import dataset_builder  # noqa: E402
from prepare_dataset import split_papers  # noqa: E402


def _make_paper(document_id: str, title: str) -> dict:
    sections = ["Abstract", "Introduction", "Methods", "Results", "Discussion", "Limitations", "Conclusion"]
    chunks = [
        {
            "chunk_id": f"{document_id}_p{i}_c0",
            "document_id": document_id,
            "page_number": i + 1,
            "section": section,
            "text": f"{section} content for {title}.",
            "token_count": 12,
        }
        for i, section in enumerate(sections)
    ]
    return {
        "document_id": document_id,
        "title": title,
        "authors": ["Author A"],
        "year": 2023,
        "source": "upload",
        "filename": f"{document_id}.pdf",
        "num_pages": len(sections),
        "status": "indexed",
        "chunks": chunks,
    }


def fake_response_generator(instruction: str, context: str, task_type: str) -> str:
    return f"FAKE RESPONSE [{task_type}]"


@pytest.fixture
def three_papers() -> list[dict]:
    return [_make_paper(f"doc{i}", f"Paper {i}") for i in range(1, 4)]


@pytest.fixture
def six_papers() -> list[dict]:
    return [_make_paper(f"doc{i}", f"Paper {i}") for i in range(1, 7)]


# ---------------------------------------------------------------------------
# load_processed_papers
# ---------------------------------------------------------------------------

def test_load_processed_papers_reads_fixture_files(tmp_path, three_papers):
    for paper in three_papers:
        (tmp_path / f"{paper['document_id']}.json").write_text(json.dumps(paper), encoding="utf-8")

    loaded = dataset_builder.load_processed_papers(tmp_path)
    assert len(loaded) == 3
    assert {p["document_id"] for p in loaded} == {"doc1", "doc2", "doc3"}


def test_load_processed_papers_empty_dir_returns_empty_list(tmp_path):
    empty_dir = tmp_path / "processed"
    empty_dir.mkdir()
    assert dataset_builder.load_processed_papers(empty_dir) == []


def test_load_processed_papers_missing_dir_returns_empty_list(tmp_path):
    missing_dir = tmp_path / "does_not_exist"
    assert dataset_builder.load_processed_papers(missing_dir) == []


# ---------------------------------------------------------------------------
# build_examples: shape, provenance, no-papers handling
# ---------------------------------------------------------------------------

def test_build_examples_with_no_papers_emits_nothing_and_warns():
    examples, stats = dataset_builder.build_examples([], response_generator=fake_response_generator)
    assert examples == []
    assert any("No processed papers found" in w for w in stats["warnings"])


def test_build_examples_shape_and_provenance(three_papers):
    examples, stats = dataset_builder.build_examples(
        three_papers, response_generator=fake_response_generator, rng=random.Random(1)
    )

    assert len(examples) > 0
    for example in examples:
        assert set(example.keys()) == {"instruction", "context", "response", "provenance", "task_type"}
        assert isinstance(example["instruction"], str) and example["instruction"]
        assert isinstance(example["context"], str) and example["context"]
        assert example["response"] == f"FAKE RESPONSE [{example['task_type']}]"

        assert isinstance(example["provenance"], list) and example["provenance"]
        for prov in example["provenance"]:
            assert set(prov.keys()) == {"paper_id", "page", "section"}
            assert prov["paper_id"] in {"doc1", "doc2", "doc3"}
            assert isinstance(prov["page"], int)
            assert isinstance(prov["section"], str)

    # every doc referenced in a task's provenance should actually be one we built
    for example in examples:
        for prov in example["provenance"]:
            assert prov["paper_id"].startswith("doc")


def test_build_examples_skips_when_response_generator_returns_none(three_papers):
    examples, stats = dataset_builder.build_examples(
        three_papers, response_generator=lambda i, c, t: None, rng=random.Random(1)
    )
    assert examples == []
    assert stats["skipped_no_response"] > 0
    assert any("0 training examples emitted" in w for w in stats["warnings"])


def test_build_examples_skips_multipaper_tasks_with_too_few_papers():
    one_paper = [_make_paper("doc1", "Solo Paper")]
    examples, stats = dataset_builder.build_examples(
        one_paper, response_generator=fake_response_generator, rng=random.Random(1)
    )
    task_types_seen = {ex["task_type"] for ex in examples}
    assert "methodology_comparison" not in task_types_seen
    assert "research_gap_identification" not in task_types_seen
    assert stats["pool_sizes"]["methodology_comparison"] == 0
    assert any("methodology_comparison" in w for w in stats["warnings"])


# ---------------------------------------------------------------------------
# allocate_counts (pure)
# ---------------------------------------------------------------------------

def test_allocate_counts_never_exceeds_pool_size():
    pool_sizes = {"a": 2, "b": 10, "c": 0}
    target_ratios = {"a": 0.5, "b": 0.5, "c": 0.0}
    counts = dataset_builder.allocate_counts(pool_sizes, target_ratios)
    assert counts["a"] <= pool_sizes["a"]
    assert counts["b"] <= pool_sizes["b"]
    assert counts["c"] == 0


# ---------------------------------------------------------------------------
# split_papers (prepare_dataset.py) — paper-level, no leakage
# ---------------------------------------------------------------------------

def test_split_papers_is_deterministic_and_covers_all_ids():
    ids = [f"doc{i}" for i in range(1, 11)]
    train_a, val_a, test_a = split_papers(ids, val_ratio=0.1, test_ratio=0.1, seed=42)
    train_b, val_b, test_b = split_papers(ids, val_ratio=0.1, test_ratio=0.1, seed=42)

    assert (train_a, val_a, test_a) == (train_b, val_b, test_b)
    assert set(train_a) | set(val_a) | set(test_a) == set(ids)
    assert not (set(train_a) & set(val_a))
    assert not (set(train_a) & set(test_a))
    assert not (set(val_a) & set(test_a))
    assert len(train_a) + len(val_a) + len(test_a) == len(ids)


def test_split_papers_handles_tiny_inputs_without_crashing():
    train_ids, val_ids, test_ids = split_papers(["doc1"], val_ratio=0.1, test_ratio=0.1, seed=42)
    assert train_ids == ["doc1"]
    assert val_ids == []
    assert test_ids == []


def test_full_pipeline_no_paper_leaks_chunks_across_splits(six_papers):
    """Reproduces prepare_dataset.py's flow: split papers first, then build
    examples separately per split, so a multi-paper example can never mix
    papers assigned to different splits."""
    document_ids = [p["document_id"] for p in six_papers]
    train_ids, val_ids, test_ids = split_papers(document_ids, val_ratio=0.1, test_ratio=0.1, seed=42)
    papers_by_id = {p["document_id"]: p for p in six_papers}

    split_paper_ids = {"train": set(train_ids), "val": set(val_ids), "test": set(test_ids)}
    referenced_paper_ids: dict[str, set[str]] = {}

    for split_name, ids in (("train", train_ids), ("val", val_ids), ("test", test_ids)):
        subset = [papers_by_id[i] for i in ids]
        examples, _ = dataset_builder.build_examples(
            subset, response_generator=fake_response_generator, rng=random.Random(42)
        )
        referenced = set()
        for example in examples:
            for prov in example["provenance"]:
                referenced.add(prov["paper_id"])
        referenced_paper_ids[split_name] = referenced

        # every paper referenced by this split's examples must belong to this split
        assert referenced.issubset(split_paper_ids[split_name])

    # and no paper's chunks show up as "referenced" in more than one split
    assert not (referenced_paper_ids["train"] & referenced_paper_ids["val"])
    assert not (referenced_paper_ids["train"] & referenced_paper_ids["test"])
    assert not (referenced_paper_ids["val"] & referenced_paper_ids["test"])
