"""GET /api/evaluation must serve real results when they exist and say "not evaluated" otherwise."""
from __future__ import annotations

import json
from types import SimpleNamespace

from backend.app.api import evaluation

COMPARISON = {
    "n": 14,
    "prompts": "6 validation + 8 test held-out prompts",
    "caveats": ["Only 14 prompts."],
    "configs": [
        {"name": "Base 1.5B", "n": 14, "cites_any": 0, "valid_ids": 0, "passes_checker": 0,
         "val_passes": 0, "val_n": 6, "test_passes": 0, "test_n": 8, "avg_sentence_coverage": 0.0},
        {"name": "qwen2.5:3b + style hint", "n": 14, "cites_any": 14, "valid_ids": 14, "passes_checker": 9,
         "val_passes": 3, "val_n": 6, "test_passes": 6, "test_n": 8, "avg_sentence_coverage": 0.94,
         "avg_answer_words": 53.0, "avg_citations": 3.0},
    ],
}


def _use_repo(monkeypatch, tmp_path):
    monkeypatch.setattr(evaluation, "get_settings", lambda: SimpleNamespace(repo_root=tmp_path))


def test_no_result_files_means_not_evaluated(monkeypatch, tmp_path):
    _use_repo(monkeypatch, tmp_path)
    summary = evaluation.get_evaluation_summary()
    assert summary.evaluated is False
    assert summary.citation_comparison is None
    assert "Not evaluated yet" in summary.note


def test_citation_comparison_is_served_when_present(monkeypatch, tmp_path):
    _use_repo(monkeypatch, tmp_path)
    d = tmp_path / "experiments" / "finetuned"
    d.mkdir(parents=True)
    (d / "citation_comparison.json").write_text(json.dumps(COMPARISON), encoding="utf-8")

    summary = evaluation.get_evaluation_summary()

    assert summary.evaluated is True
    assert summary.note is None
    assert [c.name for c in summary.citation_comparison.configs] == ["Base 1.5B", "qwen2.5:3b + style hint"]
    assert summary.citation_comparison.configs[1].passes_checker == 9
    assert summary.retrieval_metrics is None  # retrieval quality is genuinely unmeasured: never invented


def test_malformed_comparison_file_is_not_treated_as_a_result(monkeypatch, tmp_path):
    _use_repo(monkeypatch, tmp_path)
    d = tmp_path / "experiments" / "finetuned"
    d.mkdir(parents=True)
    (d / "citation_comparison.json").write_text('{"n": "not-a-number", "configs": [{}]}', encoding="utf-8")
    assert evaluation.get_evaluation_summary().evaluated is False
