"""Evaluation summary endpoint. Never fabricates numbers: if no result files
exist under experiments/, it says so plainly instead of returning zeros or
made-up metrics."""
from __future__ import annotations

import json

from fastapi import APIRouter

from backend.app.config.settings import get_settings
from backend.app.models.schemas import (
    CitationComparison,
    EvaluationSummary,
    GenerationMetrics,
    LatencyBenchmarkEntry,
    RetrievalMetrics,
)

router = APIRouter(prefix="/api/evaluation", tags=["evaluation"])

_NOT_EVALUATED = EvaluationSummary(
    evaluated=False,
    variant="none",
    note="Not evaluated yet. Run scripts/benchmark.py and training/evaluate.py to populate this.",
)


def _load_result_files() -> list[dict]:
    settings = get_settings()
    experiments_dir = settings.repo_root / "experiments"
    results = []
    if not experiments_dir.exists():
        return results
    for path in experiments_dir.rglob("*.json"):
        try:
            with open(path, "r", encoding="utf-8") as f:
                results.append(json.load(f))
        except (json.JSONDecodeError, OSError):
            continue
    return results


def _load_citation_comparison() -> CitationComparison | None:
    """experiments/finetuned/citation_comparison.json, written by scripts/summarize_citation_evals.py."""
    path = get_settings().repo_root / "experiments" / "finetuned" / "citation_comparison.json"
    try:
        with open(path, "r", encoding="utf-8") as f:
            return CitationComparison(**json.load(f))
    except (OSError, ValueError):  # missing file, bad JSON or a schema mismatch: report "not evaluated", never guess
        return None


@router.get("/", response_model=EvaluationSummary)
def get_evaluation_summary() -> EvaluationSummary:
    result_files = _load_result_files()
    citation_comparison = _load_citation_comparison()
    if not result_files and citation_comparison is None:
        return _NOT_EVALUATED

    latency_entries: list[LatencyBenchmarkEntry] = []
    retrieval_metrics = None
    generation_metrics = None
    variant = "unknown"

    for result in result_files:
        variant = result.get("variant", variant)
        if result.get("retrieval_metrics") and retrieval_metrics is None:
            try:
                retrieval_metrics = RetrievalMetrics(**result["retrieval_metrics"])
            except Exception:
                pass
        if result.get("generation_metrics") and generation_metrics is None:
            try:
                generation_metrics = GenerationMetrics(**result["generation_metrics"])
            except Exception:
                pass
        for entry in result.get("latency_by_mode", []):
            try:
                latency_entries.append(LatencyBenchmarkEntry(**entry))
            except Exception:
                continue

    if retrieval_metrics is None and generation_metrics is None and not latency_entries and citation_comparison is None:
        note = _NOT_EVALUATED.note + " (Found result files under experiments/ but could not parse any known metrics from them.)"
        return EvaluationSummary(evaluated=False, variant="none", note=note)

    return EvaluationSummary(
        evaluated=True,
        retrieval_metrics=retrieval_metrics,
        generation_metrics=generation_metrics,
        latency_by_mode=latency_entries,
        citation_comparison=citation_comparison,
        variant=variant,
        note=None,
    )
