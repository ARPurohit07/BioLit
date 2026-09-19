"""Builds SFT instruction examples from processed-paper JSON for BioLit fine-tuning.

Consumes the `data/processed/{document_id}.json` files produced by
`scripts/ingest.py` (see backend/app/ingestion). Each file has a `chunks`
list shaped like backend/app/models/schemas.py::Chunk (chunk_id,
document_id, page_number, section, text, token_count).

Task distribution (per the product spec), applied as closely as the
available papers/sections allow:
    30% paper summarization           (single paper: Abstract+Introduction+Results)
    20% methodology comparison        (two papers, Methods-section chunks)
    15% results comparison            (two papers, Results-section chunks)
    15% limitation identification     (single paper, Limitations/Discussion chunks)
    10% research-gap identification   (3+ papers, Discussion/Limitations/Conclusion)
    10% multi-paper literature synthesis (3+ papers, mixed sections)

WHY "response_generator" is pluggable and defaults to local Ollama
--------------------------------------------------------------------
There is no teacher LLM call authorized for this pipeline (no cloud APIs),
and there are no real reference papers available in this environment to
hand-write biomedical "gold" responses from — doing so would mean
hard-coding fabricated biomedical claims into training data, which the
project spec explicitly forbids ("do not hard-code biomedical facts into
the application").

Instead, this module builds the full example scaffold (instruction,
grounding context selected from real ingested chunks, and provenance)
itself, and delegates ONLY the prose drafting to a `response_generator`
callback: `response_generator(instruction, context, task_type) -> str | None`.
The default implementation is a minimal local HTTP client that calls a
locally running Ollama model (see OllamaResponseGenerator below) with a
prompt that:
  - only allows claims supported by the supplied context,
  - requires the SUPPORTED CLAIM / EVIDENCE / INTERPRETATION / LIMITATION
    structure used throughout the BioLit spec,
  - forbids citation numbers/brackets entirely (real [n] citations are
    attached later by the RAG system at inference time; these examples
    should teach analysis quality and structure, not citation mechanics),
  - forbids inventing findings, and asks the model to say so when the
    context is insufficient.

This "self-instruct from local, retrieval-grounded evidence" approach is a
legitimate, low-cost way to bootstrap SFT data from real ingested papers
without any cloud dependency. It is NOT a substitute for review: a human
should spot-check a sample of `data/training/train.jsonl` before training
on it. If Ollama is unreachable, the default generator returns None for
every candidate and the builder emits 0 examples with a clear message,
rather than fabricating response text itself.
"""
from __future__ import annotations

import itertools
import json
import os
import random
from pathlib import Path
from typing import Any, Callable, Optional

ResponseGenerator = Callable[[str, str, str], Optional[str]]

# ---------------------------------------------------------------------------
# Task types / target distribution
# ---------------------------------------------------------------------------

DEFAULT_TARGET_DISTRIBUTION: dict[str, float] = {
    "summarization": 0.30,
    "methodology_comparison": 0.20,
    "results_comparison": 0.15,
    "limitation_identification": 0.15,
    "research_gap_identification": 0.10,
    "literature_synthesis": 0.10,
}

SUMMARY_SECTIONS = {"Abstract", "Introduction", "Results"}
METHODS_SECTIONS = {"Methods"}
RESULTS_SECTIONS = {"Results"}
LIMITATIONS_SECTIONS = {"Limitations", "Discussion"}
RESEARCH_GAP_SECTIONS = {"Discussion", "Limitations", "Conclusion"}
SYNTHESIS_SECTIONS = {"Abstract", "Introduction", "Methods", "Results", "Discussion", "Conclusion"}

NO_PAPERS_MESSAGE = (
    "No processed papers found in data/processed — dataset will be empty; "
    "add PDFs to data/raw and run scripts/ingest.py + scripts/build_index.py first"
)


# ---------------------------------------------------------------------------
# Loading processed papers
# ---------------------------------------------------------------------------

def load_processed_papers(processed_dir: str | Path) -> list[dict[str, Any]]:
    """Read every data/processed/{document_id}.json file. Returns [] (and
    prints NO_PAPERS_MESSAGE) gracefully if the directory is missing/empty."""
    processed_dir = Path(processed_dir)
    if not processed_dir.exists():
        print(NO_PAPERS_MESSAGE)
        return []

    paper_files = sorted(processed_dir.glob("*.json"))
    if not paper_files:
        print(NO_PAPERS_MESSAGE)
        return []

    papers = []
    for path in paper_files:
        try:
            with open(path, "r", encoding="utf-8") as f:
                paper = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"[dataset_builder] Skipping unreadable processed file {path}: {exc}")
            continue
        if not paper.get("chunks"):
            continue
        papers.append(paper)
    return papers


# ---------------------------------------------------------------------------
# Context / provenance formatting (pure helpers)
# ---------------------------------------------------------------------------

def _chunks_by_sections(paper: dict[str, Any], sections: set[str]) -> list[dict[str, Any]]:
    return [c for c in paper.get("chunks", []) if c.get("section") in sections]


def _sorted_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(chunks, key=lambda c: (c.get("page_number") or 0, c.get("chunk_id") or ""))


def _format_paper_block(paper: dict[str, Any], chunks: list[dict[str, Any]], label: str) -> str:
    lines = [f"=== {label}: {paper.get('title') or paper.get('document_id')} ({paper.get('document_id')}) ==="]
    for c in _sorted_chunks(chunks):
        lines.append(f"[Section: {c.get('section', 'Unknown')} | Page {c.get('page_number', '?')}]")
        lines.append((c.get("text") or "").strip())
        lines.append("")
    return "\n".join(lines).strip()


def format_single_paper_context(paper: dict[str, Any], chunks: list[dict[str, Any]]) -> str:
    return _format_paper_block(paper, chunks, "Paper")


def format_multi_paper_context(paper_chunk_pairs: list[tuple[dict[str, Any], list[dict[str, Any]]]]) -> str:
    blocks = [
        _format_paper_block(paper, chunks, f"Paper {idx}")
        for idx, (paper, chunks) in enumerate(paper_chunk_pairs, start=1)
    ]
    return "\n\n".join(blocks)


def provenance_from_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"paper_id": c.get("document_id"), "page": c.get("page_number"), "section": c.get("section")}
        for c in chunks
    ]


# ---------------------------------------------------------------------------
# Candidate builders per task type
# (instruction + context + provenance; response filled in later)
# ---------------------------------------------------------------------------

def _build_summarization_candidates(papers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for paper in papers:
        chunks = _chunks_by_sections(paper, SUMMARY_SECTIONS)
        if not chunks:
            continue
        out.append({
            "instruction": "Summarize this paper's key findings.",
            "context": format_single_paper_context(paper, chunks),
            "provenance": provenance_from_chunks(chunks),
            "task_type": "summarization",
        })
    return out


def _build_pairwise_candidates(
    papers: list[dict[str, Any]],
    sections: set[str],
    instruction: str,
    task_type: str,
    rng: random.Random,
    max_pairs: int = 30,
) -> list[dict[str, Any]]:
    qualifying = [(p, _chunks_by_sections(p, sections)) for p in papers]
    qualifying = [(p, c) for p, c in qualifying if c]
    if len(qualifying) < 2:
        return []

    pairs = list(itertools.combinations(range(len(qualifying)), 2))
    if len(pairs) > max_pairs:
        pairs = rng.sample(pairs, max_pairs)
        pairs.sort()

    out = []
    for i, j in pairs:
        p1, c1 = qualifying[i]
        p2, c2 = qualifying[j]
        out.append({
            "instruction": instruction,
            "context": format_multi_paper_context([(p1, c1), (p2, c2)]),
            "provenance": provenance_from_chunks(c1) + provenance_from_chunks(c2),
            "task_type": task_type,
        })
    return out


def _build_limitations_candidates(papers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for paper in papers:
        chunks = _chunks_by_sections(paper, LIMITATIONS_SECTIONS)
        if not chunks:
            continue
        out.append({
            "instruction": "Identify the limitations discussed in this paper.",
            "context": format_single_paper_context(paper, chunks),
            "provenance": provenance_from_chunks(chunks),
            "task_type": "limitation_identification",
        })
    return out


def _build_group_candidates(
    papers: list[dict[str, Any]],
    sections: set[str],
    instruction: str,
    task_type: str,
    rng: random.Random,
    group_size: int,
    min_group: int,
    max_groups: int = 10,
) -> list[dict[str, Any]]:
    qualifying = [(p, _chunks_by_sections(p, sections)) for p in papers]
    qualifying = [(p, c) for p, c in qualifying if c]
    if len(qualifying) < min_group:
        return []

    order = list(range(len(qualifying)))
    rng.shuffle(order)

    out = []
    for start in range(0, len(order), group_size):
        idxs = order[start:start + group_size]
        if len(idxs) < min_group:
            continue  # leftover remainder too small to form a valid group
        group = [qualifying[i] for i in idxs]
        out.append({
            "instruction": instruction,
            "context": format_multi_paper_context(group),
            "provenance": [prov for _, c in group for prov in provenance_from_chunks(c)],
            "task_type": task_type,
        })
        if len(out) >= max_groups:
            break
    return out


def build_candidate_pools(papers: list[dict[str, Any]], rng: random.Random) -> dict[str, list[dict[str, Any]]]:
    """Pure(ish) w.r.t. rng: builds all feasible candidate examples per task type."""
    return {
        "summarization": _build_summarization_candidates(papers),
        "methodology_comparison": _build_pairwise_candidates(
            papers, METHODS_SECTIONS,
            "Compare the methodologies used in these studies.",
            "methodology_comparison", rng,
        ),
        "results_comparison": _build_pairwise_candidates(
            papers, RESULTS_SECTIONS,
            "Compare the results reported in these studies.",
            "results_comparison", rng,
        ),
        "limitation_identification": _build_limitations_candidates(papers),
        "research_gap_identification": _build_group_candidates(
            papers, RESEARCH_GAP_SECTIONS,
            "Identify potential research gaps across these studies.",
            "research_gap_identification", rng, group_size=4, min_group=3,
        ),
        "literature_synthesis": _build_group_candidates(
            papers, SYNTHESIS_SECTIONS,
            "Synthesize the current state of research across these papers, "
            "noting where they agree and where they diverge.",
            "literature_synthesis", rng, group_size=5, min_group=3,
        ),
    }


def allocate_counts(pool_sizes: dict[str, int], target_ratios: dict[str, float]) -> dict[str, int]:
    """Pure function: given how many candidate examples are actually available
    per task type, and the target proportions, pick how many to draw from
    each pool so the overall mix stays as close as feasible to target_ratios
    without exceeding any pool's capacity. Categories with 0 candidates are
    allocated 0 (the caller is responsible for warning about those)."""
    feasible = {k: v for k, v in pool_sizes.items() if v > 0 and target_ratios.get(k, 0) > 0}
    if not feasible:
        return {k: 0 for k in pool_sizes}

    implied_totals = [feasible[k] / target_ratios[k] for k in feasible]
    total = min(implied_totals)

    counts = {}
    for k in pool_sizes:
        if k in feasible:
            counts[k] = min(pool_sizes[k], round(total * target_ratios[k]))
        else:
            counts[k] = 0
    return counts


# ---------------------------------------------------------------------------
# Response prompt + default local-Ollama generator
# ---------------------------------------------------------------------------

def build_response_prompt(instruction: str, context: str) -> str:
    return f"""You are drafting SUPERVISED FINE-TUNING training data for a biomedical \
literature assistant. Using ONLY the evidence in CONTEXT below, write a response to \
INSTRUCTION using this exact repeating structure for each distinct point:

SUPPORTED CLAIM: <a claim directly stated in the context>
EVIDENCE: <the specific text/finding from the context that supports it>
INTERPRETATION: <your interpretation or significance, clearly marked as interpretation, not fact>
LIMITATION: <a limitation of the evidence/scope, or "None noted in the provided context">

Rules (all required):
- Do NOT include citation numbers, footnote markers, or bracketed references like [1] \
anywhere in the response. Real citations are attached separately at inference time; this \
response should teach analysis quality and structure, not citation mechanics.
- Do NOT state anything as fact unless it is directly supported by CONTEXT.
- Do NOT invent author names, statistics, study details, or findings not present in CONTEXT.
- If CONTEXT is insufficient to address INSTRUCTION, say so explicitly instead of guessing.

INSTRUCTION: {instruction}

CONTEXT:
{context}

RESPONSE:"""


class OllamaResponseGenerator:
    """Minimal local HTTP client for POST {host}/api/generate. No cloud APIs.

    After the first failed call (connection refused, timeout, non-2xx, or
    empty response), marks itself unavailable and returns None immediately
    for all further calls rather than re-attempting a dead connection for
    every remaining candidate.
    """

    def __init__(self, host: Optional[str] = None, model: Optional[str] = None, timeout: int = 120):
        self.host = (host or os.environ.get("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")
        self.model = model or "qwen2.5:1.5b-instruct"
        self.timeout = timeout
        self._unavailable = False

    def __call__(self, instruction: str, context: str, task_type: str) -> Optional[str]:
        if self._unavailable:
            return None
        import requests

        prompt = build_response_prompt(instruction, context)
        try:
            resp = requests.post(
                f"{self.host}/api/generate",
                json={"model": self.model, "prompt": prompt, "stream": False},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            text = (resp.json().get("response") or "").strip()
        except Exception as exc:  # noqa: BLE001 - any failure means "unavailable"
            self._unavailable = True
            print(
                f"[dataset_builder] Ollama call failed at {self.host} (model={self.model}): {exc}. "
                "Remaining candidate examples will be skipped — start Ollama "
                "('ollama serve') and ensure the model is pulled, then re-run "
                "training/prepare_dataset.py."
            )
            return None

        if not text:
            self._unavailable = True
            print(
                f"[dataset_builder] Ollama returned an empty response from model={self.model}; "
                "treating as unavailable for remaining candidates."
            )
            return None
        return text


def make_default_response_generator(models_config: Optional[dict[str, Any]] = None) -> OllamaResponseGenerator:
    """Builds the default Ollama-backed generator. Prefers the base model
    (base_model_name) over the fine-tuned target (model_name) since the
    fine-tuned model won't exist yet the first time this pipeline runs."""
    host = None
    model = None
    timeout = 120
    if models_config:
        ollama_cfg = models_config.get("ollama", {}) or {}
        host = ollama_cfg.get("host")
        model = ollama_cfg.get("base_model_name") or ollama_cfg.get("model_name")
        timeout = ollama_cfg.get("request_timeout_s", 120)
    return OllamaResponseGenerator(host=host, model=model, timeout=timeout)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def build_examples(
    papers: list[dict[str, Any]],
    response_generator: Optional[ResponseGenerator] = None,
    rng: Optional[random.Random] = None,
    target_distribution: Optional[dict[str, float]] = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Builds the full example list honoring target_distribution as closely
    as feasible given available papers/sections. Returns (examples, stats).

    Each example: {"instruction", "context", "response", "provenance", "task_type"}.
    `task_type` is included for reporting/splitting purposes; callers that
    need the exact {instruction, context, response, provenance} shape (e.g.
    when writing JSONL) should drop it before serializing.
    """
    if rng is None:
        rng = random.Random(42)
    if target_distribution is None:
        target_distribution = DEFAULT_TARGET_DISTRIBUTION

    warnings: list[str] = []
    if not papers:
        warnings.append(NO_PAPERS_MESSAGE)
        return [], {
            "pool_sizes": {k: 0 for k in target_distribution},
            "allocated": {k: 0 for k in target_distribution},
            "generated": {k: 0 for k in target_distribution},
            "skipped_no_response": 0,
            "warnings": warnings,
        }

    candidate_pools = build_candidate_pools(papers, rng)
    pool_sizes = {k: len(v) for k, v in candidate_pools.items()}
    for task_type, size in pool_sizes.items():
        if size == 0:
            warnings.append(
                f"Skipping task type '{task_type}': no feasible candidates from the "
                "available papers/sections (need more papers and/or that section present)."
            )

    allocated = allocate_counts(pool_sizes, target_distribution)

    if response_generator is None:
        response_generator = make_default_response_generator()

    examples: list[dict[str, Any]] = []
    generated_counts = {k: 0 for k in candidate_pools}
    skipped_no_response = 0

    for task_type, pool in candidate_pools.items():
        count = allocated.get(task_type, 0)
        if count <= 0:
            continue
        selected = rng.sample(pool, count) if count < len(pool) else list(pool)
        for cand in selected:
            response = response_generator(cand["instruction"], cand["context"], cand["task_type"])
            if not response:
                skipped_no_response += 1
                continue
            examples.append({
                "instruction": cand["instruction"],
                "context": cand["context"],
                "response": response,
                "provenance": cand["provenance"],
                "task_type": cand["task_type"],
            })
            generated_counts[task_type] += 1

    if not examples:
        if skipped_no_response > 0:
            warnings.append(
                f"{skipped_no_response} candidate examples were built but the response "
                "generator returned nothing (Ollama unavailable/empty) - 0 training "
                "examples emitted. Start Ollama and re-run to populate the dataset."
            )
        else:
            warnings.append("No candidate examples could be constructed from the available processed papers.")

    stats = {
        "pool_sizes": pool_sizes,
        "allocated": allocated,
        "generated": generated_counts,
        "skipped_no_response": skipped_no_response,
        "warnings": warnings,
    }
    return examples, stats
