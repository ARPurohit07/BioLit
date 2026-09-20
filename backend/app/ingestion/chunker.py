"""Page-aware chunking: never splits a chunk across pages, prefers sentence boundaries.

Token counting prefers a real tokenizer (transformers, cache-only so it never blocks on
network) and falls back to a whitespace-split approximation when unavailable.
"""
from __future__ import annotations

import os
import re

import numpy as np

from backend.app.ingestion.pdf_loader import PageText
from backend.app.ingestion.section_detector import SectionDetector
from backend.app.ingestion.structure import table_to_markdown
from backend.app.models.schemas import Chunk

# Splits after sentence-ending punctuation, only when followed by a capital letter/digit
# (avoids splitting on abbreviations like "e.g." followed by lowercase text).
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")

_tokenizer = None
_tokenizer_load_attempted = False

# Keyed by model name so a different embedding_model_name isn't stuck with the first
# load's outcome; a failed load caches None so we don't retry it per section-run.
_embedding_models: dict[str, object] = {}


def _get_embedding_model(model_name: str):
    if model_name not in _embedding_models:
        # Cache-only for this call: an uncached model must never block ingestion on a
        # network fetch. Restored afterwards so it doesn't leak into unrelated code paths
        # (e.g. the app's own first-run model download).
        prev_offline = os.environ.get("HF_HUB_OFFLINE")
        os.environ["HF_HUB_OFFLINE"] = "1"
        try:
            from backend.app.retrieval.embeddings import EmbeddingModel
            _embedding_models[model_name] = EmbeddingModel(model_name)
        except Exception:
            _embedding_models[model_name] = None
        finally:
            if prev_offline is None:
                os.environ.pop("HF_HUB_OFFLINE", None)
            else:
                os.environ["HF_HUB_OFFLINE"] = prev_offline
    return _embedding_models[model_name]


def _get_tokenizer():
    global _tokenizer, _tokenizer_load_attempted
    if not _tokenizer_load_attempted:
        _tokenizer_load_attempted = True
        try:
            from transformers import AutoTokenizer
            # local_files_only avoids any network call when the tokenizer isn't cached.
            _tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", local_files_only=True)
        except Exception:
            _tokenizer = None
    return _tokenizer


def count_tokens(text: str) -> int:
    tokenizer = _get_tokenizer()
    if tokenizer is not None:
        try:
            return len(tokenizer.encode(text, add_special_tokens=False))
        except Exception:
            pass
    return len(text.split())


def _split_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    return [p.strip() for p in _SENTENCE_SPLIT_RE.split(text) if p.strip()]


class PageAwareChunker:
    def __init__(
        self,
        chunk_size: int,
        chunk_overlap: int,
        min_chunk_tokens: int,
        semantic: bool = True,
        max_chunk_tokens: int = 220,
        breakpoint_percentile: int = 80,
        embedding_model_name: str = "BAAI/bge-small-en-v1.5",
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_tokens = min_chunk_tokens
        # Semantic (embedding-boundary) packing for text runs; falls back to the fixed
        # chunk_size packing below if disabled or if the embedding model is unavailable.
        self.semantic = semantic
        self.max_chunk_tokens = max_chunk_tokens
        self.breakpoint_percentile = breakpoint_percentile
        self.embedding_model_name = embedding_model_name

    def chunk_document(self, document_id: str, pages: list[PageText],
                        section_detector: SectionDetector) -> list[Chunk]:
        """Chunks are never split across a section boundary: a page is first cut into
        runs at each detected heading line (the heading itself is dropped from the
        content), then each run is packed into chunks independently. A page with no
        heading of its own inherits the section that was active at the end of the
        previous page, rather than falling back to "Unknown" — headings routinely sit
        mid-page (after a figure caption, etc.) or on an earlier page than the content
        they introduce."""
        chunks: list[Chunk] = []
        current_section = "Unknown"

        for page in pages:
            if not page.text.strip() and not page.tables and not page.figures:
                continue

            page_chunk_index = 0
            # A page can hold only a table or a full-page figure; its body text is then empty but it still yields chunks.
            for section, run_text in self._section_runs(page.text, section_detector, current_section):
                # Once References has started, pin the section there rather than letting
                # a numbered appendix/checklist item (e.g. a NeurIPS "2. Limitations"
                # reproducibility-checklist question) resurrect an earlier section label —
                # real papers essentially never have a genuine Methods/Results/Limitations
                # section after References.
                if current_section != "References" or section == "References":
                    current_section = section
                run_chunk_texts = self._merge_small(self._pack_page(_split_sentences(run_text)))
                for run_chunk_text in run_chunk_texts:
                    chunks.append(Chunk(
                        chunk_id=f"{document_id}_p{page.page_number}_c{page_chunk_index}",
                        document_id=document_id,
                        page_number=page.page_number,
                        section=current_section,
                        text=run_chunk_text,
                        token_count=count_tokens(run_chunk_text),
                    ))
                    page_chunk_index += 1

            # No References guard here: a table or figure after the bibliography is an appendix item, not a citation.
            chunks += self._structural_chunks(document_id, page, current_section)

        return chunks

    def _structural_chunks(self, document_id: str, page: PageText, section: str) -> list[Chunk]:
        """One chunk per figure (caption + in-figure text) and one or more per table (caption + Markdown grid).
        A long table is split by rows, repeating the caption and header so every part reads on its own."""
        out: list[Chunk] = []
        for n, fig in enumerate(page.figures):
            text = fig.caption + (f"\nText in the figure: {fig.figure_text}" if fig.figure_text else "")
            out.append(Chunk(
                chunk_id=f"{document_id}_p{page.page_number}_f{n}", document_id=document_id,
                page_number=page.page_number, section=section, text=text, token_count=count_tokens(text),
                chunk_type="figure", label=fig.label, image_path=fig.image_path,
            ))
        part = 0
        for tb in page.tables:
            header, rows = tb.rows[0], tb.rows[1:]
            prefix = tb.caption or tb.label
            budget = max(self.chunk_size - count_tokens(prefix) - count_tokens(table_to_markdown([header])), 60)
            groups, current, used = [], [], 0
            for r in rows:
                t = count_tokens(" | ".join(r))
                if current and used + t > budget:
                    groups.append(current)
                    current, used = [], 0
                current.append(r)
                used += t
            if current or not groups:
                groups.append(current)
            for g in groups:
                text = f"{prefix}\n{table_to_markdown([header] + g)}"
                out.append(Chunk(
                    chunk_id=f"{document_id}_p{page.page_number}_t{part}", document_id=document_id,
                    page_number=page.page_number, section=section, text=text, token_count=count_tokens(text),
                    chunk_type="table", label=tb.label,
                ))
                part += 1
        return out

    def _section_runs(self, page_text: str, section_detector: SectionDetector,
                       starting_section: str) -> list[tuple[str, str]]:
        """Splits page_text at heading lines into [(section, text), ...] runs, carrying
        starting_section forward until the first heading (if any) is found."""
        lines = page_text.splitlines()
        headings = dict(section_detector.scan_headings(page_text))

        runs: list[tuple[str, str]] = []
        current_section = starting_section
        buffer: list[str] = []

        for idx, line in enumerate(lines):
            if idx in headings:
                if buffer:
                    runs.append((current_section, " ".join(buffer)))
                    buffer = []
                current_section = headings[idx]
                continue  # the heading line itself isn't kept as chunk content
            if line.strip():
                buffer.append(line.strip())

        if buffer:
            runs.append((current_section, " ".join(buffer)))

        return runs

    def _pack_page(self, sentences: list[str]) -> list[str]:
        """Packs one section-run's sentences into chunks. Semantic mode (default) cuts
        at embedding topic-shift boundaries under a max_chunk_tokens budget; otherwise
        (or if the embedding model can't be built/run) falls back to fixed packing."""
        if self.semantic and len(sentences) >= 2:
            model = _get_embedding_model(self.embedding_model_name)
            if model is not None:
                try:
                    embeddings = model.encode(sentences)
                    distances = self._adjacent_distances(embeddings)
                    threshold = float(np.percentile(distances, self.breakpoint_percentile))
                except Exception:
                    model = None
                if model is not None:
                    packed = self._pack_by_topic_breaks(sentences, distances, threshold)
                    return [piece for text in packed for piece in self._hard_split(text)]

        return self._pack_page_fixed(sentences)

    def _pack_page_fixed(self, sentences: list[str]) -> list[str]:
        """Greedily packs sentences into ~chunk_size-token windows with token overlap."""
        chunks: list[str] = []
        current: list[str] = []
        current_tokens = 0
        i = 0
        while i < len(sentences):
            sent = sentences[i]
            sent_tokens = count_tokens(sent)

            if current and current_tokens + sent_tokens > self.chunk_size:
                chunks.append(" ".join(current))
                current, current_tokens = self._overlap_or_empty(current, sent_tokens, self.chunk_size)
                continue  # retry the same sentence against the trimmed (or dropped) window

            current.append(sent)
            current_tokens += sent_tokens
            i += 1

        if current:
            chunks.append(" ".join(current))
        return chunks

    def _adjacent_distances(self, embeddings: np.ndarray) -> np.ndarray:
        """Cosine distance (1 - similarity) between each pair of consecutive rows."""
        a, b = embeddings[:-1], embeddings[1:]
        denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
        denom = np.where(denom == 0, 1e-8, denom)
        similarity = np.sum(a * b, axis=1) / denom
        return 1.0 - similarity

    def _pack_by_topic_breaks(self, sentences: list[str], distances: np.ndarray, threshold: float) -> list[str]:
        """Starts a new chunk after any sentence whose distance to its successor exceeds
        threshold, and also whenever max_chunk_tokens would otherwise be exceeded."""
        chunks: list[str] = []
        current: list[str] = []
        current_tokens = 0
        idx = 0
        while idx < len(sentences):
            sent = sentences[idx]
            sent_tokens = count_tokens(sent)

            if current and current_tokens + sent_tokens > self.max_chunk_tokens:
                chunks.append(" ".join(current))
                current, current_tokens = self._overlap_or_empty(current, sent_tokens, self.max_chunk_tokens)
                continue  # retry the same sentence against the trimmed (or dropped) window

            current.append(sent)
            current_tokens += sent_tokens

            if idx < len(distances) and distances[idx] > threshold:
                chunks.append(" ".join(current))
                current, current_tokens = self._trailing_overlap(current)

            idx += 1

        if current:
            chunks.append(" ".join(current))
        return chunks

    def _hard_split(self, text: str) -> list[str]:
        """Word-level fallback when a chunk (typically a single long sentence) is still
        over max_chunk_tokens after topic-based packing."""
        if count_tokens(text) <= self.max_chunk_tokens:
            return [text]
        parts: list[str] = []
        current: list[str] = []
        for word in text.split():
            current.append(word)
            if count_tokens(" ".join(current)) >= self.max_chunk_tokens:
                parts.append(" ".join(current))
                current = []
        if current:
            parts.append(" ".join(current))
        return parts

    def _overlap_or_empty(self, current: list[str], sent_tokens: int, budget: int) -> tuple[list[str], int]:
        """Overlap to carry into the next window after a budget-forced flush. Dropped
        entirely (forcing a clean start) if it would still be too large to admit the next
        sentence — otherwise a sentence over budget on its own would repeat the same
        flush forever, since the overlap of an unchanged window is itself unchanged."""
        overlap, overlap_tokens = self._trailing_overlap(current)
        if overlap_tokens + sent_tokens > budget:
            return [], 0
        return overlap, overlap_tokens

    def _trailing_overlap(self, sentences: list[str]) -> tuple[list[str], int]:
        overlap_sents: list[str] = []
        overlap_tokens = 0
        for s in reversed(sentences):
            t = count_tokens(s)
            if overlap_tokens + t > self.chunk_overlap:
                break
            overlap_sents.insert(0, s)
            overlap_tokens += t
        return overlap_sents, overlap_tokens

    def _merge_small(self, chunk_texts: list[str]) -> list[str]:
        """Merges a chunk below min_chunk_tokens into the previous chunk on the same page."""
        if len(chunk_texts) <= 1:
            return chunk_texts

        merged: list[str] = [chunk_texts[0]]
        for text in chunk_texts[1:]:
            if count_tokens(text) < self.min_chunk_tokens:
                merged[-1] = merged[-1] + " " + text
            else:
                merged.append(text)
        return merged
