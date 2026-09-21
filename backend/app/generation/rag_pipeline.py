"""The core RAG orchestrator: implements the FAST / BALANCED / HIGH_FAITHFULNESS
pipelines plus the compare / literature-review / conflict-detection flows on top
of them.

Retrieval and reranking behavior (which stages run) is driven entirely by
configs/retrieval.yaml's `modes` section (passed in as `config`), not by
hard-coded per-mode branching, so tuning retrieval.yaml doesn't require code
changes here.
"""
from __future__ import annotations

import re
import time
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from backend.app import db
from backend.app.generation.ollama_client import OllamaClient
from backend.app.generation.prompts import (
    build_comparison_prompt,
    build_conflict_detection_prompt,
    build_literature_review_section_prompt,
    build_qa_prompt,
    build_research_gap_prompt,
    build_summarization_prompt,
)
from backend.app.models.schemas import (
    Chunk,
    Claim,
    ClaimStatus,
    EvidenceItem,
    LatencyBreakdown,
    QueryResponse,
    QueryType,
    RAGMode,
)
from backend.app.verification.citation_validator import compute_citation_metrics
from backend.app.verification.claims import is_abstention, strip_regeneration_scaffold
from backend.app.verification.grounding import signals as grounding_signals

if TYPE_CHECKING:
    from backend.app.retrieval.hybrid import HybridRetriever
    from backend.app.retrieval.reranker import Reranker
    from backend.app.verification.claims import ClaimExtractor
    from backend.app.verification.verifier import ClaimVerifier

_UNSUPPORTED_NOTE = " *(unsupported — could not be verified against retrieved evidence)*"

_UNSUPPORTED_LIKE = {ClaimStatus.UNSUPPORTED, ClaimStatus.CONTRADICTED}


_MARKER_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


_NO_ANSWER = "The retrieved evidence does not state the answer to this question."


def has_substance(answer: str) -> bool:
    """False for a reply that is only citation markers, labels or punctuation (e.g. "[4]"): it answers nothing."""
    bare = _MARKER_RE.sub("", answer)
    bare = re.sub(r"SUPPORTED CLAIM|INTERPRETATION|LIMITATION", "", bare, flags=re.IGNORECASE)
    return bool(re.search(r"[A-Za-z0-9]{2,}", bare))


def _has_valid_citation(answer: str, evidence: list[EvidenceItem]) -> bool:
    """True if the answer contains at least one [n] marker that refers to a real evidence block."""
    valid = {e.citation_id for e in evidence}
    return any(
        piece.strip().isdigit() and int(piece) in valid
        for match in _MARKER_RE.finditer(answer)
        for piece in match.group(1).split(",")
    )


def _cited_fraction(claims: list[Claim]) -> float:
    return sum(bool(c.citation_ids) for c in claims) / len(claims) if claims else 0.0


def _unsupported_rate(claims: list[Claim], unsupported: list[Claim]) -> float:
    return len(unsupported) / len(claims) if claims else 1.0


_ASPECT_LABELS = {
    QueryType.COMPARE_PAPERS: "overall approach and findings",
    QueryType.COMPARE_METHODOLOGY: "methodology",
    QueryType.COMPARE_RESULTS: "results",
}

_COMPARE_QUESTIONS = {
    QueryType.COMPARE_PAPERS: "Compare these papers overall.",
    QueryType.COMPARE_METHODOLOGY: "Compare the methodology used across these papers.",
    QueryType.COMPARE_RESULTS: "Compare the results and findings reported across these papers.",
}


class RAGPipeline:
    def __init__(
        self,
        retriever: "HybridRetriever",
        reranker: Optional["Reranker"],
        ollama_client: OllamaClient,
        claim_extractor: "ClaimExtractor",
        claim_verifier: "ClaimVerifier",
        config: dict,
    ):
        self.retriever = retriever
        self.reranker = reranker
        self.ollama_client = ollama_client
        self.claim_extractor = claim_extractor
        self.claim_verifier = claim_verifier
        self.config = config or {}
        self._title_cache: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _mode_config(self, mode: RAGMode) -> dict:
        return self.config.get("modes", {}).get(mode.value, {})

    def _title_for(self, document_id: str) -> str:
        if document_id not in self._title_cache:
            meta = db.get_document(document_id)
            self._title_cache[document_id] = meta.title if meta else document_id
        return self._title_cache[document_id]

    def _decompose(self, query: str) -> list[str]:
        """Simple heuristic decomposition: split on 'and'/'compare' style multi-clause
        questions. Not a full agentic decomposition — just enough to retrieve evidence
        for each clause of a compound question separately."""
        lowered = query.lower()
        if " and " not in lowered and "compare" not in lowered:
            return [query]
        parts = [p.strip(" ,.;") for p in re.split(r"\band\b", query, flags=re.IGNORECASE)]
        parts = [p for p in parts if p]
        if len(parts) <= 1:
            return [query]
        return parts

    def _retrieve(
        self, query: str, mode: RAGMode, document_ids: Optional[list[str]], decompose: bool
    ) -> tuple[list[tuple[Chunk, float]], float]:
        t0 = time.perf_counter()
        sub_queries = self._decompose(query) if decompose else [query]
        merged: dict[str, tuple[Chunk, float]] = {}
        for q in sub_queries:
            for chunk, score in self.retriever.retrieve(q, mode=mode.value, document_ids=document_ids):
                existing = merged.get(chunk.chunk_id)
                if existing is None or score > existing[1]:
                    merged[chunk.chunk_id] = (chunk, score)
        candidates = sorted(merged.values(), key=lambda cs: cs[1], reverse=True)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return candidates, latency_ms

    def _rerank(
        self, query: str, candidates: list[tuple[Chunk, float]], use_reranker: bool
    ) -> tuple[list[tuple[Chunk, float]], float]:
        if not use_reranker or not self.reranker or not candidates:
            return candidates, 0.0
        output_k = self.config.get("reranker", {}).get("output_k", 5)
        t0 = time.perf_counter()
        reranked = self.reranker.rerank(query, candidates, top_k=output_k)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return reranked, latency_ms

    def _build_evidence(self, candidates: list[tuple[Chunk, float]]) -> list[EvidenceItem]:
        evidence = []
        for i, (chunk, score) in enumerate(candidates, start=1):
            evidence.append(
                EvidenceItem(
                    citation_id=i,
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    document_title=self._title_for(chunk.document_id),
                    page_number=chunk.page_number,
                    section=chunk.section,
                    text=chunk.text,
                    score=score,
                    chunk_type=chunk.chunk_type,
                    label=chunk.label,
                    image_url=f"/figures/{Path(chunk.image_path).name}" if chunk.image_path else None,
                )
            )
        return evidence

    def _build_prompt(self, query_type: QueryType, question: str, evidence: list[EvidenceItem]) -> tuple[str, str]:
        if query_type == QueryType.SUMMARIZE:
            return build_summarization_prompt(evidence)
        if query_type in _ASPECT_LABELS:
            return build_comparison_prompt(evidence, _ASPECT_LABELS[query_type])
        if query_type == QueryType.RESEARCH_GAPS:
            return build_research_gap_prompt(evidence)
        if query_type == QueryType.CONFLICT_DETECTION:
            return build_conflict_detection_prompt(evidence)
        if query_type == QueryType.LITERATURE_REVIEW:
            return build_literature_review_section_prompt(evidence, subtopic=question)
        if query_type == QueryType.LIMITATIONS:
            q = f"What are the limitations of the studies/methods discussed, in relation to: {question}"
            return build_qa_prompt(q, evidence)
        if query_type == QueryType.STRUCTURED_TABLE:
            q = f"{question}\n\n(Present the answer as a markdown table with clear columns.)"
            return build_qa_prompt(q, evidence)
        return build_qa_prompt(question, evidence)

    def _build_regeneration_prompt(self, user_prompt: str, answer: str, unsupported: list[Claim]) -> str:
        flagged = "\n".join(f"- {c.text}" for c in unsupported)
        return (
            f"{user_prompt}\n\n---\n"
            f"Your previous answer was:\n{answer}\n\n"
            "The following statements from that answer could NOT be verified against the "
            "evidence (they are unsupported or contradicted):\n"
            f"{flagged}\n\n"
            "Revise the answer: remove or correct these specific statements so every "
            "remaining claim is properly grounded in the numbered evidence with [n] "
            "citations. Keep the rest of the answer intact. Output only the revised answer."
        )

    def _build_citation_retry_prompt(self, user_prompt: str, answer: str) -> str:
        return (
            f"{user_prompt}\n\n---\n"
            f"Your previous answer contained no [n] citation markers:\n{answer}\n\n"
            "Rewrite it so that every factual sentence ends with the [n] marker(s) of the numbered "
            "evidence block(s) that support it, placed immediately before the final period. "
            "Keep the content; do not add facts the evidence does not state. Output only the "
            "rewritten answer."
        )

    def _ensure_substance(self, user_prompt: str, system_prompt: str, answer: str) -> str:
        """A small model sometimes replies with nothing but a citation marker ("[4]"), most often when the evidence is a
        table. Retry twice with a reminder; if it still says nothing, say so honestly instead of returning an empty reply."""
        if has_substance(answer):
            return answer
        for attempt in range(2):
            try:
                revised = strip_regeneration_scaffold(
                    self.ollama_client.generate(
                        f"{user_prompt}\n\n---\nYour previous reply was only {answer.strip()!r}, which does not answer the "
                        "question. Reply again with one or two complete sentences that state the answer in words, using the "
                        "evidence, and end each sentence with its [n] marker.",
                        system=system_prompt,
                        temperature=0.0 if attempt == 0 else 0.4,
                    )
                )
            except Exception:
                break
            if has_substance(revised):
                return revised
        return _NO_ANSWER

    def _ensure_citations(
        self, user_prompt: str, system_prompt: str, answer: str, evidence: list[EvidenceItem]
    ) -> str:
        """If the answer cites nothing, ask once more with a reminder; keep the retry only if it does cite.

        A small model sometimes answers with no [n] markers at all, which leaves every claim unverifiable.
        An honest "the evidence does not specify X" answer has nothing to cite, so it is left alone, and a
        retry that still fails to cite is discarded rather than replacing an answer we already have.
        """
        answer = self._ensure_substance(user_prompt, system_prompt, answer)
        cfg = self.config.get("citation_retry", {})
        attempts = int(cfg.get("max_attempts", 1)) if cfg.get("enabled", True) else 0
        for attempt in range(attempts):
            if not evidence or _has_valid_citation(answer, evidence) or is_abstention(answer):
                break
            try:
                revised = strip_regeneration_scaffold(
                    self.ollama_client.generate(
                        self._build_citation_retry_prompt(user_prompt, answer),
                        system=system_prompt,
                        temperature=0.0 if attempt == 0 else 0.4,
                    )
                )
            except Exception:
                break
            if _has_valid_citation(revised, evidence):
                answer = revised
                break
        return answer

    def _unverified_numbers(self, answer: str, evidence: list[EvidenceItem]) -> list[str]:
        """Every number in the answer that does not appear in the text of a block its own sentence cites.

        Reuses the extractor's sentence/citation split (so "the sentence containing it" and "its citation ids"
        are exactly what the claim verifier already uses) and grounding's number check (which already strips
        citation markers before scanning for numbers, so a marker's own digits are never flagged)."""
        by_id = {e.citation_id: e.text for e in evidence}
        bad: list[str] = []
        for claim in self.claim_extractor.extract(answer, evidence):
            cited = [by_id[i] for i in claim.citation_ids if i in by_id]
            if cited:
                bad.extend(grounding_signals(claim.text, cited).missing_numbers)
        return bad

    def _build_value_retry_prompt(self, user_prompt: str, answer: str, bad_numbers: list[str]) -> str:
        numbers = ", ".join(sorted(set(bad_numbers)))
        return (
            f"{user_prompt}\n\n---\n"
            f"Your previous answer was:\n{answer}\n\n"
            f"The number(s) {numbers} do not appear in the evidence block(s) your answer cites for them — you "
            "likely pulled a value from a different block than the one you cited. Check each cited evidence "
            "block again and give the value exactly as that evidence states it, or say the evidence does not "
            "state it if no cited block contains it. Output only the revised answer."
        )

    def _ensure_values(
        self, user_prompt: str, system_prompt: str, answer: str, evidence: list[EvidenceItem]
    ) -> str:
        """Catch a number stated in the answer that was actually read off a different evidence block than the
        one cited for it. If nothing is unverified, return as-is with no LLM call; otherwise retry once, naming
        the offending number(s), and keep the retry only if it leaves strictly fewer numbers unverified."""
        cfg = self.config.get("value_check", {})
        if not cfg.get("enabled", True) or not evidence:
            return answer
        bad = self._unverified_numbers(answer, evidence)
        if not bad:
            return answer
        try:
            revised = strip_regeneration_scaffold(
                self.ollama_client.generate(
                    self._build_value_retry_prompt(user_prompt, answer, bad),
                    system=system_prompt,
                    temperature=0.0,
                )
            )
        except Exception:
            return answer
        if revised.strip() and len(self._unverified_numbers(revised, evidence)) < len(bad):
            return revised
        return answer

    def _generate_cited(self, user_prompt: str, system_prompt: str, evidence: list[EvidenceItem]) -> str:
        return self._ensure_citations(
            user_prompt, system_prompt, self.ollama_client.generate(user_prompt, system=system_prompt), evidence
        )

    def _annotate_unsupported(self, answer: str, unsupported: list[Claim]) -> str:
        leftover = []
        for claim in unsupported:
            if claim.text and claim.text in answer:
                answer = answer.replace(claim.text, claim.text + _UNSUPPORTED_NOTE, 1)
            else:
                leftover.append(claim.text)
        if leftover:
            bullets = "\n".join(f"- {t}{_UNSUPPORTED_NOTE}" for t in leftover)
            answer = f"{answer}\n\n**Unverified statements:**\n{bullets}"
        return answer

    def _verify_and_regenerate(
        self,
        user_prompt: str,
        system_prompt: str,
        answer: str,
        claims: list[Claim],
        evidence: list[EvidenceItem],
        max_attempts: int,
    ) -> tuple[str, list[Claim]]:
        claims = self.claim_verifier.verify_all(claims, evidence)
        unsupported = [c for c in claims if c.status in _UNSUPPORTED_LIKE]
        attempt = 0
        while unsupported and attempt < max_attempts:
            revision_prompt = self._build_regeneration_prompt(user_prompt, answer, unsupported)
            try:
                revised = strip_regeneration_scaffold(
                    self.ollama_client.generate(revision_prompt, system=system_prompt, temperature=0.0)
                )
            except Exception:
                break
            revised_claims = self.claim_extractor.extract(revised, evidence)
            # A small model often "revises" by dropping every [n] marker. Uncited claims can't be verified,
            # so that turns a partly-grounded answer into a wholly unverifiable one. Keep a revision only if
            # it retains the citations and lowers the unsupported rate; otherwise stop and annotate the
            # answer we already have.
            if _cited_fraction(revised_claims) < _cited_fraction(claims):
                break
            revised_claims = self.claim_verifier.verify_all(revised_claims, evidence)
            revised_unsupported = [c for c in revised_claims if c.status in _UNSUPPORTED_LIKE]
            if _unsupported_rate(revised_claims, revised_unsupported) >= _unsupported_rate(claims, unsupported):
                break
            answer, claims, unsupported = revised, revised_claims, revised_unsupported
            attempt += 1
        if unsupported:
            answer = self._annotate_unsupported(answer, unsupported)
        return answer, claims

    # ------------------------------------------------------------------
    # Core entry point
    # ------------------------------------------------------------------

    def run(
        self,
        query: str,
        mode: RAGMode,
        query_type: QueryType,
        document_ids: Optional[list[str]] = None,
    ) -> QueryResponse:
        t_start = time.perf_counter()
        mode_cfg = self._mode_config(mode)
        use_reranker = bool(mode_cfg.get("use_reranker", False))
        use_verification = bool(mode_cfg.get("use_verification", False))
        query_decomposition = bool(mode_cfg.get("query_decomposition", False))
        max_regen = int(mode_cfg.get("max_regeneration_attempts", 2))

        latency = LatencyBreakdown()

        candidates, retrieval_ms = self._retrieve(query, mode, document_ids, query_decomposition)
        latency.retrieval_latency_ms = retrieval_ms

        candidates, rerank_ms = self._rerank(query, candidates, use_reranker)
        latency.rerank_latency_ms = rerank_ms

        evidence = self._build_evidence(candidates)
        system_prompt, user_prompt = self._build_prompt(query_type, query, evidence)

        t0 = time.perf_counter()
        answer, stats = self.ollama_client.generate_with_stats(user_prompt, system=system_prompt)
        answer = self._ensure_citations(user_prompt, system_prompt, answer, evidence)
        if query_type == QueryType.QUESTION_ANSWERING:
            answer = self._ensure_values(user_prompt, system_prompt, answer, evidence)
        latency.generation_latency_ms = (time.perf_counter() - t0) * 1000.0

        claims = self.claim_extractor.extract(answer, evidence)

        if use_verification:
            t0 = time.perf_counter()
            answer, claims = self._verify_and_regenerate(
                user_prompt, system_prompt, answer, claims, evidence, max_regen
            )
            latency.verification_latency_ms = (time.perf_counter() - t0) * 1000.0

        citation_metrics = compute_citation_metrics(claims)
        latency.total_latency_ms = (time.perf_counter() - t_start) * 1000.0

        return QueryResponse(
            answer_markdown=answer,
            claims=claims,
            evidence=evidence,
            citation_metrics=citation_metrics,
            latency=latency,
            mode=mode,
            num_sources=len(evidence),
            tokens_generated=stats.get("eval_count"),
            tokens_per_second=stats.get("tokens_per_second"),
        )

    # ------------------------------------------------------------------
    # Higher-level flows
    # ------------------------------------------------------------------

    def compare(self, document_ids: list[str], aspect: QueryType, mode: RAGMode) -> QueryResponse:
        question = _COMPARE_QUESTIONS.get(aspect, f"Compare these papers with respect to {aspect.value}.")
        return self.run(query=question, mode=mode, query_type=aspect, document_ids=document_ids)

    def conflict_detection(self, document_ids: Optional[list[str]], topic: str, mode: RAGMode) -> QueryResponse:
        return self.run(
            query=topic, mode=mode, query_type=QueryType.CONFLICT_DETECTION, document_ids=document_ids
        )

    def _cap_by_documents(
        self, candidates: list[tuple[Chunk, float]], max_papers: int
    ) -> list[tuple[Chunk, float]]:
        seen: list[str] = []
        result = []
        for chunk, score in candidates:
            if chunk.document_id not in seen:
                if len(seen) >= max_papers:
                    continue
                seen.append(chunk.document_id)
            result.append((chunk, score))
        return result

    def _cluster_by_section(
        self, evidence: list[EvidenceItem], max_clusters: int = 6
    ) -> dict[str, list[EvidenceItem]]:
        buckets: dict[str, list[EvidenceItem]] = {}
        for item in evidence:
            key = (item.section or "General").strip() or "General"
            buckets.setdefault(key, []).append(item)
        if len(buckets) <= max_clusters:
            return buckets
        sorted_keys = sorted(buckets, key=lambda k: len(buckets[k]), reverse=True)
        kept = sorted_keys[: max_clusters - 1]
        merged = {k: buckets[k] for k in kept}
        other: list[EvidenceItem] = []
        for k in sorted_keys[max_clusters - 1 :]:
            other.extend(buckets[k])
        if other:
            merged["Other"] = other
        return merged

    def literature_review(
        self,
        topic: str,
        document_ids: Optional[list[str]],
        mode: RAGMode,
        max_papers: int = 15,
    ) -> QueryResponse:
        t_start = time.perf_counter()
        mode_cfg = self._mode_config(mode)
        use_reranker = bool(mode_cfg.get("use_reranker", False))
        use_verification = bool(mode_cfg.get("use_verification", False))
        query_decomposition = bool(mode_cfg.get("query_decomposition", False))

        latency = LatencyBreakdown()

        candidates, retrieval_ms = self._retrieve(topic, mode, document_ids, query_decomposition)
        latency.retrieval_latency_ms = retrieval_ms
        candidates = self._cap_by_documents(candidates, max_papers)

        candidates, rerank_ms = self._rerank(topic, candidates, use_reranker)
        latency.rerank_latency_ms = rerank_ms

        evidence = self._build_evidence(candidates)
        clusters = self._cluster_by_section(evidence)

        gen_ms_total = 0.0
        section_texts: dict[str, str] = {}
        system_prompt = None
        for section_name, items in clusters.items():
            system_prompt, user_prompt = build_literature_review_section_prompt(items, subtopic=f"{topic} — {section_name}")
            t0 = time.perf_counter()
            try:
                text = self._generate_cited(user_prompt, system_prompt, evidence)
            except Exception as exc:
                text = f"_Could not generate this subsection: {exc}_"
            gen_ms_total += (time.perf_counter() - t0) * 1000.0
            section_texts[section_name] = text

        comp_system, comp_user = build_comparison_prompt(evidence, aspect=f"key findings related to {topic}")
        t0 = time.perf_counter()
        try:
            comparative_findings = self._generate_cited(comp_user, comp_system, evidence)
        except Exception as exc:
            comparative_findings = f"_Could not generate comparative findings: {exc}_"
        gen_ms_total += (time.perf_counter() - t0) * 1000.0

        lim_system, lim_user = build_qa_prompt(
            f"What limitations are reported in the literature on {topic}?", evidence
        )
        t0 = time.perf_counter()
        try:
            limitations = self._generate_cited(lim_user, lim_system, evidence)
        except Exception as exc:
            limitations = f"_Could not generate limitations: {exc}_"
        gen_ms_total += (time.perf_counter() - t0) * 1000.0

        gaps_system, gaps_user = build_research_gap_prompt(evidence)
        t0 = time.perf_counter()
        try:
            research_gaps = self._generate_cited(gaps_user, gaps_system, evidence)
        except Exception as exc:
            research_gaps = f"_Could not generate research gaps: {exc}_"
        gen_ms_total += (time.perf_counter() - t0) * 1000.0

        bg_system, bg_user = build_qa_prompt(f"Provide a brief background introduction to: {topic}", evidence[:5])
        t0 = time.perf_counter()
        try:
            background = self._generate_cited(bg_user, bg_system, evidence[:5])
        except Exception as exc:
            background = f"_Could not generate background: {exc}_"
        gen_ms_total += (time.perf_counter() - t0) * 1000.0

        latency.generation_latency_ms = gen_ms_total

        full_text = "\n\n".join(
            [background, *section_texts.values(), comparative_findings, limitations, research_gaps]
        )
        claims = self.claim_extractor.extract(full_text, evidence)

        if use_verification:
            t0 = time.perf_counter()
            claims = self.claim_verifier.verify_all(claims, evidence)
            latency.verification_latency_ms = (time.perf_counter() - t0) * 1000.0
            unsupported = [c for c in claims if c.status in _UNSUPPORTED_LIKE]
            if unsupported:
                for c in unsupported:
                    background = background.replace(c.text, c.text + _UNSUPPORTED_NOTE, 1) if c.text in background else background
                    for k in section_texts:
                        if c.text in section_texts[k]:
                            section_texts[k] = section_texts[k].replace(c.text, c.text + _UNSUPPORTED_NOTE, 1)
                    comparative_findings = (
                        comparative_findings.replace(c.text, c.text + _UNSUPPORTED_NOTE, 1)
                        if c.text in comparative_findings
                        else comparative_findings
                    )

        references = "\n".join(
            f"[{e.citation_id}] {e.document_title} — page {e.page_number}, {e.section}" for e in evidence
        )

        parts = [f"# Literature Review: {topic}", "## Background", background]
        for section_name, text in section_texts.items():
            parts.append(f"## {section_name}")
            parts.append(text)
        parts += [
            "## Comparative Findings",
            comparative_findings,
            "## Limitations in Current Literature",
            limitations,
            "## Research Gaps",
            research_gaps,
            "## References",
            references,
        ]
        answer_markdown = "\n\n".join(parts)

        citation_metrics = compute_citation_metrics(claims)
        latency.total_latency_ms = (time.perf_counter() - t_start) * 1000.0

        return QueryResponse(
            answer_markdown=answer_markdown,
            claims=claims,
            evidence=evidence,
            citation_metrics=citation_metrics,
            latency=latency,
            mode=mode,
            num_sources=len(evidence),
            tokens_generated=None,
            tokens_per_second=None,
        )
