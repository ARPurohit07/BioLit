"""Pydantic schemas shared across the BioLit backend API.

This module is the contract between ingestion, retrieval, generation,
verification and the FastAPI routes. Frontend TypeScript types in
frontend/src/types mirror these shapes.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class RAGMode(str, Enum):
    FAST = "fast"
    BALANCED = "balanced"
    HIGH_FAITHFULNESS = "high_faithfulness"


class ClaimStatus(str, Enum):
    NOT_VERIFIED = "NOT_VERIFIED"  # extracted from the answer but never checked (Fast and Balanced modes do not verify)
    SUPPORTED = "SUPPORTED"
    PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
    UNSUPPORTED = "UNSUPPORTED"
    CONTRADICTED = "CONTRADICTED"


class DocumentStatus(str, Enum):
    UPLOADED = "uploaded"
    PARSING = "parsing"
    INDEXING = "indexing"
    INDEXED = "indexed"
    FAILED = "failed"
    SCANNED_NEEDS_OCR = "scanned_needs_ocr"


class QueryType(str, Enum):
    SUMMARIZE = "summarize"
    COMPARE_PAPERS = "compare_papers"
    COMPARE_METHODOLOGY = "compare_methodology"
    COMPARE_RESULTS = "compare_results"
    LIMITATIONS = "limitations"
    RESEARCH_GAPS = "research_gaps"
    LITERATURE_REVIEW = "literature_review"
    QUESTION_ANSWERING = "question_answering"
    STRUCTURED_TABLE = "structured_table"
    CONFLICT_DETECTION = "conflict_detection"


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------

class DocumentMetadata(BaseModel):
    document_id: str
    title: str
    authors: list[str] = Field(default_factory=list)
    year: Optional[int] = None
    source: str = "upload"
    filename: str
    num_pages: int = 0
    num_chunks: int = 0
    status: DocumentStatus = DocumentStatus.UPLOADED
    uploaded_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    error_message: Optional[str] = None


class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    status: DocumentStatus
    message: str


class DocumentListResponse(BaseModel):
    documents: list[DocumentMetadata]
    total: int


# ---------------------------------------------------------------------------
# Chunks / evidence
# ---------------------------------------------------------------------------

class Chunk(BaseModel):
    chunk_id: str
    document_id: str
    page_number: int
    section: str
    text: str
    token_count: int


class EvidenceItem(BaseModel):
    """A single retrieved/cited piece of evidence, resolvable in the Evidence Viewer."""
    citation_id: int
    chunk_id: str
    document_id: str
    document_title: str
    page_number: int
    section: str
    text: str
    score: Optional[float] = None


# ---------------------------------------------------------------------------
# Claims / verification
# ---------------------------------------------------------------------------

class Claim(BaseModel):
    claim_id: str
    text: str
    citation_ids: list[int] = Field(default_factory=list)
    status: ClaimStatus = ClaimStatus.NOT_VERIFIED  # the verifier sets a real status; unverified modes leave this
    verifier_rationale: Optional[str] = None


class CitationMetrics(BaseModel):
    """citation_coverage needs no verifier; the other three are None when no claim was verified ("not measured")."""
    citation_precision: Optional[float] = None
    citation_coverage: float
    faithfulness: Optional[float] = None
    unsupported_claim_rate: Optional[float] = None
    verified_claims: int = 0
    total_claims: int
    total_citations: int


# ---------------------------------------------------------------------------
# Query / answer
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    question: str
    mode: RAGMode = RAGMode.BALANCED
    query_type: QueryType = QueryType.QUESTION_ANSWERING
    document_ids: Optional[list[str]] = None  # restrict retrieval scope; None = all indexed docs
    top_k: Optional[int] = None


class LatencyBreakdown(BaseModel):
    retrieval_latency_ms: float = 0.0
    dense_latency_ms: float = 0.0
    bm25_latency_ms: float = 0.0
    rerank_latency_ms: float = 0.0
    generation_latency_ms: float = 0.0
    verification_latency_ms: float = 0.0
    total_latency_ms: float = 0.0


class QueryResponse(BaseModel):
    answer_markdown: str
    claims: list[Claim]
    evidence: list[EvidenceItem]
    citation_metrics: Optional[CitationMetrics] = None
    latency: LatencyBreakdown
    mode: RAGMode
    num_sources: int
    tokens_generated: Optional[int] = None
    tokens_per_second: Optional[float] = None


class CompareRequest(BaseModel):
    document_ids: list[str]
    aspect: QueryType = QueryType.COMPARE_METHODOLOGY
    mode: RAGMode = RAGMode.BALANCED


class LiteratureReviewRequest(BaseModel):
    topic: str
    document_ids: Optional[list[str]] = None
    mode: RAGMode = RAGMode.HIGH_FAITHFULNESS
    max_papers: int = 15


class VerifyRequest(BaseModel):
    claim_text: str
    evidence_chunk_ids: list[str]


class VerifyResponse(BaseModel):
    status: ClaimStatus
    rationale: str


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

class RetrievalMetrics(BaseModel):
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    recall_at_10: float
    mrr: float
    ndcg_at_10: Optional[float] = None
    num_queries: int


class GenerationMetrics(BaseModel):
    citation_precision: Optional[float] = None
    citation_coverage: float
    faithfulness: Optional[float] = None
    unsupported_claim_rate: Optional[float] = None
    answer_relevance_approx: Optional[float] = None
    note: str = "Semantic relevance metrics are approximate; not a substitute for human review."


class LatencyBenchmarkEntry(BaseModel):
    mode: RAGMode
    retrieval_latency_ms: float
    generation_latency_ms: float
    verification_latency_ms: float
    total_latency_ms: float
    tokens_per_second: Optional[float] = None


class CitationConfigResult(BaseModel):
    """One model/prompt configuration scored on the held-out citation prompts (see scripts/summarize_citation_evals.py)."""
    name: str
    description: str = ""
    n: int
    cites_any: int
    valid_ids: int
    passes_checker: int
    val_passes: int = 0
    val_n: int = 0
    test_passes: int = 0
    test_n: int = 0
    avg_sentence_coverage: float  # 0-1: share of factual sentences that carry an [n] marker
    avg_answer_words: float = 0.0
    avg_citations: float = 0.0


class CitationComparison(BaseModel):
    n: int
    prompts: str = ""
    caveats: list[str] = Field(default_factory=list)
    configs: list[CitationConfigResult] = Field(default_factory=list)


class EvaluationSummary(BaseModel):
    evaluated: bool
    retrieval_metrics: Optional[RetrievalMetrics] = None
    generation_metrics: Optional[GenerationMetrics] = None
    latency_by_mode: list[LatencyBenchmarkEntry] = Field(default_factory=list)
    citation_comparison: Optional[CitationComparison] = None
    variant: str  # e.g. "base_rag", "finetuned_rag", "finetuned_no_rag"
    note: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    ollama_available: bool
    ollama_model: Optional[str] = None
    num_indexed_documents: int
    num_indexed_chunks: int
    reranker_device: Optional[str] = None  # e.g. "cuda (fp16)", "cpu", or "cpu (fell back from GPU: ...)"
