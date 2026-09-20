// TypeScript mirror of backend/app/models/schemas.py. Keep in sync by hand.

export type RAGMode = "fast" | "balanced" | "high_faithfulness";

export type ClaimStatus =
  | "NOT_VERIFIED"
  | "SUPPORTED"
  | "PARTIALLY_SUPPORTED"
  | "UNSUPPORTED"
  | "CONTRADICTED";

export type DocumentStatus =
  | "uploaded"
  | "parsing"
  | "indexing"
  | "indexed"
  | "failed"
  | "scanned_needs_ocr";

export type QueryType =
  | "summarize"
  | "compare_papers"
  | "compare_methodology"
  | "compare_results"
  | "limitations"
  | "research_gaps"
  | "literature_review"
  | "question_answering"
  | "structured_table"
  | "conflict_detection";

// ---------------------------------------------------------------------------
// Documents
// ---------------------------------------------------------------------------

export interface DocumentMetadata {
  document_id: string;
  title: string;
  authors: string[];
  year?: number;
  source: string;
  filename: string;
  num_pages: number;
  num_chunks: number;
  status: DocumentStatus;
  uploaded_at: string;
  error_message?: string;
}

export interface DocumentUploadResponse {
  document_id: string;
  filename: string;
  status: DocumentStatus;
  message: string;
}

export interface DocumentListResponse {
  documents: DocumentMetadata[];
  total: number;
}

export interface DocumentIndexResult {
  document_id: string;
  status: DocumentStatus;
  message: string;
}

export interface DocumentIndexResponse {
  results: DocumentIndexResult[];
}

// ---------------------------------------------------------------------------
// Chunks / evidence
// ---------------------------------------------------------------------------

export interface Chunk {
  chunk_id: string;
  document_id: string;
  page_number: number;
  section: string;
  text: string;
  token_count: number;
}

export interface EvidenceItem {
  citation_id: number;
  chunk_id: string;
  document_id: string;
  document_title: string;
  page_number: number;
  section: string;
  text: string;
  score?: number;
}

// ---------------------------------------------------------------------------
// Claims / verification
// ---------------------------------------------------------------------------

export interface Claim {
  claim_id: string;
  text: string;
  citation_ids: number[];
  status: ClaimStatus;
  verifier_rationale?: string;
  /** Fast LLM-free text-overlap heuristic; only "none" is shown (a check-this flag, not a verdict). */
  grounding?: "strong" | "weak" | "none" | null;
  grounding_score?: number | null;
}

export interface CitationMetrics {
  /** null = not measured (no claim was verified: Fast and Balanced modes). */
  citation_precision: number | null;
  citation_coverage: number;
  faithfulness: number | null;
  unsupported_claim_rate: number | null;
  verified_claims: number;
  /** Claims the heuristic flags for a manual check; not a faithfulness measure. */
  flagged_claims?: number;
  total_claims: number;
  total_citations: number;
}

// ---------------------------------------------------------------------------
// Query / answer
// ---------------------------------------------------------------------------

export interface QueryRequest {
  question: string;
  mode: RAGMode;
  query_type: QueryType;
  document_ids?: string[];
  top_k?: number;
}

export interface LatencyBreakdown {
  retrieval_latency_ms: number;
  dense_latency_ms: number;
  bm25_latency_ms: number;
  rerank_latency_ms: number;
  generation_latency_ms: number;
  verification_latency_ms: number;
  total_latency_ms: number;
}

export interface QueryResponse {
  answer_markdown: string;
  claims: Claim[];
  evidence: EvidenceItem[];
  citation_metrics?: CitationMetrics;
  latency: LatencyBreakdown;
  mode: RAGMode;
  num_sources: number;
  tokens_generated?: number;
  tokens_per_second?: number;
}

export interface CompareRequest {
  document_ids: string[];
  aspect: QueryType;
  mode: RAGMode;
}

export interface LiteratureReviewRequest {
  topic: string;
  document_ids?: string[];
  mode: RAGMode;
  max_papers: number;
}

export interface VerifyRequest {
  claim_text: string;
  evidence_chunk_ids: string[];
}

export interface VerifyResponse {
  status: ClaimStatus;
  rationale: string;
}

// ---------------------------------------------------------------------------
// Evaluation
// ---------------------------------------------------------------------------

export interface RetrievalMetrics {
  recall_at_1: number;
  recall_at_3: number;
  recall_at_5: number;
  recall_at_10: number;
  mrr: number;
  ndcg_at_10?: number;
  num_queries: number;
}

export interface GenerationMetrics {
  citation_precision: number | null;
  citation_coverage: number;
  faithfulness: number | null;
  unsupported_claim_rate: number | null;
  answer_relevance_approx?: number;
  note: string;
}

export interface LatencyBenchmarkEntry {
  mode: RAGMode;
  retrieval_latency_ms: number;
  generation_latency_ms: number;
  verification_latency_ms: number;
  total_latency_ms: number;
  tokens_per_second?: number;
}

export interface CitationConfigResult {
  name: string;
  description: string;
  n: number;
  cites_any: number;
  valid_ids: number;
  passes_checker: number;
  val_passes: number;
  val_n: number;
  test_passes: number;
  test_n: number;
  /** 0-1: share of factual sentences that carry an [n] marker. */
  avg_sentence_coverage: number;
  avg_answer_words: number;
  avg_citations: number;
}

export interface CitationComparison {
  n: number;
  prompts: string;
  caveats: string[];
  configs: CitationConfigResult[];
}

export interface EvaluationSummary {
  evaluated: boolean;
  retrieval_metrics?: RetrievalMetrics;
  generation_metrics?: GenerationMetrics;
  latency_by_mode: LatencyBenchmarkEntry[];
  citation_comparison?: CitationComparison;
  variant: string;
  note?: string;
}

export interface HealthResponse {
  status: string;
  ollama_available: boolean;
  ollama_model?: string;
  num_indexed_documents: number;
  num_indexed_chunks: number;
  /** Where the reranker runs, e.g. "cuda (fp16)" or "cpu (fell back from GPU: ...)". */
  reranker_device?: string;
}
