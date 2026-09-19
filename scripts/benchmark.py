"""CLI smoke-test / latency-and-citation benchmark across the three RAG modes.

Runs a small set of example questions (from data/test/eval_questions.jsonl if
present, else a couple of built-in biomedical-style questions used purely as
pipeline smoke-test inputs — NOT fabricated ground truth) through FAST,
BALANCED and HIGH_FAITHFULNESS modes, and writes results to
data/results/benchmark_<timestamp>.json and experiments/rag_modes/latest.json.

If Ollama or the indexes aren't available, the run records that it could not
complete rather than crashing or inventing numbers.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.app import db  # noqa: E402
from backend.app.config.settings import get_settings  # noqa: E402
from backend.app.evaluation.citation_metrics import average_citation_metrics  # noqa: E402
from backend.app.generation.ollama_client import OllamaClient  # noqa: E402
from backend.app.generation.rag_pipeline import RAGPipeline  # noqa: E402
from backend.app.models.schemas import QueryType, RAGMode  # noqa: E402
from backend.app.verification.claims import ClaimExtractor  # noqa: E402
from backend.app.verification.verifier import ClaimVerifier  # noqa: E402

BUILT_IN_QUESTIONS = [
    "What are common methodologies used to study treatment efficacy in clinical trials?",
    "What limitations do researchers typically report when studying biomarkers?",
    "Summarize typical statistical approaches used to analyze patient outcome data.",
]


def _load_questions() -> list[str]:
    path = REPO_ROOT / "data" / "test" / "eval_questions.jsonl"
    if not path.exists():
        return BUILT_IN_QUESTIONS
    questions = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            q = record.get("question") or record.get("query")
            if q:
                questions.append(q)
    return questions or BUILT_IN_QUESTIONS


def _build_pipeline():
    settings = get_settings()
    ollama_client = OllamaClient(host=settings.ollama_host, model=settings.ollama_model)

    from backend.app.retrieval.bm25 import BM25Index
    from backend.app.retrieval.embeddings import EmbeddingModel
    from backend.app.retrieval.hybrid import HybridRetriever
    from backend.app.retrieval.reranker import Reranker
    from backend.app.retrieval.vector_store import FAISSVectorStore

    vs_cfg = settings.retrieval_config.get("vector_store", {})
    emb_cfg = settings.models_config.get("embedding_model", {})
    bm25_cfg = settings.retrieval_config.get("bm25", {})
    rr_cfg = settings.models_config.get("reranker", {})

    embedding_model = EmbeddingModel(model_name=emb_cfg.get("name", "BAAI/bge-small-en-v1.5"))
    vector_store = FAISSVectorStore(
        dim=emb_cfg.get("dim", 384),
        index_path=str(REPO_ROOT / vs_cfg.get("index_path", "data/index/faiss.index")),
        metadata_path=str(REPO_ROOT / vs_cfg.get("metadata_path", "data/index/chunk_metadata.jsonl")),
    )
    vector_store.load()
    bm25_index = BM25Index(
        index_path=str(REPO_ROOT / bm25_cfg.get("index_path", "data/index/bm25_index.pkl")),
        k1=bm25_cfg.get("k1", 1.5), b=bm25_cfg.get("b", 0.75),
    )
    bm25_index.load()
    reranker = Reranker(model_name=rr_cfg.get("name", "BAAI/bge-reranker-base"))
    retriever = HybridRetriever(vector_store, bm25_index, embedding_model, settings.retrieval_config)

    claim_extractor = ClaimExtractor()
    claim_verifier = ClaimVerifier(ollama_client)

    pipeline = RAGPipeline(retriever, reranker, ollama_client, claim_extractor, claim_verifier, settings.retrieval_config)
    return pipeline, ollama_client


def main() -> None:
    settings = get_settings()
    db.init_db()
    questions = _load_questions()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    result: dict = {
        "variant": "rag_modes",
        "timestamp": timestamp,
        "questions": questions,
        "completed": False,
        "note": None,
        "latency_by_mode": [],
        "generation_metrics": None,
    }

    try:
        pipeline, ollama_client = _build_pipeline()
    except Exception as exc:
        result["note"] = f"Could not build RAG pipeline (likely missing indexes/models): {exc}"
        _write_results(result, timestamp)
        print(result["note"])
        return

    if not ollama_client.is_available():
        result["note"] = "Ollama server is not available at the configured host; benchmark could not run."
        _write_results(result, timestamp)
        print(result["note"])
        return

    modes = [RAGMode.FAST, RAGMode.BALANCED, RAGMode.HIGH_FAITHFULNESS]
    responses_by_mode: dict[str, list] = {m.value: [] for m in modes}

    for mode in modes:
        for question in questions:
            try:
                response = pipeline.run(query=question, mode=mode, query_type=QueryType.QUESTION_ANSWERING)
                responses_by_mode[mode.value].append(response)
                result["latency_by_mode"].append(
                    {
                        "mode": mode.value,
                        "retrieval_latency_ms": response.latency.retrieval_latency_ms,
                        "generation_latency_ms": response.latency.generation_latency_ms,
                        "verification_latency_ms": response.latency.verification_latency_ms,
                        "total_latency_ms": response.latency.total_latency_ms,
                        "tokens_per_second": response.tokens_per_second,
                    }
                )
            except Exception as exc:
                result["latency_by_mode"].append({"mode": mode.value, "error": str(exc)})

    all_responses = [r for rs in responses_by_mode.values() for r in rs]
    if all_responses:
        # Precision / faithfulness only mean something for claims that were actually verified. Fast and Balanced
        # never verify, so their claims keep the default UNSUPPORTED status; averaging them in would drag the
        # figures toward 0%. Report these from High-Faithfulness responses only (coverage there is still real).
        verified = responses_by_mode[RAGMode.HIGH_FAITHFULNESS.value]
        gen_metrics = average_citation_metrics(verified or all_responses)
        scope = "High-Faithfulness responses only" if verified else "all modes (no verified responses)"
        result["generation_metrics"] = {
            "citation_precision": gen_metrics.citation_precision,
            "citation_coverage": gen_metrics.citation_coverage,
            "faithfulness": gen_metrics.faithfulness,
            "unsupported_claim_rate": gen_metrics.unsupported_claim_rate,
            "answer_relevance_approx": None,
            "note": f"Automated/approximate metrics from scripts/benchmark.py ({scope}); not a substitute for human review.",
        }
        result["completed"] = True
    else:
        result["note"] = "No responses were generated successfully across any mode."

    _write_results(result, timestamp)
    print(f"Benchmark {'completed' if result['completed'] else 'did not complete'}. "
          f"Results written to data/results/benchmark_{timestamp}.json and experiments/rag_modes/latest.json")


def _write_results(result: dict, timestamp: str) -> None:
    settings = get_settings()
    results_dir = settings.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"benchmark_{timestamp}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)

    latest_dir = settings.repo_root / "experiments" / "rag_modes"
    latest_dir.mkdir(parents=True, exist_ok=True)
    with open(latest_dir / "latest.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)


if __name__ == "__main__":
    main()
