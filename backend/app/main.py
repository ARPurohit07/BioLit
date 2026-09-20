"""FastAPI application entrypoint.

Startup is defensive: any heavy model/index load (torch, faiss, Ollama) that
fails is caught and recorded in app.state.startup_errors rather than crashing
the process, so the API can still come up (with degraded functionality) even
before all dependencies/models are installed/pulled or before indexes exist.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.app import db
from backend.app.api.documents import router as documents_router
from backend.app.api.evaluation import router as evaluation_router
from backend.app.api.query import router as query_router
from backend.app.config.settings import get_settings
from backend.app.generation.ollama_client import OllamaClient
from backend.app.generation.rag_pipeline import RAGPipeline
from backend.app.models.schemas import HealthResponse
from backend.app.verification.claims import ClaimExtractor
from backend.app.verification.verifier import ClaimVerifier

logger = logging.getLogger("biolit")


def _init_state(app: FastAPI) -> None:
    settings = get_settings()
    errors: list[str] = []

    app.state.settings = settings
    app.state.vector_store = None
    app.state.bm25_index = None
    app.state.embedding_model = None
    app.state.reranker = None
    app.state.ollama_client = None
    app.state.retriever = None
    app.state.claim_extractor = None
    app.state.claim_verifier = None
    app.state.rag_pipeline = None

    try:
        db.init_db()
    except Exception as exc:
        errors.append(f"db.init_db failed: {exc}")

    try:
        app.state.ollama_client = OllamaClient(
            host=settings.ollama_host, model=settings.ollama_model,
            timeout_s=settings.models_config.get("ollama", {}).get("request_timeout_s", 120),
        )
    except Exception as exc:
        errors.append(f"OllamaClient init failed: {exc}")

    try:
        from backend.app.retrieval.embeddings import EmbeddingModel

        emb_cfg = settings.models_config.get("embedding_model", {})
        app.state.embedding_model = EmbeddingModel(
            model_name=emb_cfg.get("name", "BAAI/bge-small-en-v1.5"),
            device=emb_cfg.get("device", "auto"),
            normalize=emb_cfg.get("normalize", True),
        )
    except Exception as exc:
        errors.append(f"EmbeddingModel init failed: {exc}")

    try:
        from backend.app.retrieval.vector_store import FAISSVectorStore

        vs_cfg = settings.retrieval_config.get("vector_store", {})
        emb_dim = settings.models_config.get("embedding_model", {}).get("dim", 384)
        vector_store = FAISSVectorStore(
            dim=emb_dim,
            index_path=str(settings.repo_root / vs_cfg.get("index_path", "data/index/faiss.index")),
            metadata_path=str(settings.repo_root / vs_cfg.get("metadata_path", "data/index/chunk_metadata.jsonl")),
        )
        vector_store.load()
        app.state.vector_store = vector_store
    except Exception as exc:
        errors.append(f"FAISSVectorStore init/load failed: {exc}")

    try:
        from backend.app.retrieval.bm25 import BM25Index

        bm25_cfg = settings.retrieval_config.get("bm25", {})
        bm25_index = BM25Index(
            index_path=str(settings.repo_root / bm25_cfg.get("index_path", "data/index/bm25_index.pkl")),
            k1=bm25_cfg.get("k1", 1.5),
            b=bm25_cfg.get("b", 0.75),
        )
        bm25_index.load()
        app.state.bm25_index = bm25_index
    except Exception as exc:
        errors.append(f"BM25Index init/load failed: {exc}")

    try:
        from backend.app.retrieval.reranker import Reranker

        rr_cfg = settings.models_config.get("reranker", {})
        app.state.reranker = Reranker(
            model_name=rr_cfg.get("name", "BAAI/bge-reranker-base"),
            device=rr_cfg.get("device", "auto"),
            max_length=rr_cfg.get("max_length", 512),
            batch_size=rr_cfg.get("batch_size", 16),
            half_precision=rr_cfg.get("half_precision", True),
            fallback_to_cpu=rr_cfg.get("fallback_to_cpu", True),
        )
    except Exception as exc:
        errors.append(f"Reranker init failed: {exc}")

    try:
        from backend.app.retrieval.hybrid import HybridRetriever

        if app.state.vector_store is not None and app.state.bm25_index is not None and app.state.embedding_model is not None:
            app.state.retriever = HybridRetriever(
                vector_store=app.state.vector_store,
                bm25_index=app.state.bm25_index,
                embedding_model=app.state.embedding_model,
                config=settings.retrieval_config,
            )
        else:
            errors.append("HybridRetriever not constructed: a dependency (vector_store/bm25_index/embedding_model) failed to load.")
    except Exception as exc:
        errors.append(f"HybridRetriever init failed: {exc}")

    try:
        app.state.claim_extractor = ClaimExtractor()
        if app.state.ollama_client is not None:
            app.state.claim_verifier = ClaimVerifier(app.state.ollama_client)
    except Exception as exc:
        errors.append(f"Claim extractor/verifier init failed: {exc}")

    try:
        if app.state.retriever is not None and app.state.ollama_client is not None and app.state.claim_extractor and app.state.claim_verifier:
            app.state.rag_pipeline = RAGPipeline(
                retriever=app.state.retriever,
                reranker=app.state.reranker,
                ollama_client=app.state.ollama_client,
                claim_extractor=app.state.claim_extractor,
                claim_verifier=app.state.claim_verifier,
                config=settings.retrieval_config,
            )
        else:
            errors.append("RAGPipeline not constructed: one or more dependencies failed to load.")
    except Exception as exc:
        errors.append(f"RAGPipeline init failed: {exc}")

    app.state.startup_errors = errors
    for e in errors:
        logger.warning("BioLit startup: %s", e)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _init_state(app)
    yield


app = FastAPI(title="BioLit", description="Local, privacy-preserving biomedical literature RAG system.", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents_router)
app.include_router(query_router)
app.include_router(evaluation_router)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    settings = get_settings()
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    if settings.debug:
        return JSONResponse(status_code=500, content={"error": "internal_error", "detail": str(exc)})
    return JSONResponse(status_code=500, content={"error": "internal_error", "detail": "An unexpected error occurred."})


@app.get("/api/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    settings = get_settings()
    ollama_client = getattr(request.app.state, "ollama_client", None)
    ollama_available = ollama_client.is_available() if ollama_client else False

    vector_store = getattr(request.app.state, "vector_store", None)
    bm25_index = getattr(request.app.state, "bm25_index", None)
    num_chunks = 0
    if vector_store is not None:
        try:
            num_chunks = len(vector_store)
        except Exception:
            num_chunks = 0

    num_docs = 0
    try:
        num_docs = len(db.list_documents())
    except Exception:
        num_docs = 0

    status = "ok" if ollama_available and getattr(request.app.state, "rag_pipeline", None) is not None else "degraded"

    reranker = getattr(request.app.state, "reranker", None)

    return HealthResponse(
        status=status,
        reranker_device=reranker.describe() if reranker is not None and hasattr(reranker, "describe") else None,
        ollama_available=ollama_available,
        ollama_model=settings.ollama_model,
        num_indexed_documents=num_docs,
        num_indexed_chunks=num_chunks,
    )
