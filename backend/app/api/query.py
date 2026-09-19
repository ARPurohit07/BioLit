"""Query / compare / literature-review / ad-hoc claim verification endpoints."""
from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException, Request

from backend.app.config.settings import get_settings
from backend.app.models.schemas import (
    ClaimStatus,
    CompareRequest,
    EvidenceItem,
    LiteratureReviewRequest,
    QueryRequest,
    QueryResponse,
    VerifyRequest,
    VerifyResponse,
)
from backend.app.verification.claims import Claim

router = APIRouter(prefix="/api", tags=["query"])


def _require_pipeline(request: Request):
    pipeline = getattr(request.app.state, "rag_pipeline", None)
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="RAG pipeline is not available. Check /api/health for startup errors "
            "(likely Ollama is not running or models failed to load).",
        )
    return pipeline


@router.post("/query", response_model=QueryResponse)
def query(body: QueryRequest, request: Request) -> QueryResponse:
    pipeline = _require_pipeline(request)
    try:
        return pipeline.run(
            query=body.question,
            mode=body.mode,
            query_type=body.query_type,
            document_ids=body.document_ids,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Query failed: {exc}")


@router.post("/compare", response_model=QueryResponse)
def compare(body: CompareRequest, request: Request) -> QueryResponse:
    pipeline = _require_pipeline(request)
    try:
        return pipeline.compare(document_ids=body.document_ids, aspect=body.aspect, mode=body.mode)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Compare failed: {exc}")


@router.post("/literature-review", response_model=QueryResponse)
def literature_review(body: LiteratureReviewRequest, request: Request) -> QueryResponse:
    pipeline = _require_pipeline(request)
    try:
        return pipeline.literature_review(
            topic=body.topic,
            document_ids=body.document_ids,
            mode=body.mode,
            max_papers=body.max_papers,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Literature review failed: {exc}")


def _load_chunks_by_id(chunk_ids: list[str]) -> dict[str, dict]:
    """Reads chunk records straight out of the vector store's metadata JSONL file
    (path from configs/retrieval.yaml), independent of any in-memory index API,
    so this endpoint works as long as the file exists."""
    settings = get_settings()
    rel_path = settings.retrieval_config.get("vector_store", {}).get("metadata_path", "data/index/chunk_metadata.jsonl")
    path = settings.repo_root / rel_path
    wanted = set(chunk_ids)
    found: dict[str, dict] = {}
    if not path.exists():
        return found
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            cid = record.get("chunk_id")
            if cid in wanted:
                found[cid] = record
                if len(found) == len(wanted):
                    break
    return found


@router.post("/verify", response_model=VerifyResponse)
def verify(body: VerifyRequest, request: Request) -> VerifyResponse:
    verifier = getattr(request.app.state, "claim_verifier", None)
    if verifier is None:
        raise HTTPException(status_code=503, detail="Claim verifier is not available.")

    chunk_records = _load_chunks_by_id(body.evidence_chunk_ids)
    if not chunk_records:
        return VerifyResponse(
            status=ClaimStatus.UNSUPPORTED,
            rationale="None of the given evidence_chunk_ids could be resolved to indexed chunk text.",
        )

    evidence = []
    for i, chunk_id in enumerate(body.evidence_chunk_ids, start=1):
        record = chunk_records.get(chunk_id)
        if not record:
            continue
        evidence.append(
            EvidenceItem(
                citation_id=i,
                chunk_id=chunk_id,
                document_id=record.get("document_id", ""),
                document_title=record.get("document_id", ""),
                page_number=record.get("page_number", 0),
                section=record.get("section", ""),
                text=record.get("text", ""),
            )
        )

    claim = Claim(claim_id="adhoc", text=body.claim_text, citation_ids=[e.citation_id for e in evidence])
    verified = verifier.verify(claim, evidence)
    return VerifyResponse(status=verified.status, rationale=verified.verifier_rationale or "")
