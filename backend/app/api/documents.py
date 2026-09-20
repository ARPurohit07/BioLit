"""Document upload / listing / deletion / indexing endpoints.

Indexing (POST /index) imports the ingestion package lazily so this module
can still be imported (and the rest of the API served) even before that
package's files exist. Shared retrieval-side objects (vector_store, bm25_index,
embedding_model) are read off app.state, where main.py constructs them once
at startup.

NOTE: FAISSVectorStore/BM25Index's exact "add chunks" / "save" method names
weren't part of the frozen interface this module was written against (only
__init__/load/__len__/remove_document were specified). We call `.add(...)`
and `.save()` defensively and catch any mismatch per-document so a signature
difference in the concurrently-built retrieval package fails one document's
indexing rather than the whole batch/endpoint.
"""
from __future__ import annotations

import uuid
from typing import Optional

from fastapi import APIRouter, HTTPException, Request, UploadFile
from pydantic import BaseModel

from backend.app import db
from backend.app.config.settings import get_settings
from backend.app.models.schemas import (
    DocumentListResponse,
    DocumentMetadata,
    DocumentStatus,
    DocumentUploadResponse,
)

router = APIRouter(prefix="/api/documents", tags=["documents"])


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile) -> DocumentUploadResponse:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only .pdf files are accepted.")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    settings = get_settings()
    document_id = uuid.uuid4().hex
    dest = settings.documents_dir / f"{document_id}.pdf"
    dest.write_bytes(content)

    meta = DocumentMetadata(
        document_id=document_id,
        title=file.filename.rsplit(".", 1)[0],
        filename=file.filename,
        status=DocumentStatus.UPLOADED,
    )
    db.upsert_document(meta)

    return DocumentUploadResponse(
        document_id=document_id,
        filename=file.filename,
        status=DocumentStatus.UPLOADED,
        message="Uploaded. Call POST /api/documents/index to build the search index.",
    )


@router.get("/", response_model=DocumentListResponse)
def list_documents() -> DocumentListResponse:
    docs = db.list_documents()
    return DocumentListResponse(documents=docs, total=len(docs))


@router.delete("/{document_id}")
def delete_document(document_id: str, request: Request) -> dict:
    meta = db.get_document(document_id)
    if meta is None:
        raise HTTPException(status_code=404, detail="Document not found.")

    settings = get_settings()
    pdf_path = settings.documents_dir / f"{document_id}.pdf"
    if pdf_path.exists():
        pdf_path.unlink()

    vector_store = getattr(request.app.state, "vector_store", None)
    bm25_index = getattr(request.app.state, "bm25_index", None)
    for index in (vector_store, bm25_index):
        if index is not None:
            try:
                index.remove_document(document_id)
            except Exception:
                pass
    if vector_store is not None:
        try:
            vector_store.save()  # type: ignore[attr-defined]
        except Exception:
            pass
    if bm25_index is not None:
        try:
            bm25_index.save()  # type: ignore[attr-defined]
        except Exception:
            pass

    db.delete_document(document_id)
    return {"document_id": document_id, "deleted": True}


class IndexRequest(BaseModel):
    document_ids: Optional[list[str]] = None


class IndexResult(BaseModel):
    document_id: str
    status: DocumentStatus
    message: str


class IndexResponse(BaseModel):
    results: list[IndexResult]


def _index_one_document(request: Request, meta: DocumentMetadata) -> IndexResult:
    from backend.app.ingestion.chunker import PageAwareChunker
    from backend.app.ingestion.metadata_extractor import MetadataExtractor
    from backend.app.ingestion.pdf_loader import PDFLoader
    from backend.app.ingestion.section_detector import SectionDetector

    settings = get_settings()
    pdf_path = settings.documents_dir / f"{meta.document_id}.pdf"
    if not pdf_path.exists():
        meta.status = DocumentStatus.FAILED
        meta.error_message = "PDF file missing on disk."
        db.upsert_document(meta)
        return IndexResult(document_id=meta.document_id, status=meta.status, message=meta.error_message)

    loader = PDFLoader()
    if loader.is_scanned(str(pdf_path)):
        meta.status = DocumentStatus.SCANNED_NEEDS_OCR
        meta.error_message = "Scanned PDF (no extractable text layer); OCR not yet supported."
        db.upsert_document(meta)
        return IndexResult(document_id=meta.document_id, status=meta.status, message=meta.error_message)

    # Structure-aware: tables and figures become their own chunks (same as scripts/ingest.py).
    pages = loader.load_structured(
        str(pdf_path), meta.document_id, settings.repo_root / "data" / "figures", settings.repo_root
    )

    extractor = MetadataExtractor()
    extracted = extractor.extract(str(pdf_path), pages)
    if extracted.get("title"):
        meta.title = extracted["title"]
    if extracted.get("authors"):
        meta.authors = extracted["authors"]
    if extracted.get("year"):
        meta.year = extracted["year"]
    meta.num_pages = len(pages)

    chunk_cfg = settings.retrieval_config.get("chunking", {})
    chunker = PageAwareChunker(
        chunk_size=chunk_cfg.get("chunk_size", 400),
        chunk_overlap=chunk_cfg.get("chunk_overlap", 60),
        min_chunk_tokens=chunk_cfg.get("min_chunk_tokens", 40),
    )
    section_detector = SectionDetector()
    chunks = chunker.chunk_document(meta.document_id, pages, section_detector)

    embedding_model = getattr(request.app.state, "embedding_model", None)
    vector_store = getattr(request.app.state, "vector_store", None)
    bm25_index = getattr(request.app.state, "bm25_index", None)
    if embedding_model is None or vector_store is None or bm25_index is None:
        meta.status = DocumentStatus.FAILED
        meta.error_message = "Embedding model / vector store / BM25 index not available (startup failed)."
        db.upsert_document(meta)
        return IndexResult(document_id=meta.document_id, status=meta.status, message=meta.error_message)

    # Bibliography entries are dense with on-topic keywords (cited paper titles) and
    # routinely outrank real evidence in retrieval, but a reference-list line is never
    # itself usable evidence for a claim — exclude from the search index (they're still
    # counted in meta.num_chunks / kept in the chunk record for provenance).
    # Text only: a table or figure after the bibliography is an appendix item, not a citation.
    indexable_chunks = [c for c in chunks if not (c.section == "References" and c.chunk_type == "text")]
    texts = [c.text for c in indexable_chunks]
    vectors = embedding_model.encode(texts) if texts else None

    if indexable_chunks:
        vector_store.add(indexable_chunks, vectors)  # type: ignore[attr-defined]
        bm25_index.add(indexable_chunks)  # type: ignore[attr-defined]

    meta.num_chunks = len(chunks)
    meta.status = DocumentStatus.INDEXED
    meta.error_message = None
    db.upsert_document(meta)
    return IndexResult(document_id=meta.document_id, status=meta.status, message=f"Indexed {len(chunks)} chunks.")


@router.post("/index", response_model=IndexResponse)
def index_documents(body: IndexRequest, request: Request) -> IndexResponse:
    if body.document_ids:
        metas = [db.get_document(doc_id) for doc_id in body.document_ids]
        metas = [m for m in metas if m is not None]
    else:
        metas = [m for m in db.list_documents() if m.status == DocumentStatus.UPLOADED]

    results: list[IndexResult] = []
    for meta in metas:
        try:
            results.append(_index_one_document(request, meta))
        except Exception as exc:
            meta.status = DocumentStatus.FAILED
            meta.error_message = f"Indexing failed: {exc}"
            db.upsert_document(meta)
            results.append(IndexResult(document_id=meta.document_id, status=meta.status, message=meta.error_message))

    vector_store = getattr(request.app.state, "vector_store", None)
    bm25_index = getattr(request.app.state, "bm25_index", None)
    for index in (vector_store, bm25_index):
        if index is not None:
            try:
                index.save()  # type: ignore[attr-defined]
            except Exception:
                pass

    return IndexResponse(results=results)
