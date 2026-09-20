"""CLI: build the FAISS dense index and BM25 sparse index from data/processed/*.json.

Usage: python scripts/build_index.py --processed data/processed
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.app import db  # noqa: E402
from backend.app.config.settings import get_settings  # noqa: E402
from backend.app.models.schemas import Chunk, DocumentMetadata, DocumentStatus  # noqa: E402
from backend.app.retrieval.bm25 import BM25Index  # noqa: E402
from backend.app.retrieval.embeddings import EmbeddingModel  # noqa: E402
from backend.app.retrieval.vector_store import FAISSVectorStore  # noqa: E402


def load_chunks(processed_dir: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for json_path in sorted(processed_dir.glob("*.json")):
        with open(json_path, "r", encoding="utf-8") as f:
            record = json.load(f)
        if record.get("status") == "scanned_needs_ocr":
            continue
        for c in record.get("chunks", []):
            # Bibliography entries are dense with on-topic keywords (cited paper titles)
            # and routinely outrank real evidence in retrieval, but a reference-list line
            # is never itself usable evidence for a claim — exclude from the search index.
            # They stay in data/processed/*.json for provenance/completeness.
            # Text only: a table or figure that comes after the bibliography is an appendix item, not a citation.
            if c.get("section") == "References" and c.get("chunk_type", "text") == "text":
                continue
            chunks.append(Chunk(**c))
    return chunks


def register_documents(processed_dir: Path) -> int:
    """Upsert one SQLite `documents` row per processed paper so the API/UI list them.

    The CLI pipeline (ingest -> build_index) otherwise only writes data/processed and the
    search indexes, leaving GET /api/documents and /api/health reporting zero documents.
    Returns the number of documents registered.
    """
    count = 0
    for json_path in sorted(processed_dir.glob("*.json")):
        with open(json_path, "r", encoding="utf-8") as f:
            record = json.load(f)

        status = DocumentStatus(record.get("status", DocumentStatus.INDEXED.value))
        meta = DocumentMetadata(
            document_id=record["document_id"],
            title=record.get("title") or record.get("filename") or record["document_id"],
            authors=record.get("authors") or [],
            year=record.get("year"),
            source=record.get("source", "upload"),
            filename=record.get("filename") or json_path.stem,
            num_pages=record.get("num_pages", 0),
            num_chunks=len(record.get("chunks", [])),
            status=status,
            error_message=(
                "Scanned PDF (no extractable text layer); OCR not yet supported."
                if status == DocumentStatus.SCANNED_NEEDS_OCR else None
            ),
        )
        existing = db.get_document(meta.document_id)
        if existing is not None:
            meta.uploaded_at = existing.uploaded_at  # re-running shouldn't reshuffle the list
        db.upsert_document(meta)
        count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FAISS + BM25 indexes from processed chunk JSON files.")
    parser.add_argument("--processed", default="data/processed")
    parser.add_argument(
        "--register-only", action="store_true",
        help="Only register the processed papers in the documents database; don't re-embed or rebuild the indexes.",
    )
    args = parser.parse_args()

    if args.register_only:
        n = register_documents(Path(args.processed))
        print(f"Registered {n} documents in {get_settings().sqlite_path}")
        return

    settings = get_settings()
    retrieval_cfg = settings.retrieval_config
    models_cfg = settings.models_config

    processed_dir = Path(args.processed)
    chunks = load_chunks(processed_dir)
    print(f"Loaded {len(chunks)} chunks from {processed_dir}")
    if not chunks:
        print("Nothing to index.")
        register_documents(processed_dir)  # still surfaces e.g. scanned-PDF records in the UI
        return

    embedding_cfg = models_cfg.get("embedding_model", {})
    embed_model = EmbeddingModel(
        model_name=embedding_cfg.get("name", "BAAI/bge-small-en-v1.5"),
        device=embedding_cfg.get("device", "auto"),
        normalize=embedding_cfg.get("normalize", True),
    )

    vector_cfg = retrieval_cfg.get("vector_store", {})
    bm25_cfg = retrieval_cfg.get("bm25", {})

    vector_store = FAISSVectorStore(
        dim=embedding_cfg.get("dim", 384),
        index_path=str(settings.repo_root / vector_cfg.get("index_path", "data/index/faiss.index")),
        metadata_path=str(settings.repo_root / vector_cfg.get("metadata_path", "data/index/chunk_metadata.jsonl")),
    )
    bm25_index = BM25Index(
        index_path=str(settings.repo_root / bm25_cfg.get("index_path", "data/index/bm25_index.pkl")),
        k1=bm25_cfg.get("k1", 1.5),
        b=bm25_cfg.get("b", 0.75),
    )

    t0 = time.time()
    texts = [c.text for c in chunks]
    batch_size = embedding_cfg.get("batch_size", 32)
    embeddings = embed_model.encode(texts, batch_size=batch_size)
    embed_time = time.time() - t0
    print(f"Encoded {len(texts)} chunks in {embed_time:.2f}s")

    vector_store.add(chunks, embeddings)
    bm25_index.add(chunks)

    vector_store.save()
    bm25_index.save()

    # Only after the indexes are saved, so a failed build never leaves papers marked "indexed".
    n_registered = register_documents(processed_dir)

    total_time = time.time() - t0
    print("\n=== Index build summary ===")
    print(f"{'chunks indexed':<20}{len(chunks)}")
    print(f"{'dense vectors':<20}{len(vector_store)}")
    print(f"{'bm25 docs':<20}{len(bm25_index)}")
    print(f"{'documents registered':<20}{n_registered}")
    print(f"{'embed time (s)':<20}{embed_time:.2f}")
    print(f"{'total time (s)':<20}{total_time:.2f}")


if __name__ == "__main__":
    main()
