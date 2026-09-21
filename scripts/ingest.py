"""CLI: parse PDFs into page-aware chunk JSON records.

Usage: python scripts/ingest.py --input data/raw --output data/processed
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.app.config.settings import get_settings  # noqa: E402
from backend.app.ingestion.chunker import PageAwareChunker  # noqa: E402
from backend.app.ingestion.metadata_extractor import MetadataExtractor  # noqa: E402
from backend.app.ingestion.pdf_loader import PDFLoader  # noqa: E402
from backend.app.ingestion.section_detector import SectionDetector  # noqa: E402


def _slugify(stem: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", stem.lower()).strip("-")
    return slug or "document"


USE_TABLE_CACHE = bool(get_settings().retrieval_config.get("chunking", {}).get("use_table_transcriptions", False))


def make_document_id(pdf_path: Path) -> str:
    digest = hashlib.sha1(pdf_path.read_bytes()).hexdigest()[:8]
    return f"{digest}_{_slugify(pdf_path.stem)}"


def process_pdf(pdf_path: Path, loader: PDFLoader, extractor: MetadataExtractor,
                 chunker: PageAwareChunker, section_detector: SectionDetector) -> dict:
    document_id = make_document_id(pdf_path)

    if loader.is_scanned(str(pdf_path)):
        return {
            "document_id": document_id, "title": pdf_path.stem, "authors": [], "year": None,
            "source": "upload", "filename": pdf_path.name, "num_pages": 0,
            "status": "scanned_needs_ocr", "chunks": [],
        }

    pages = loader.load_structured(str(pdf_path), document_id, REPO_ROOT / "data" / "figures", REPO_ROOT, use_table_cache=USE_TABLE_CACHE)
    metadata = extractor.extract(str(pdf_path), pages)
    chunks = chunker.chunk_document(document_id, pages, section_detector)

    return {
        "document_id": document_id,
        "title": metadata["title"],
        "authors": metadata["authors"],
        "year": metadata["year"],
        "source": "upload",
        "filename": pdf_path.name,
        "num_pages": len(pages),
        "status": "indexed",
        "chunks": [
            {
                "chunk_id": c.chunk_id, "document_id": c.document_id, "page_number": c.page_number,
                "section": c.section, "text": c.text, "token_count": c.token_count,
                "chunk_type": c.chunk_type, "label": c.label, "image_path": c.image_path,
            }
            for c in chunks
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest PDFs into page-aware chunk JSON records.")
    parser.add_argument("--input", default="data/raw")
    parser.add_argument("--output", default="data/processed")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    settings = get_settings()
    chunking_cfg = settings.retrieval_config.get("chunking", {})
    embedding_cfg = settings.models_config.get("embedding_model", {})

    loader = PDFLoader()
    extractor = MetadataExtractor()
    section_detector = SectionDetector()
    chunker = PageAwareChunker.from_config(chunking_cfg, embedding_cfg.get("name", "BAAI/bge-small-en-v1.5"))

    pdf_paths = sorted(input_dir.glob("*.pdf"))
    if not pdf_paths:
        print(f"No PDFs found in {input_dir}")
        return

    n_files = n_scanned = n_pages = n_chunks = n_failed = 0
    type_counts: dict[str, int] = {}
    t0 = time.time()

    for pdf_path in pdf_paths:
        try:
            record = process_pdf(pdf_path, loader, extractor, chunker, section_detector)
        except Exception as exc:
            print(f"  FAILED {pdf_path.name}: {exc}")
            n_failed += 1
            continue

        out_path = output_dir / f"{record['document_id']}.json"
        try:
            out_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
        except (UnicodeEncodeError, OSError) as exc:
            print(f"  FAILED {pdf_path.name}: could not write {out_path.name}: {exc}")
            n_failed += 1
            continue

        n_files += 1
        n_pages += record["num_pages"]
        n_chunks += len(record["chunks"])
        for c in record["chunks"]:
            type_counts[c.get("chunk_type", "text")] = type_counts.get(c.get("chunk_type", "text"), 0) + 1
        if record["status"] == "scanned_needs_ocr":
            n_scanned += 1
        print(f"  {pdf_path.name} -> {out_path.name} "
              f"({record['num_pages']} pages, {len(record['chunks'])} chunks, status={record['status']})")

    elapsed = time.time() - t0
    print("\n=== Ingestion summary ===")
    print(f"{'files processed':<20}{n_files}")
    print(f"{'pages':<20}{n_pages}")
    print(f"{'chunks':<20}{n_chunks}")
    print(f"{'  by type':<20}{type_counts}")
    print(f"{'scanned (skipped)':<20}{n_scanned}")
    print(f"{'failed':<20}{n_failed}")
    print(f"{'elapsed (s)':<20}{elapsed:.2f}")


if __name__ == "__main__":
    main()
