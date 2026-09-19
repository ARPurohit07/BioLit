"""scripts/build_index.py must register processed papers in the documents DB,
otherwise the API/UI report zero documents for a CLI-built index."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

pytest.importorskip("faiss", reason="faiss not installed")
pytest.importorskip("sentence_transformers", reason="sentence-transformers not installed")

from backend.app import db  # noqa: E402
from backend.app.models.schemas import DocumentStatus  # noqa: E402
from scripts.build_index import register_documents  # noqa: E402


def _write_record(directory: Path, document_id: str, **overrides) -> None:
    record = {
        "document_id": document_id, "title": f"Paper {document_id}", "authors": ["A. Author"], "year": 2023,
        "source": "upload", "filename": f"{document_id}.pdf", "num_pages": 3, "status": "indexed",
        "chunks": [{"chunk_id": f"{document_id}_0", "document_id": document_id, "page_number": 1,
                    "section": "Abstract", "text": "x", "token_count": 1}] * 4,
    }
    record.update(overrides)
    (directory / f"{document_id}.json").write_text(json.dumps(record), encoding="utf-8")


@pytest.fixture()
def tmp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "_db_path", lambda: tmp_path / "biolit.db")
    return tmp_path


def test_registers_indexed_and_scanned_papers(tmp_db):
    _write_record(tmp_db, "aaa")
    _write_record(tmp_db, "bbb", status="scanned_needs_ocr", chunks=[], num_pages=0)

    assert register_documents(tmp_db) == 2

    docs = {d.document_id: d for d in db.list_documents()}
    assert docs["aaa"].status == DocumentStatus.INDEXED
    assert docs["aaa"].title == "Paper aaa"
    assert docs["aaa"].authors == ["A. Author"]
    assert docs["aaa"].num_chunks == 4
    assert docs["bbb"].status == DocumentStatus.SCANNED_NEEDS_OCR
    assert docs["bbb"].error_message


def test_rerunning_is_idempotent_and_keeps_uploaded_at(tmp_db):
    _write_record(tmp_db, "aaa")
    register_documents(tmp_db)
    first = db.get_document("aaa").uploaded_at

    register_documents(tmp_db)

    assert len(db.list_documents()) == 1
    assert db.get_document("aaa").uploaded_at == first
