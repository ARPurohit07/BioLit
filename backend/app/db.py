"""SQLite-backed metadata store for uploaded/indexed documents.

Uses stdlib sqlite3 directly (no ORM). One row per DocumentMetadata, at
settings.sqlite_path.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

from backend.app.config.settings import get_settings
from backend.app.models.schemas import DocumentMetadata, DocumentStatus

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    document_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    authors TEXT NOT NULL DEFAULT '[]',
    year INTEGER,
    source TEXT NOT NULL DEFAULT 'upload',
    filename TEXT NOT NULL,
    num_pages INTEGER NOT NULL DEFAULT 0,
    num_chunks INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'uploaded',
    uploaded_at TEXT NOT NULL,
    error_message TEXT
);
"""


def _db_path() -> Path:
    return get_settings().sqlite_path


@contextmanager
def _connect() -> Iterator[sqlite3.Connection]:
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    try:
        conn.row_factory = sqlite3.Row
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with _connect() as conn:
        conn.execute(_SCHEMA)


def _row_to_meta(row: sqlite3.Row) -> DocumentMetadata:
    return DocumentMetadata(
        document_id=row["document_id"],
        title=row["title"],
        authors=json.loads(row["authors"]),
        year=row["year"],
        source=row["source"],
        filename=row["filename"],
        num_pages=row["num_pages"],
        num_chunks=row["num_chunks"],
        status=DocumentStatus(row["status"]),
        uploaded_at=row["uploaded_at"],
        error_message=row["error_message"],
    )


def upsert_document(meta: DocumentMetadata) -> None:
    init_db()
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO documents (document_id, title, authors, year, source, filename,
                                    num_pages, num_chunks, status, uploaded_at, error_message)
            VALUES (:document_id, :title, :authors, :year, :source, :filename,
                    :num_pages, :num_chunks, :status, :uploaded_at, :error_message)
            ON CONFLICT(document_id) DO UPDATE SET
                title=excluded.title,
                authors=excluded.authors,
                year=excluded.year,
                source=excluded.source,
                filename=excluded.filename,
                num_pages=excluded.num_pages,
                num_chunks=excluded.num_chunks,
                status=excluded.status,
                uploaded_at=excluded.uploaded_at,
                error_message=excluded.error_message
            """,
            {
                "document_id": meta.document_id,
                "title": meta.title,
                "authors": json.dumps(meta.authors),
                "year": meta.year,
                "source": meta.source,
                "filename": meta.filename,
                "num_pages": meta.num_pages,
                "num_chunks": meta.num_chunks,
                "status": meta.status.value,
                "uploaded_at": meta.uploaded_at.isoformat(),
                "error_message": meta.error_message,
            },
        )


def get_document(document_id: str) -> Optional[DocumentMetadata]:
    init_db()
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM documents WHERE document_id = ?", (document_id,)
        ).fetchone()
        return _row_to_meta(row) if row else None


def list_documents() -> list[DocumentMetadata]:
    init_db()
    with _connect() as conn:
        rows = conn.execute("SELECT * FROM documents ORDER BY uploaded_at DESC").fetchall()
        return [_row_to_meta(r) for r in rows]


def delete_document(document_id: str) -> None:
    init_db()
    with _connect() as conn:
        conn.execute("DELETE FROM documents WHERE document_id = ?", (document_id,))
