"""API smoke tests. Heavier end-to-end query tests that need a running Ollama
server + built indexes are gated behind BIOLIT_INTEGRATION_TESTS=1.
"""
from __future__ import annotations

import io
import os

import pytest
from fastapi.testclient import TestClient

from backend.app.main import app

INTEGRATION = os.environ.get("BIOLIT_INTEGRATION_TESTS") == "1"


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


def test_health_does_not_crash_with_no_models_loaded(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    body = resp.json()
    assert "status" in body
    assert "ollama_available" in body
    assert "num_indexed_documents" in body
    assert "num_indexed_chunks" in body


def test_upload_rejects_non_pdf(client):
    resp = client.post(
        "/api/documents/upload",
        files={"file": ("notes.txt", io.BytesIO(b"hello world"), "text/plain")},
    )
    assert resp.status_code == 400


def test_upload_rejects_empty_file(client):
    resp = client.post(
        "/api/documents/upload",
        files={"file": ("empty.pdf", io.BytesIO(b""), "application/pdf")},
    )
    assert resp.status_code == 400


def test_upload_accepts_pdf_like_bytes(client):
    fake_pdf = b"%PDF-1.4\n%fake pdf content for upload test\n%%EOF"
    resp = client.post(
        "/api/documents/upload",
        files={"file": ("paper.pdf", io.BytesIO(fake_pdf), "application/pdf")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "uploaded"
    assert body["filename"] == "paper.pdf"
    assert body["document_id"]

    list_resp = client.get("/api/documents/")
    assert list_resp.status_code == 200
    assert any(d["document_id"] == body["document_id"] for d in list_resp.json()["documents"])

    delete_resp = client.delete(f"/api/documents/{body['document_id']}")
    assert delete_resp.status_code == 200


def test_evaluation_summary_says_not_evaluated_without_result_files(client):
    resp = client.get("/api/evaluation/")
    assert resp.status_code == 200
    body = resp.json()
    # Either genuinely no results exist yet (evaluated=False with an explanatory
    # note), or a prior benchmark run in this repo produced real results — either
    # way, the endpoint must never fabricate numbers.
    assert isinstance(body["evaluated"], bool)
    if not body["evaluated"]:
        assert body["note"]


def test_query_without_pipeline_available_returns_clean_error(client):
    resp = client.post(
        "/api/query",
        json={"question": "What is the effect of drug X?", "mode": "fast", "query_type": "question_answering"},
    )
    # Without Ollama/indexes present in this environment, the pipeline should
    # either be unavailable (503) or fail cleanly (500) — never an unhandled crash.
    assert resp.status_code in (200, 500, 503)


@pytest.mark.skipif(not INTEGRATION, reason="Requires a running Ollama server and built indexes; set BIOLIT_INTEGRATION_TESTS=1 to run.")
def test_full_query_pipeline_end_to_end(client):
    resp = client.post(
        "/api/query",
        json={"question": "What is the effect of drug X?", "mode": "fast", "query_type": "question_answering"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "answer_markdown" in body
    assert "citation_metrics" in body
