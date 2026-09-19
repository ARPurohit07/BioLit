"""End-to-end integration test: PDF -> ingestion -> retrieval -> Ollama -> citation validation.

This exercises the full pipeline against a tiny synthetic PDF built on the fly
with PyMuPDF, so it doesn't depend on any real biomedical paper being present.
It requires the heavy optional dependencies (fitz, faiss, sentence-transformers,
torch) AND a reachable local Ollama server with the configured model pulled —
both are skipped gracefully when unavailable, since this environment may not
have them installed/running yet.

Run with:
    pytest tests/test_integration.py -v
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

fitz = pytest.importorskip("fitz", reason="PyMuPDF not installed")
pytest.importorskip("faiss", reason="faiss not installed")
pytest.importorskip("sentence_transformers", reason="sentence-transformers not installed")
requests = pytest.importorskip("requests", reason="requests not installed")

from backend.app.config.settings import get_settings  # noqa: E402


def _ollama_reachable() -> bool:
    settings = get_settings()
    try:
        r = requests.get(f"{settings.ollama_host}/api/tags", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def _make_synthetic_pdf(path: Path) -> None:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Abstract")
    page.insert_text(
        (72, 100),
        "This study evaluates Compound X in a murine xenograft model of breast cancer.",
    )
    page2 = doc.new_page()
    page2.insert_text((72, 72), "Results")
    page2.insert_text(
        (72, 100),
        "Compound X reduced tumor volume by 32 percent relative to vehicle control (p<0.01).",
    )
    doc.save(str(path))
    doc.close()


@pytest.mark.skipif(not _ollama_reachable(), reason="Ollama server not reachable at configured OLLAMA_HOST")
def test_full_pipeline(tmp_path: Path):
    from backend.app.ingestion.pdf_loader import PDFLoader
    from backend.app.ingestion.section_detector import SectionDetector
    from backend.app.ingestion.chunker import PageAwareChunker
    from backend.app.retrieval.embeddings import EmbeddingModel
    from backend.app.retrieval.vector_store import FAISSVectorStore
    from backend.app.retrieval.bm25 import BM25Index
    from backend.app.retrieval.hybrid import HybridRetriever
    from backend.app.generation.ollama_client import OllamaClient
    from backend.app.verification.claims import ClaimExtractor
    from backend.app.verification.citation_validator import compute_citation_metrics

    settings = get_settings()
    pdf_path = tmp_path / "synthetic_paper.pdf"
    _make_synthetic_pdf(pdf_path)

    loader = PDFLoader()
    pages = loader.load(str(pdf_path))
    assert len(pages) == 2

    chunker = PageAwareChunker(chunk_size=400, chunk_overlap=60, min_chunk_tokens=5)
    chunks = chunker.chunk_document("synthetic_paper", pages, SectionDetector())
    assert len(chunks) >= 1

    embed_model = EmbeddingModel(settings.models_config["embedding_model"]["name"])
    embeddings = embed_model.encode([c.text for c in chunks])

    vector_store = FAISSVectorStore(
        dim=embeddings.shape[1],
        index_path=str(tmp_path / "faiss.index"),
        metadata_path=str(tmp_path / "chunk_metadata.jsonl"),
    )
    vector_store.add(chunks, embeddings)

    bm25 = BM25Index(index_path=str(tmp_path / "bm25.pkl"))
    bm25.add(chunks)

    retriever = HybridRetriever(vector_store, bm25, embed_model, settings.retrieval_config)
    results = retriever.retrieve("How much did Compound X reduce tumor volume?", mode="balanced")
    assert len(results) >= 1

    client = OllamaClient(settings.ollama_host, settings.ollama_model)
    assert client.is_available()

    # Minimal grounded-generation smoke check.
    top_chunk, _ = results[0]
    answer = client.generate(
        prompt=f"Evidence: {top_chunk.text}\n\nQuestion: How much did Compound X reduce tumor volume? "
               f"Answer in one sentence and cite as [1].",
        system="Answer only from the given evidence. Cite as [1].",
    )
    assert isinstance(answer, str) and len(answer) > 0

    claims = ClaimExtractor().extract(answer, evidence=[])
    metrics = compute_citation_metrics(claims)
    assert 0.0 <= metrics.faithfulness <= 1.0
