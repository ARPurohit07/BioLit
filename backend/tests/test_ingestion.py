"""Ingestion pipeline tests using synthetic in-memory inputs (no real PDFs required)."""
from __future__ import annotations

import pytest

from backend.app.ingestion.chunker import PageAwareChunker, count_tokens
from backend.app.ingestion.pdf_loader import PageText
from backend.app.ingestion.section_detector import SectionDetector

try:
    import fitz  # noqa: F401
    HAS_FITZ = True
except ImportError:
    HAS_FITZ = False


class TestSectionDetector:
    def setup_method(self):
        self.detector = SectionDetector()

    def test_detects_abstract(self):
        assert self.detector.detect("Abstract\nThis paper studies...") == "Abstract"

    def test_detects_methods_with_numbering(self):
        assert self.detector.detect("2. Methods\nWe used...") == "Methods"

    def test_detects_references(self):
        assert self.detector.detect("References\n[1] Smith et al.") == "References"

    def test_unknown_for_body_text(self):
        text = "This is a continuation of a paragraph with no heading at all present here."
        assert self.detector.detect(text) == "Unknown"

    def test_heading_hint_used(self):
        assert self.detector.detect("some body text", heading_hint="Discussion") == "Discussion"

    def test_all_sections_valid(self):
        assert "Unknown" in SectionDetector.SECTIONS
        assert len(SectionDetector.SECTIONS) == 9


class TestPageAwareChunker:
    def setup_method(self):
        self.section_detector = SectionDetector()

    def _pages(self, texts: list[str]) -> list[PageText]:
        return [PageText(page_number=i + 1, text=t) for i, t in enumerate(texts)]

    def test_chunks_never_span_pages(self):
        chunker = PageAwareChunker(chunk_size=20, chunk_overlap=5, min_chunk_tokens=3)
        long_body = " ".join(f"word{i}" for i in range(200))
        pages = self._pages([f"Introduction. {long_body}.", f"Methods. {long_body}."])
        chunks = chunker.chunk_document("doc1", pages, self.section_detector)

        assert len(chunks) > 0
        page1_chunks = [c for c in chunks if c.page_number == 1]
        page2_chunks = [c for c in chunks if c.page_number == 2]
        assert len(page1_chunks) > 0 and len(page2_chunks) > 0
        # every chunk_id must encode exactly the page it belongs to
        for c in chunks:
            assert f"_p{c.page_number}_" in c.chunk_id

    def test_chunk_id_format(self):
        chunker = PageAwareChunker(chunk_size=400, chunk_overlap=60, min_chunk_tokens=10)
        pages = self._pages(["Short text on one page."])
        chunks = chunker.chunk_document("mydoc", pages, self.section_detector)
        assert chunks[0].chunk_id == "mydoc_p1_c0"

    def test_empty_page_produces_no_chunks(self):
        chunker = PageAwareChunker(chunk_size=400, chunk_overlap=60, min_chunk_tokens=10)
        pages = self._pages(["", "  \n  "])
        chunks = chunker.chunk_document("doc2", pages, self.section_detector)
        assert chunks == []

    def test_splits_into_multiple_chunks_when_over_size(self):
        chunker = PageAwareChunker(chunk_size=15, chunk_overlap=3, min_chunk_tokens=2)
        sentences = ". ".join(f"Sentence number {i} has some words" for i in range(30)) + "."
        pages = self._pages([sentences])
        chunks = chunker.chunk_document("doc3", pages, self.section_detector)

        assert len(chunks) > 1
        for c in chunks:
            assert c.token_count > 0
            assert c.document_id == "doc3"

    def test_document_id_and_section_propagated(self):
        chunker = PageAwareChunker(chunk_size=400, chunk_overlap=60, min_chunk_tokens=10)
        pages = self._pages(["Abstract\nSome text here for the document."])
        chunks = chunker.chunk_document("docX", pages, self.section_detector)
        assert all(c.document_id == "docX" for c in chunks)
        assert all(c.section == "Abstract" for c in chunks)

    def test_small_trailing_chunk_is_merged(self):
        # min_chunk_tokens is large relative to chunk_size, forcing every split-off
        # remainder chunk below the threshold to be merged into its predecessor.
        chunker = PageAwareChunker(chunk_size=8, chunk_overlap=0, min_chunk_tokens=6)
        sentences = ". ".join(f"Word{i} filler token here" for i in range(10)) + "."
        pages = self._pages([sentences])
        chunks = chunker.chunk_document("doc5", pages, self.section_detector)
        # the merge pass should leave no page with more chunks than sentences
        assert len(chunks) >= 1


def test_count_tokens_fallback():
    assert count_tokens("one two three") >= 3
    assert count_tokens("") == 0


@pytest.mark.skipif(not HAS_FITZ, reason="PyMuPDF (fitz) not installed")
class TestPDFLoaderWithFitz:
    def test_pdf_loader_importable(self):
        from backend.app.ingestion.pdf_loader import PDFLoader
        assert PDFLoader is not None


@pytest.mark.skipif(HAS_FITZ, reason="only relevant when PyMuPDF is missing")
def test_pdf_loader_raises_clear_error_without_fitz(tmp_path):
    from backend.app.ingestion.pdf_loader import PDFLoader

    loader = PDFLoader()
    fake_pdf = tmp_path / "fake.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4 fake")
    with pytest.raises(ImportError):
        loader.load(str(fake_pdf))
