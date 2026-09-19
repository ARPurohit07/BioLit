"""PDF text extraction and scanned-document detection via PyMuPDF.

fitz is imported defensively (not at hard module scope) so that other
ingestion modules which only need the ``PageText`` dataclass (e.g. the
chunker) stay importable on machines where PyMuPDF isn't installed yet.
"""
from __future__ import annotations

from dataclasses import dataclass

try:
    import fitz  # PyMuPDF
except ImportError:  # pragma: no cover - exercised only when PyMuPDF is missing
    fitz = None


@dataclass
class PageText:
    page_number: int  # 1-indexed
    text: str


class PDFLoader:
    """Extracts per-page text from PDFs and flags likely-scanned documents."""

    # Below this average extractable-characters-per-page, treat the PDF as scanned/image-only.
    SCANNED_CHARS_PER_PAGE_THRESHOLD = 50

    def _require_fitz(self) -> None:
        if fitz is None:
            raise ImportError("PyMuPDF (fitz) is required for PDFLoader — pip install PyMuPDF")

    def load(self, pdf_path: str) -> list[PageText]:
        self._require_fitz()
        pages: list[PageText] = []
        with fitz.open(pdf_path) as doc:
            for i, page in enumerate(doc):
                text = page.get_text("text") or ""
                pages.append(PageText(page_number=i + 1, text=text.strip()))
        return pages

    def is_scanned(self, pdf_path: str) -> bool:
        pages = self.load(pdf_path)
        if not pages:
            return True
        avg_chars = sum(len(p.text) for p in pages) / len(pages)
        return avg_chars < self.SCANNED_CHARS_PER_PAGE_THRESHOLD
