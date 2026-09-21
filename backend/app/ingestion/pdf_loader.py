"""PDF text extraction and scanned-document detection via PyMuPDF.

fitz is imported defensively (not at hard module scope) so that other
ingestion modules which only need the ``PageText`` dataclass (e.g. the
chunker) stay importable on machines where PyMuPDF isn't installed yet.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

try:
    import fitz  # PyMuPDF
except ImportError:  # pragma: no cover - exercised only when PyMuPDF is missing
    fitz = None


def _strip_surrogates(text: str) -> str:
    return "".join(ch for ch in text if not 0xD800 <= ord(ch) <= 0xDFFF)


@dataclass
class PageText:
    page_number: int  # 1-indexed
    text: str
    tables: list = field(default_factory=list)   # structure.TableBlock, set by load_structured()
    figures: list = field(default_factory=list)  # structure.FigureBlock, set by load_structured()


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

    def load_structured(self, pdf_path: str, doc_key: str, figures_dir: Path, repo_root: Path,
                        use_table_cache: bool = False) -> list[PageText]:
        """Like load(), but tables and figures are separated from the body text (see ingestion/structure.py)."""
        self._require_fitz()
        from backend.app.ingestion.structure import StructureParser, apply_transcriptions

        parser = StructureParser(figures_dir, repo_root)
        cache_path = Path(repo_root) / "data" / "table_cache" / f"{doc_key}.json"
        cache = json.loads(cache_path.read_text(encoding="utf-8")) if use_table_cache and cache_path.exists() else {}
        pages: list[PageText] = []
        with fitz.open(pdf_path) as doc:
            for i, page in enumerate(doc):
                try:
                    s = parser.parse_page(page, doc_key, i + 1)
                    apply_transcriptions(s.tables, i + 1, cache)
                    pages.append(PageText(i + 1, s.body_text, s.tables, s.figures))
                except Exception:  # a page the parser cannot handle keeps its plain text: never lose content
                    pages.append(PageText(i + 1, _strip_surrogates(page.get_text("text") or "").strip()))
        return pages

    def is_scanned(self, pdf_path: str) -> bool:
        pages = self.load(pdf_path)
        if not pages:
            return True
        avg_chars = sum(len(p.text) for p in pages) / len(pages)
        return avg_chars < self.SCANNED_CHARS_PER_PAGE_THRESHOLD
