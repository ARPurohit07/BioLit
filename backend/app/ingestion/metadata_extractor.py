"""Document metadata extraction: PDF metadata first, page-1 heuristics as fallback."""
from __future__ import annotations

import re

from backend.app.ingestion.pdf_loader import PageText, fitz

_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
_AUTHOR_SPLIT_RE = re.compile(r",| and |;|&")
# arXiv stamps a watermark ("arXiv:2208.06361v1  [q-bio.BM]  6 Jul 2022") on page 1 of
# every preprint, often at a larger font size than the real title — must be excluded
# from the largest-font title heuristic or it wins outright.
_ARXIV_WATERMARK_RE = re.compile(r"^arxiv:\s*\d{4}\.\d{4,5}", re.IGNORECASE)
_AFFIL_KEYWORDS = (
    "university", "institute", "department", "dept", "college", "hospital",
    "corp", "inc", "ltd", "laboratory", "lab", "school", "center", "centre",
    "faculty", "division", "group",
)
_NAME_LINE_RE = re.compile(r"^[A-Z][\w.'-]*(\s+[A-Z][\w.'-]*){1,4}$")


class MetadataExtractor:
    def extract(self, pdf_path: str, pages: list[PageText]) -> dict:
        title: str | None = None
        authors: list[str] = []
        year: int | None = None

        if fitz is not None:
            try:
                with fitz.open(pdf_path) as doc:
                    meta = doc.metadata or {}
                    title = (meta.get("title") or "").strip() or None
                    author_field = (meta.get("author") or "").strip()
                    if author_field:
                        authors = self._split_authors(author_field)
                    year = self._year_from_metadata(meta)
                    if title is None or not authors:
                        page1_lines = self._page1_lines(doc)
                        heuristic_title, title_line_idx = self._title_from_lines(page1_lines)
                        title = title or heuristic_title
                        if not authors and title_line_idx is not None:
                            authors = self._authors_from_lines(page1_lines, title_line_idx)
            except Exception:
                pass

        if title is None:
            title = self._title_from_pages(pages)
        if year is None:
            year = self._year_from_pages(pages)

        return {"title": title or "Untitled", "authors": authors, "year": year}

    def _split_authors(self, author_field: str) -> list[str]:
        parts = _AUTHOR_SPLIT_RE.split(author_field)
        return [p.strip() for p in parts if p.strip()]

    def _year_from_metadata(self, meta: dict) -> int | None:
        for key in ("creationDate", "modDate"):
            val = meta.get(key) or ""
            m = re.search(r"D:(\d{4})", val)
            if m:
                return int(m.group(1))
        return None

    def _page1_lines(self, doc) -> list[tuple[float, str]]:
        """Page-1 text lines as (max_font_size_in_line, text), in document order,
        with the arXiv watermark line dropped."""
        if doc.page_count == 0:
            return []
        page = doc[0]
        lines: list[tuple[float, str]] = []
        for block in page.get_text("dict").get("blocks", []):
            for line in block.get("lines", []):
                spans = [s for s in line.get("spans", []) if s.get("text", "").strip()]
                if not spans:
                    continue
                text = " ".join(s["text"].strip() for s in spans)
                if _ARXIV_WATERMARK_RE.match(text):
                    continue
                max_size = max(s.get("size", 0.0) for s in spans)
                lines.append((max_size, text))
        return lines

    def _title_from_lines(self, lines: list[tuple[float, str]]) -> tuple[str | None, int | None]:
        """Largest-font-size line(s) on page 1 are usually the title. Returns the
        title text and the index of its last line (so callers can scan for authors
        immediately below it)."""
        if not lines:
            return None, None
        max_size = max(size for size, _ in lines)
        title_idx = [i for i, (size, _) in enumerate(lines) if size >= max_size - 0.5][:3]
        if not title_idx:
            return None, None
        title = " ".join(lines[i][1] for i in title_idx).strip()
        return (title or None), title_idx[-1]

    def _authors_from_lines(self, lines: list[tuple[float, str]], title_line_idx: int) -> list[str]:
        """Scans lines immediately below the title for person-name-shaped lines,
        stopping at the first heading (Abstract/Introduction/...) or after a short
        window with nothing more to find. Font size alone can't separate authors
        from body text here — arXiv author blocks are frequently the same size as
        the abstract paragraph — so this relies on shape (short, capitalized,
        2-5 words) and excludes affiliation/email lines."""
        stop_keywords = {"abstract", "introduction", "keywords", "index terms"}
        authors: list[str] = []
        window = lines[title_line_idx + 1: title_line_idx + 1 + 25]
        for _size, text in window:
            normalized = text.strip().lower().rstrip(":")
            if normalized in stop_keywords:
                break
            if "@" in text or any(kw in normalized for kw in _AFFIL_KEYWORDS):
                continue
            if any(ch.isdigit() for ch in text):
                continue
            if _NAME_LINE_RE.match(text.strip()):
                authors.append(text.strip())
            if len(authors) >= 10:
                break
        return authors

    def _title_from_pages(self, pages: list[PageText]) -> str | None:
        if not pages:
            return None
        lines = [line.strip() for line in pages[0].text.splitlines() if line.strip()]
        return lines[0] if lines else None

    def _year_from_pages(self, pages: list[PageText]) -> int | None:
        if not pages:
            return None
        m = _YEAR_RE.search(pages[0].text)
        return int(m.group(0)) if m else None
