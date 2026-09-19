"""Regex/keyword-based section-heading detection for biomedical paper pages."""
from __future__ import annotations

import re


class SectionDetector:
    SECTIONS = ["Abstract", "Introduction", "Methods", "Results", "Discussion",
                "Limitations", "Conclusion", "References", "Unknown"]

    # Keyword variants checked against a normalized (lowercased, punctuation-stripped) line.
    # Includes singular forms ("Method", "Result") since many ML/computational papers use
    # them instead of "Methods"/"Results".
    _KEYWORDS: dict[str, list[str]] = {
        "Abstract": ["abstract", "summary"],
        "Introduction": ["introduction", "background"],
        "Methods": ["method", "methods", "materials and methods", "materials & methods",
                    "methodology", "experimental setup", "experimental methods",
                    "materials and method"],
        "Results": ["result", "results", "findings", "results and discussion",
                    "experimental results", "experiments and results"],
        "Discussion": ["discussion"],
        "Limitations": ["limitations", "limitation", "study limitations"],
        "Conclusion": ["conclusion", "conclusions", "concluding remarks",
                       "conclusion and future work", "conclusions and future work"],
        "References": ["references", "bibliography", "works cited", "literature cited"],
    }

    # Strips a leading numbering prefix like "2." or "II)" before matching keywords.
    _NUMBERING_PREFIX_RE = re.compile(r"^\s*(?:[ivxlc]+[.)]|\d+[.)]?)\s*", re.IGNORECASE)

    def detect(self, page_text: str, heading_hint: str | None = None) -> str:
        """Best single section guess for a page (first heading found, else Unknown).

        Prefer scan_headings() when position within the page matters — a heading
        can appear anywhere on the page (after a figure caption, mid-page, etc.),
        not just in the first few lines.
        """
        if heading_hint:
            section = self._match_keyword(heading_hint)
            if section:
                return section
        for _line_idx, section in self.scan_headings(page_text):
            return section
        return "Unknown"

    def scan_headings(self, page_text: str) -> list[tuple[int, str]]:
        """Returns [(line_index, section), ...] for every heading-like line on the
        page, in top-to-bottom order. line_index is into page_text.splitlines()
        (including blank lines, so callers can align it back to the original text)."""
        hits: list[tuple[int, str]] = []
        for idx, line in enumerate(page_text.splitlines()):
            stripped = line.strip()
            if not stripped:
                continue
            section = self._match_keyword(stripped)
            if section:
                hits.append((idx, section))
        return hits

    def _match_keyword(self, line: str) -> str | None:
        normalized = self._NUMBERING_PREFIX_RE.sub("", line.strip().lower()).strip(" :.-")
        if not normalized:
            return None
        for section, keywords in self._KEYWORDS.items():
            for kw in keywords:
                if normalized == kw or normalized == kw + "s":
                    return section
        return None
