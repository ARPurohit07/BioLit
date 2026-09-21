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


# "Surname, A." / "Surname, A. B." entries. The lookbehind rejects the "K. Schwarz, A. Pliego" pattern of
# initials-first author bylines, where the comma-initial is really the next author's.
_SURNAME_INITIAL_RE = re.compile(r"(?<![A-Z]\. )\b[A-Z][A-Za-z\-]{1,30},\s?[A-Z]\.(?:\s?[A-Z]\.)?")
_NUMBERED_INITIAL_RE = re.compile(r"\[\d{1,3}\]\s*[A-Z]\.\s?(?:[A-Z]\.\s?)?[A-Z][A-Za-z\-]+")
_YEAR_RE = re.compile(r"\b(?:19[5-9]\d|20[0-2]\d)[a-z]?\b")
_LOCATOR_RE = re.compile(
    r"\b(?:pp\.\s?\d+|vol\.\s?\d+|\d{1,4}\s?\(\d{1,3}\)\s?[:,]\s?[A-Za-z]?\d+|\d{1,5}\s?[–-]\s?\d{1,5}\)?[.,])")
_FINITE_VERB_RE = re.compile(
    r"\b(?:is|are|was|were|has|have|had|shows?|shown|showed|propose[sd]?|introduce[sd]?|demonstrate[sd]?|"
    r"present(?:s|ed)?|use[sd]?|provide[sd]?|found|report(?:s|ed)?|achieve[sd]?|outperform(?:s|ed)?|"
    r"train(?:s|ed)?|evaluate[sd]?|indicate[sd]?|suggest(?:s|ed)?|can|could|will|would|should|"
    r"need(?:s|ed)?|allow(?:s|ed)?|require[sd]?)\b", re.IGNORECASE)

_MIN_REFERENCE_WORDS = 20


def _reference_score(text: str) -> float:
    """>= 1.0 means the text is a reference list. Every signal is a density or needs several
    co-occurring hits, so prose that merely cites a few works ("Smith et al., 2020") scores ~0."""
    n = len(text.split())
    if n < _MIN_REFERENCE_WORDS:
        return 0.0
    surname_initials = len(_SURNAME_INITIAL_RE.findall(text))
    numbered = len(_NUMBERED_INITIAL_RE.findall(text))
    locators = len(_LOCATOR_RE.findall(text))
    years = len(_YEAR_RE.findall(text))
    verbs_per_100 = len(_FINITE_VERB_RE.findall(text)) / n * 100

    author_style = min(surname_initials / 3, (surname_initials / n * 100) / 2.0)
    numbered_style = numbered / 2
    locator_style = min(locators / 4, years / 3, 2.0 / verbs_per_100 if verbs_per_100 else 1e3)
    return max(author_style, numbered_style, locator_style)


def is_reference_list(text: str) -> bool:
    return _reference_score(text) >= 1.0
