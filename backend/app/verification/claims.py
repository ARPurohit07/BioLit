"""Splits an LLM answer into sentence-level Claims with citation markers parsed out.

A "claim" is a factual assertion that could be checked against evidence. Scaffolding is not:
headings, lead-ins ("The main findings are:"), bare citation markers, label-only lines, an
uncited "the evidence does not specify X" caveat, and any echo of the regeneration prompt.
Counting those as claims made every answer look mostly unsupported.
"""
from __future__ import annotations

import re

from backend.app.models.schemas import Claim, EvidenceItem
from backend.app.verification.grounding import annotate as annotate_grounding

_SINGLE_MARKER_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")

# Split on sentence boundaries; pieces that are only citation markers are re-attached afterwards.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\[\-\*])")

_LABELS = r"SUPPORTED CLAIM|INTERPRETATION|LIMITATION"
# Leading markdown list / heading markers. Requires whitespace after the marker so a statement that
# starts with a number ("19.32% relative improvement...") is left intact.
_LIST_MARKER_RE = re.compile(r"^\s*(?:#{1,6}\s+|[-*•]\s+|\d+[.)]\s+)")
_LEADING_LABEL_RE = re.compile(rf"^\s*(?:{_LABELS})\s*:\s*", re.IGNORECASE)
_TRAILING_LABEL_RE = re.compile(rf"\s*(?:{_LABELS})\W*$", re.IGNORECASE)
# e.g. "[1][4] SUPPORTED CLAIM" — the model put the citation after the period.
_CITATION_ONLY_RE = re.compile(
    rf"^(?:\[\d+(?:\s*,\s*\d+)*\]\s*)+(?:(?:{_LABELS})\W*)?$", re.IGNORECASE
)
_HEADING_LINE_RE = re.compile(r"^\s*#{1,6}\s+")
_BOLD_ONLY_RE = re.compile(r"^\s*(?:\*\*|__)[^*_]+(?:\*\*|__)\s*:?\s*$")

# The pipeline's own inline note (see RAGPipeline._annotate_unsupported). It is removed from a
# statement, not used to discard the statement: real claims can sit right next to it.
_NOTE_RE = re.compile(r"\*?\(unsupported\s*[—-][^)]*\)\*?", re.IGNORECASE)

# Text the pipeline adds around regenerated answers; a model often echoes it back verbatim.
_SCAFFOLD_RE = re.compile(
    r"the following statements? from that answer|\bunverified statements\b|"
    r"^\W*(?:revised answer|here is the revised|your previous answer)\b",
    re.IGNORECASE,
)
_BULLET_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")

# "The evidence does not specify X" is a caveat, not an assertion to verify.
_ABSTAIN_RE = re.compile(
    r"\binsufficient (?:evidence|information)\b"
    r"|\b(?:does|do|did) not (?:explicitly |directly |specifically )?"
    r"(?:specify|provide|mention|state|address|describe|demonstrate|contain|include|discuss|report|cover)\b"
    r"|\bnot (?:explicitly |directly |specifically )?(?:provided|specified|mentioned|stated|described|"
    r"demonstrated|addressed|covered|reported)\b",
    re.IGNORECASE,
)

_MIN_WORDS_UNCITED = 4


def _extract_citation_ids(statement: str) -> list[int]:
    ids: list[int] = []
    for match in _SINGLE_MARKER_RE.finditer(statement):
        for piece in match.group(1).split(","):
            piece = piece.strip()
            if piece.isdigit():
                n = int(piece)
                if n not in ids:
                    ids.append(n)
    return ids


def is_abstention(answer_text: str) -> bool:
    """True when the answer only says the evidence is insufficient — there is nothing in it to cite."""
    statements = [
        s
        for line in answer_text.splitlines()
        if line.strip()
        for s in _statements(_clean_line(line.strip()))
        if len(re.findall(r"\w+", s)) >= 3
    ]
    return bool(statements) and all(_ABSTAIN_RE.search(s) for s in statements)


def strip_regeneration_scaffold(answer_text: str) -> str:
    """Drop echoed regeneration-prompt scaffolding (and the flagged-claim bullets listed under it)."""
    kept: list[str] = []
    in_echoed_list = False
    for line in answer_text.splitlines():
        if _SCAFFOLD_RE.search(line):
            in_echoed_list = True
            continue
        if in_echoed_list and (_BULLET_RE.match(line) or not line.strip()):
            continue
        in_echoed_list = False
        kept.append(line)
    return "\n".join(kept).strip()


def _clean_line(line: str) -> str:
    """Strip list/heading markers, bold markers and a leading SUPPORTED CLAIM:/LIMITATION: label."""
    line = _NOTE_RE.sub("", line)
    line = _LIST_MARKER_RE.sub("", line)
    line = line.replace("**", "").replace("__", "")
    return re.sub(r"\s{2,}", " ", _LEADING_LABEL_RE.sub("", line)).strip()


def _statements(line: str) -> list[str]:
    """One cleaned line -> claim-candidate statements, with citation-only fragments re-attached."""
    pieces: list[str] = []
    for piece in _SENTENCE_SPLIT_RE.split(line):
        piece = piece.strip()
        if not piece:
            continue
        if pieces and _CITATION_ONLY_RE.match(piece):
            pieces[-1] = _TRAILING_LABEL_RE.sub("", f"{pieces[-1]} {piece}").strip()
        else:
            pieces.append(piece)
    return pieces


def _is_claim(statement: str, has_citation: bool) -> bool:
    words = re.findall(r"\w+", _SINGLE_MARKER_RE.sub("", statement))
    if not words:
        return False  # a bare "[3]"
    if statement.rstrip().endswith(":"):
        return False  # lead-in such as "The main findings are:"
    if _SCAFFOLD_RE.search(statement):
        return False
    if not has_citation and _ABSTAIN_RE.search(statement):
        return False  # uncited caveat, not an assertion
    if not has_citation and len(words) < _MIN_WORDS_UNCITED:
        return False  # uncited fragment
    return True


class ClaimExtractor:
    def extract(self, answer_text: str, evidence: list[EvidenceItem]) -> list[Claim]:
        valid_ids = {e.citation_id for e in evidence}
        claims: list[Claim] = []
        idx = 0

        for raw_line in answer_text.splitlines():
            line = raw_line.strip()
            if not line or _HEADING_LINE_RE.match(line) or _BOLD_ONLY_RE.match(line):
                continue
            cleaned = _clean_line(line)
            if _CITATION_ONLY_RE.match(cleaned):
                # "SUPPORTED CLAIM: [3][4]" on its own line belongs to the claim above it.
                if claims:
                    extra = [c for c in _extract_citation_ids(cleaned) if c in valid_ids]
                    claims[-1].citation_ids += [c for c in extra if c not in claims[-1].citation_ids]
                    claims[-1].text = f"{claims[-1].text} {_TRAILING_LABEL_RE.sub('', cleaned).strip()}"
                continue
            for statement in _statements(cleaned):
                citation_ids = [c for c in _extract_citation_ids(statement) if c in valid_ids]
                if not _is_claim(statement, has_citation=bool(_extract_citation_ids(statement))):
                    continue
                claims.append(
                    Claim(
                        claim_id=f"claim_{idx}",
                        text=statement,
                        citation_ids=citation_ids,
                    )
                )
                idx += 1

        annotate_grounding(claims, evidence)  # a fast, LLM-free estimate; the verifier (when it runs) is separate
        return claims
