"""A fast, deterministic estimate of whether a claim is grounded in the evidence it cites. No LLM.

Fast and Balanced modes do not run the LLM verifier, so their claims are NOT_VERIFIED. This module gives them an
*estimate* instead of a blank: it is a heuristic, computed in milliseconds from the claim text and the cited
evidence blocks, and it is NOT a verdict. It must always be presented as an estimate, never as "faithfulness".

Signals (all cheap, all computed per claim):
  * the claim must cite at least one real evidence block, otherwise it is not grounded by construction;
  * wording overlap: how many of the claim's content words appear in the cited evidence;
  * numbers: every number in the claim must appear in the cited evidence, and specific numbers (decimals, 3+
    digits) must sit near the claim's other words there, which catches "right number, wrong setting";
  * per-citation relevance: each cited block must itself share some of the claim's wording, so a relevant block
    cannot vouch for an unrelated extra citation.

How far to trust it was measured against the LLM verifier on real claims (scripts/calibrate_grounding.py; results in
experiments/grounding_calibration.json and the README). Only the "none" label is a useful signal: it is a "check this
against the source" flag, right about half the time. "strong"/"weak" barely beat the base rate, so the UI does not
show them and no aggregate percentage is derived from them. The thresholds are module constants, fixed before the
calibration data was collected.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

_STOP = frozenset(
    "that this with from were have been which their these those also than then such using used "
    "into over both each other more most only some when while where about between within across "
    "based paper study studies results method methods approach model models proposed show shown "
    "shows however thus therefore they them does not are and the for can may our its it's".split()
)
# Words that describe the citation rather than the fact ("[1] mentions that ...", "specifically provides ..."):
# they are the answer's scaffolding and rarely appear in the evidence, so they would depress the overlap.
_REPORTING = frozenset(
    "mention mentions mentioned mentioning specifically specific provide provides provided providing detailed "
    "detail details suggest suggests suggested suggesting indicate indicates indicated highlight highlights "
    "highlighted describe describes described state states stated note notes noted discuss discusses discussed "
    "according evidence information context indeed further additionally overall generally notably".split()
)
_MARKER = re.compile(r"\[\d+(?:\s*,\s*\d+)*\]")
_LABEL = re.compile(r"^\s*(?:SUPPORTED CLAIM|INTERPRETATION|LIMITATION)\s*:\s*", re.IGNORECASE)
_NUMBER = re.compile(r"\d+(?:\.\d+)?")
_SPECIFIC_NUMBER = re.compile(r"\d+\.\d+|\d{3,}")

STRONG_MIN = 0.70          # pooled wording overlap at or above this = "strong"
WEAK_MIN = 0.45            # at or above this (but below STRONG_MIN) = "weak"; below = "none"
MIN_CITE_OVERLAP = 0.15    # each cited block must contain at least this share of the claim's words
MIN_NUMBER_CONTEXT = 0.40  # share of the claim's words that must sit near a specific number in the evidence
NUMBER_WINDOW = 30         # tokens either side of the number

STRONG, WEAK, NONE = "strong", "weak", "none"


def stem(word: str) -> str:
    """Crude suffix stripping so 'utilizes'/'utilize' and 'networks'/'network' match. Not linguistic stemming."""
    for suffix in ("ing", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[: -len(suffix)]
    return word


def content_words(text: str) -> set[str]:
    return {
        stem(w)
        for w in re.findall(r"[a-z][a-z0-9\-]{3,}", text.lower())
        if w not in _STOP and w not in _REPORTING
    }


def clean_claim(text: str) -> str:
    return _LABEL.sub("", _MARKER.sub("", text)).strip()


@dataclass(frozen=True)
class Signals:
    cited: bool
    overlap: float          # pooled content-word recall against the cited evidence (0-1)
    min_cite_overlap: float  # the least-related cited block's overlap (0-1)
    numbers_ok: bool        # every number in the claim appears in the cited evidence
    number_context_ok: bool  # specific numbers sit near the claim's wording in the evidence
    missing_numbers: tuple[str, ...] = ()


@dataclass(frozen=True)
class Grounding:
    label: str    # "strong" | "weak" | "none"
    score: float  # 0-1, the pooled overlap after the caps below
    reason: str


def _number_context_ok(claim_words: set[str], claim_text: str, evidence: str) -> bool:
    specific = _SPECIFIC_NUMBER.findall(claim_text)
    if not specific or not claim_words:
        return True
    tokens = [stem(t.strip(".-")) if t[:1].isalpha() else t.strip(".-") for t in re.findall(r"[a-z0-9.\-]+", evidence.lower())]
    for number in specific:
        best = 0.0
        for i, tok in enumerate(tokens):
            if tok == number:
                near = set(tokens[max(0, i - NUMBER_WINDOW): i + NUMBER_WINDOW + 1])
                best = max(best, len(claim_words & near) / len(claim_words))
        if best < MIN_NUMBER_CONTEXT:
            return False
    return True


def signals(claim_text: str, cited_texts: list[str]) -> Signals:
    """Compute the raw signals for a claim against the text of the evidence blocks it cites."""
    if not cited_texts:
        return Signals(cited=False, overlap=0.0, min_cite_overlap=0.0, numbers_ok=True, number_context_ok=True)
    bare = clean_claim(claim_text)
    words = content_words(bare)
    joined = " ".join(cited_texts)
    if not words:
        return Signals(cited=True, overlap=0.0, min_cite_overlap=0.0, numbers_ok=True, number_context_ok=True)
    overlap = len(words & content_words(joined)) / len(words)
    per_block = [len(words & content_words(t)) / len(words) for t in cited_texts]
    missing = tuple(n for n in _NUMBER.findall(bare) if n not in joined)
    return Signals(
        cited=True,
        overlap=overlap,
        min_cite_overlap=min(per_block),
        numbers_ok=not missing,
        number_context_ok=_number_context_ok(words, bare, joined),
        missing_numbers=missing,
    )


def label_from(sig: Signals, strong_min: float = STRONG_MIN, weak_min: float = WEAK_MIN) -> Grounding:
    """Turn raw signals into a label. Hard failures cap the label regardless of overlap."""
    if not sig.cited:
        return Grounding(NONE, 0.0, "the claim cites no evidence")
    if not sig.numbers_ok:
        return Grounding(NONE, min(sig.overlap, 0.3), f"number(s) not in the cited evidence: {', '.join(sig.missing_numbers)}")
    label = STRONG if sig.overlap >= strong_min else WEAK if sig.overlap >= weak_min else NONE
    score, reason = sig.overlap, f"{sig.overlap:.0%} of the claim's wording appears in the cited evidence"
    if not sig.number_context_ok and label == STRONG:
        label, reason = WEAK, "a number appears in the cited evidence but not near the claim's wording"
    if sig.min_cite_overlap < MIN_CITE_OVERLAP and label == STRONG:
        label, reason = WEAK, "one of the cited blocks shares little wording with the claim"
    return Grounding(label, score, reason)


def estimate(claim_text: str, cited_texts: list[str]) -> Grounding:
    return label_from(signals(claim_text, cited_texts))


def annotate(claims: Iterable, evidence: list) -> None:
    """Set claim.grounding / claim.grounding_score on each claim, in place, from the evidence it cites."""
    by_id = {e.citation_id: e.text for e in evidence}
    for claim in claims:
        cited = [by_id[i] for i in claim.citation_ids if i in by_id]
        g = estimate(claim.text, cited)
        claim.grounding, claim.grounding_score = g.label, round(g.score, 3)


def count_flagged(claims: list) -> int:
    """Claims the heuristic flags (label "none"): uncited, a number not in the source, or almost no shared wording."""
    return sum(getattr(c, "grounding", None) == NONE for c in claims)
