"""LLM-based claim verification against retrieved evidence."""
from __future__ import annotations

import re

from backend.app.generation.ollama_client import OllamaClient
from backend.app.generation.prompts import build_claim_verification_prompt
from backend.app.models.schemas import Claim, ClaimStatus, EvidenceItem

_STATUS_RE = re.compile(r"STATUS:\s*(SUPPORTED|PARTIALLY_SUPPORTED|UNSUPPORTED|CONTRADICTED)", re.IGNORECASE)
_RATIONALE_RE = re.compile(r"RATIONALE:\s*(.+)", re.IGNORECASE | re.DOTALL)

_STATUS_KEYWORDS = [
    ClaimStatus.PARTIALLY_SUPPORTED,  # check before SUPPORTED since it contains "SUPPORTED"
    ClaimStatus.CONTRADICTED,
    ClaimStatus.UNSUPPORTED,
    ClaimStatus.SUPPORTED,
]


def _parse_verdict(response: str) -> tuple[ClaimStatus, str]:
    status_match = _STATUS_RE.search(response)
    rationale_match = _RATIONALE_RE.search(response)
    rationale = rationale_match.group(1).strip() if rationale_match else response.strip()

    if status_match:
        raw = status_match.group(1).upper()
        try:
            return ClaimStatus(raw), rationale
        except ValueError:
            pass

    upper = response.upper()
    for status in _STATUS_KEYWORDS:
        if status.value in upper:
            return status, rationale

    return ClaimStatus.UNSUPPORTED, rationale or "Could not parse verifier response; defaulting to UNSUPPORTED."


class ClaimVerifier:
    def __init__(self, ollama_client: OllamaClient):
        self.ollama_client = ollama_client

    def verify(self, claim: Claim, evidence: list[EvidenceItem]) -> Claim:
        if not claim.citation_ids:
            claim.status = ClaimStatus.UNSUPPORTED
            claim.verifier_rationale = "Claim has no citation marker; nothing to verify against."
            return claim

        matched = [e for e in evidence if e.citation_id in claim.citation_ids]
        if not matched:
            claim.status = ClaimStatus.UNSUPPORTED
            claim.verifier_rationale = "Cited evidence id(s) not found among retrieved evidence."
            return claim

        evidence_text = "\n\n".join(f"[{e.citation_id}] {e.text}" for e in matched)
        system, user = build_claim_verification_prompt(claim.text, evidence_text)

        try:
            response = self.ollama_client.generate(user, system=system, temperature=0.0)
        except Exception as exc:
            claim.status = ClaimStatus.UNSUPPORTED
            claim.verifier_rationale = f"Verifier call failed: {exc}"
            return claim

        status, rationale = _parse_verdict(response)
        claim.status = status
        claim.verifier_rationale = rationale
        return claim

    def verify_all(self, claims: list[Claim], evidence: list[EvidenceItem]) -> list[Claim]:
        return [self.verify(c, evidence) for c in claims]
