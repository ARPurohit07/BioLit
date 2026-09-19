"""High-Faithfulness regeneration must never make an answer less verifiable. Stubbed LLM/verifier: no models."""
from __future__ import annotations

from backend.app.generation.rag_pipeline import RAGPipeline
from backend.app.models.schemas import ClaimStatus, EvidenceItem
from backend.app.verification.claims import ClaimExtractor


class _Client:
    def __init__(self, revised: str):
        self.revised, self.calls = revised, 0

    def generate(self, prompt, system=None, temperature=0.1, **_):
        self.calls += 1
        return self.revised


class _Verifier:
    """A claim is SUPPORTED iff its text contains the word 'grounded'."""

    def verify_all(self, claims, evidence):
        for c in claims:
            c.status = ClaimStatus.SUPPORTED if (c.citation_ids and "grounded" in c.text) else ClaimStatus.UNSUPPORTED
        return claims


def _evidence():
    return [EvidenceItem(citation_id=i, chunk_id=f"c{i}", document_id="d", document_title="P", page_number=i,
                         section="Results", text="t") for i in (1, 2)]


def _pipeline(revised: str) -> tuple[RAGPipeline, _Client]:
    client = _Client(revised)
    return RAGPipeline(None, None, client, ClaimExtractor(), _Verifier(), {}), client


def _run(pipeline, answer):
    claims = pipeline.claim_extractor.extract(answer, _evidence())
    return pipeline._verify_and_regenerate("user", "system", answer, claims, _evidence(), max_attempts=2)


ORIGINAL = "This finding is grounded in the trial data [1]. Another statement is unverified in nature [2]."


def test_revision_that_drops_citations_is_discarded():
    pipeline, client = _pipeline("This finding is grounded in the trial data. Another statement is unverified in nature.")
    answer, claims = _run(pipeline, ORIGINAL)
    assert client.calls == 1                       # tried once, then stopped instead of looping
    assert "[1]" in answer and "[2]" in answer      # the ORIGINAL cited answer is what the user sees
    assert any(c.status == ClaimStatus.SUPPORTED for c in claims)
    assert "unsupported" in answer                  # ...with the unverified claim annotated


def test_revision_that_lowers_the_unsupported_rate_is_kept():
    pipeline, client = _pipeline("This finding is grounded in the trial data [1]. A second point is grounded too [2].")
    answer, claims = _run(pipeline, ORIGINAL)
    assert "grounded too" in answer
    assert all(c.status == ClaimStatus.SUPPORTED for c in claims)
    assert "unsupported" not in answer


def test_revision_that_does_not_improve_is_discarded():
    pipeline, client = _pipeline("Nothing here is checkable at all [1]. Still nothing checkable here [2].")
    answer, claims = _run(pipeline, ORIGINAL)
    assert "Nothing here is checkable" not in answer
    assert "grounded in the trial data" in answer


def test_fully_supported_answer_is_never_regenerated():
    pipeline, client = _pipeline("should not be used")
    answer, _ = _run(pipeline, "This finding is grounded in the trial data [1].")
    assert client.calls == 0
    assert answer == "This finding is grounded in the trial data [1]."
