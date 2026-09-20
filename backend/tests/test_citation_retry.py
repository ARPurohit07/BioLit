"""An answer with no [n] markers is retried once; the retry is kept only if it cites. Stubbed LLM: no models."""
from __future__ import annotations

from backend.app.generation.rag_pipeline import RAGPipeline
from backend.app.models.schemas import EvidenceItem
from backend.app.verification.claims import ClaimExtractor, is_abstention

UNCITED = "The bilinear attention network maps features from both channels into a common hidden space."
CITED = "The bilinear attention network maps features from both channels into a common hidden space [1]."


class _Client:
    def __init__(self, *responses: str):
        self.responses, self.prompts, self.temperatures = list(responses), [], []

    def generate(self, prompt, system=None, temperature=0.1, **_):
        self.prompts.append(prompt)
        self.temperatures.append(temperature)
        return self.responses.pop(0)


def _evidence(n: int = 2) -> list[EvidenceItem]:
    return [EvidenceItem(citation_id=i, chunk_id=f"c{i}", document_id="d", document_title="P", page_number=i,
                         section="Results", text="t") for i in range(1, n + 1)]


def _pipeline(*retry_responses: str, config: dict | None = None) -> tuple[RAGPipeline, _Client]:
    client = _Client(*retry_responses)
    return RAGPipeline(None, None, client, ClaimExtractor(), None, config or {}), client  # type: ignore[arg-type]


def test_uncited_answer_is_retried_and_a_cited_retry_is_kept():
    pipeline, client = _pipeline(CITED)
    out = pipeline._ensure_citations("USER PROMPT", "SYSTEM", UNCITED, _evidence())
    assert out == CITED
    assert len(client.prompts) == 1
    assert "no [n] citation markers" in client.prompts[0] and "USER PROMPT" in client.prompts[0]
    assert UNCITED in client.prompts[0]  # the retry sees what it is being asked to fix


def test_a_retry_that_still_does_not_cite_is_discarded():
    pipeline, client = _pipeline("Still nothing cited here at all.")
    assert pipeline._ensure_citations("u", "s", UNCITED, _evidence()) == UNCITED
    assert len(client.prompts) == 1  # one attempt only: no loop on a failing retry


def test_a_retry_citing_only_nonexistent_blocks_is_discarded():
    pipeline, _ = _pipeline("The bilinear attention network maps features into one space [9].")
    assert pipeline._ensure_citations("u", "s", UNCITED, _evidence(2)) == UNCITED


def test_an_already_cited_answer_is_never_retried():
    pipeline, client = _pipeline("must not be used")
    assert pipeline._ensure_citations("u", "s", CITED, _evidence()) == CITED
    assert client.prompts == []


def test_an_honest_insufficient_evidence_answer_is_left_alone():
    answer = "The evidence does not specify how the network handles missing data."
    pipeline, client = _pipeline("Invented [1].")
    assert pipeline._ensure_citations("u", "s", answer, _evidence()) == answer
    assert client.prompts == []


def test_retry_can_be_disabled_in_config():
    pipeline, client = _pipeline(CITED, config={"citation_retry": {"enabled": False}})
    assert pipeline._ensure_citations("u", "s", UNCITED, _evidence()) == UNCITED
    assert client.prompts == []


def test_no_evidence_means_no_retry():
    pipeline, client = _pipeline(CITED)
    assert pipeline._ensure_citations("u", "s", UNCITED, []) == UNCITED
    assert client.prompts == []


def test_echoed_retry_scaffolding_is_stripped_from_the_kept_answer():
    echoed = f"Your previous answer contained no citation markers.\n{CITED}"
    pipeline, _ = _pipeline(echoed)
    assert pipeline._ensure_citations("u", "s", UNCITED, _evidence()) == CITED


def test_generate_cited_wraps_generation_with_the_retry():
    # first call = the normal generation (uncited), second = the retry (cited)
    pipeline, client = _pipeline(UNCITED, CITED)
    assert pipeline._generate_cited("u", "s", _evidence()) == CITED
    assert len(client.prompts) == 2


def test_is_abstention_only_for_answers_with_nothing_to_cite():
    assert is_abstention("The evidence does not specify the dosing regimen.")
    assert is_abstention("LIMITATION: There is insufficient evidence to determine the mechanism.")
    assert not is_abstention("Drug X reduced tumor size in mice. The evidence does not specify dosing.")
    assert not is_abstention(UNCITED)
    assert not is_abstention("")


# ---------------------------------------------------------------- an answer that says nothing
from backend.app.generation.rag_pipeline import _NO_ANSWER, has_substance  # noqa: E402


def test_marker_only_replies_have_no_substance():
    assert not has_substance("[4]")
    assert not has_substance("[2][3]")
    assert not has_substance("SUPPORTED CLAIM: [1].")
    assert not has_substance("")
    assert has_substance("GEM reaches 0.856 on BACE [4].")
    assert has_substance("six [2]")


def test_a_marker_only_reply_is_retried_and_a_real_answer_is_kept():
    pipeline, client = _pipeline("GEM reaches 0.856 on the BACE dataset [4].")
    out = pipeline._ensure_substance("USER", "SYSTEM", "[4]")
    assert out == "GEM reaches 0.856 on the BACE dataset [4]."
    assert len(client.prompts) == 1 and "'[4]'" in client.prompts[0] and "USER" in client.prompts[0]


def test_a_reply_that_stays_empty_becomes_an_honest_no_answer_not_an_empty_string():
    pipeline, client = _pipeline("[4]", "[4]")
    assert pipeline._ensure_substance("u", "s", "[4]") == _NO_ANSWER
    assert len(client.prompts) == 2                      # two attempts, no loop
    assert is_abstention(_NO_ANSWER)                     # and it is not mistaken for a claim to verify


def test_an_answer_with_content_is_never_retried():
    pipeline, client = _pipeline("must not be used")
    assert pipeline._ensure_substance("u", "s", CITED) == CITED
    assert client.prompts == []


def test_ensure_citations_also_guards_against_empty_answers():
    pipeline, _ = _pipeline(CITED)
    assert pipeline._ensure_citations("u", "s", "[1]", _evidence()) == CITED


def test_no_prompt_contains_an_example_sentence_a_small_model_could_copy_into_its_answer():
    # "Method A improves recall over baseline B" was in the rules and the retry prompt, and the model sometimes appended it
    # to real answers as if it were a finding. Instructions describe the format; they must not contain a sentence to copy.
    from backend.app.generation import prompts

    pipeline, _ = _pipeline()
    for text in (prompts.CITATION_RULES, prompts.QA_RULES, pipeline._build_citation_retry_prompt("u", "a")):
        assert "Method A" not in text
