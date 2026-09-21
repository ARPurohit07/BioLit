"""Model-free tests for behaviour added late: client selection, the OpenRouter client, the number guard, bibliography
detection, table transcription helpers, the packing loop that once never terminated, and the shared chunker constructor."""
from __future__ import annotations

import threading

import pytest
import requests

from backend.app.api.query import _http_error
from backend.app.generation import openrouter_client as orc
from backend.app.generation.factory import build_llm_client
from backend.app.generation.rag_pipeline import RAGPipeline
from backend.app.ingestion.chunker import PageAwareChunker
from backend.app.ingestion.section_detector import is_reference_list
from backend.app.ingestion.structure import TableBlock, apply_transcriptions, rows_from_markdown
from backend.app.models.schemas import EvidenceItem
from backend.app.verification.claims import ClaimExtractor


# ------------------------------------------------------------------ which LLM, and does it leave the machine
def test_default_is_local_ollama():
    c = build_llm_client({}, "http://localhost:11434", "qwen2.5:3b", environ={})
    assert c.label == "ollama:qwen2.5:3b" and c.remote is False and c.warning is None


def test_an_ollama_cloud_model_is_reported_as_remote():
    assert build_llm_client({}, "h", "gpt-oss:120b-cloud", environ={}).remote is True


def test_openrouter_with_a_key_is_remote():
    cfg = {"generation": {"provider": "openrouter", "openrouter": {"model": "openai/gpt-oss-120b"}}}
    c = build_llm_client(cfg, "h", "qwen2.5:3b", environ={"OPENROUTER_KEY": "sk-test"})
    assert c.label == "openrouter:openai/gpt-oss-120b" and c.remote is True


def test_openrouter_without_a_key_falls_back_to_local_and_says_so():
    c = build_llm_client({"generation": {"provider": "openrouter"}}, "h", "qwen2.5:3b", environ={})
    assert c.remote is False and c.label.startswith("ollama:") and "OPENROUTER_KEY" in c.warning


def test_the_environment_can_override_the_configured_provider():
    c = build_llm_client({"generation": {"provider": "ollama"}}, "h", "m",
                         environ={"BIOLIT_GENERATION_PROVIDER": "openrouter", "OPENROUTER_KEY": "k"})
    assert c.remote is True


# ------------------------------------------------------------------ the OpenRouter client
class _Resp:
    def __init__(self, status=200, content="hi", usage=None):
        self.status_code, self._content, self._usage = status, content, usage or {"completion_tokens": 3}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code}")

    def json(self):
        return {"choices": [{"message": {"content": self._content}}], "usage": self._usage}


def test_the_client_sends_the_key_and_reasoning_effort_and_strips_the_reply(monkeypatch):
    seen = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        seen.update(url=url, payload=json, headers=headers)
        return _Resp(content="  answer  ")

    monkeypatch.setattr(orc.requests, "post", fake_post)
    text, stats = orc.OpenRouterClient("sk-test", "m", reasoning_effort="low").generate_with_stats("q", system="s")
    assert text == "answer" and stats["eval_count"] == 3
    assert seen["headers"]["Authorization"] == "Bearer sk-test" and seen["payload"]["reasoning"] == {"effort": "low"}
    assert seen["payload"]["messages"][0] == {"role": "system", "content": "s"}


def test_a_rate_limit_is_retried_but_payment_required_is_not(monkeypatch):
    monkeypatch.setattr(orc.time, "sleep", lambda s: None)
    replies = iter([_Resp(429), _Resp(200, "ok")])
    monkeypatch.setattr(orc.requests, "post", lambda *a, **k: next(replies))
    assert orc.OpenRouterClient("k", "m").generate("q") == "ok"

    calls = []
    monkeypatch.setattr(orc.requests, "post", lambda *a, **k: calls.append(1) or _Resp(402))
    with pytest.raises(requests.HTTPError):
        orc.OpenRouterClient("k", "m").generate("q")
    assert len(calls) == 1                                    # out of credit: retrying cannot help


def test_an_empty_reply_is_an_empty_string_not_a_crash(monkeypatch):
    monkeypatch.setattr(orc.requests, "post", lambda *a, **k: _Resp(content=None))
    assert orc.OpenRouterClient("k", "m").generate("q") == ""


def test_a_failing_language_model_is_a_502_and_a_bug_is_a_500():
    assert _http_error(requests.ConnectionError("down"), "Query failed").status_code == 502
    assert _http_error(ValueError("bug"), "Query failed").status_code == 500


# ------------------------------------------------------------------ the number guard
class _Client:
    def __init__(self, *replies):
        self.replies, self.prompts = list(replies), []

    def generate(self, prompt, system=None, temperature=0.1, **_):
        self.prompts.append(prompt)
        return self.replies.pop(0)


def _evidence():
    texts = ["The model reaches an AUC of 0.83 on the GDSC benchmark.", "Inference took 45 ms per sample on a CPU."]
    return [EvidenceItem(citation_id=i, chunk_id=f"c{i}", document_id="d", document_title="P", page_number=i,
                         section="Results", text=t) for i, t in enumerate(texts, 1)]


def _pipeline(*replies, config=None):
    client = _Client(*replies)
    return RAGPipeline(None, None, client, ClaimExtractor(), None, config or {}), client


def test_numbers_that_match_their_cited_block_cost_no_llm_call():
    pipeline, client = _pipeline("unused")
    answer = "The model reaches an AUC of 0.83 [1]."
    assert pipeline._ensure_values("u", "s", answer, _evidence()) == answer
    assert client.prompts == []


def test_a_number_from_a_different_block_triggers_one_retry_and_a_better_answer_replaces_it():
    pipeline, client = _pipeline("The model reaches an AUC of 0.83 [1].")
    out = pipeline._ensure_values("u", "s", "The model reaches an AUC of 45 [1].", _evidence())
    assert out == "The model reaches an AUC of 0.83 [1]." and len(client.prompts) == 1
    assert "45" in client.prompts[0]                          # the retry names the offending number


def test_a_retry_that_is_no_better_is_discarded_and_the_original_kept():
    pipeline, client = _pipeline("The model reaches an AUC of 0.99 and 12 ms [1].")
    original = "The model reaches an AUC of 45 [1]."
    assert pipeline._ensure_values("u", "s", original, _evidence()) == original
    assert len(client.prompts) == 1                           # one attempt, no loop


def test_the_guard_can_be_switched_off():
    pipeline, client = _pipeline("unused", config={"value_check": {"enabled": False}})
    answer = "The model reaches an AUC of 45 [1]."
    assert pipeline._ensure_values("u", "s", answer, _evidence()) == answer and client.prompts == []


# ------------------------------------------------------------------ bibliography detection
REFERENCES = ("[1] Jia Jia, Feng Zhu, Xiaohua Ma, Zhiwei W Cao, Yixue X Li, and Yu Zong Chen. Mechanisms of drug combinations: "
              "interaction and network perspectives. Nature reviews Drug discovery, 8(2):111-128, 2009. [2] Tracy Hampton. New "
              "insight on preventing egfr inhibitor induced adverse effects. JAMA, 319(4):321, 2018. [3] Emerging functions of the "
              "egfr in cancer. Molecular oncology, 12(1):3-20, 2018. [4] Yan Wang and Bo Xu. Deep learning for drug response. "
              "Bioinformatics, 35(7):1-9, 2019.")
PROSE = ("Prior work such as Smith et al., 2020 showed that graph neural networks improve drug response prediction, and Lee et al. "
         "(2021) reported that the approach can generalise to unseen cell lines. We extend these methods and evaluate them on "
         "three datasets, where our model outperforms the baselines by a wide margin.")


def test_a_reference_list_is_detected():
    assert is_reference_list(REFERENCES)


def test_prose_that_merely_cites_papers_is_never_dropped():
    assert not is_reference_list(PROSE)                       # a false positive would silently delete real evidence


def test_a_short_fragment_is_never_called_a_reference_list():
    assert not is_reference_list("Nature Communications, 14(1):2585, 2023.")


# ------------------------------------------------------------------ table transcription helpers
def test_markdown_becomes_rows_without_the_separator():
    md = "| A | B |\n| --- | :---: |\n| 1 | 2 |\n| 3 | 4 |"
    assert rows_from_markdown(md) == [["A", "B"], ["1", "2"], ["3", "4"]]


def _table():
    return TableBlock(1, "Table 1", "Table 1: t", [["old", "cells"], ["x", ""]], (0, 0, 1, 1), "lines")


def test_only_an_accepted_transcription_replaces_a_tables_rows():
    good = {"p1_t0": {"accepted": True, "markdown": "| A | B |\n| --- | --- |\n| 1 | 2 |"}}
    t = _table()
    apply_transcriptions([t], 1, good)
    assert t.rows == [["A", "B"], ["1", "2"]]
    for cache in ({"p1_t0": {"accepted": False, "markdown": "| A | B |\n| 1 | 2 |"}}, {}, {"p1_t0": {"accepted": True, "markdown": "| A |"}}):
        t = _table()
        apply_transcriptions([t], 1, cache)
        assert t.rows == [["old", "cells"], ["x", ""]]


# ------------------------------------------------------------------ chunking
def _finishes(fn, seconds=10):
    box = {}
    th = threading.Thread(target=lambda: box.setdefault("out", fn()), daemon=True)
    th.start()
    th.join(seconds)
    return not th.is_alive(), box.get("out")


def test_a_sentence_larger_than_the_budget_is_emitted_not_looped_on_forever():
    chunker = PageAwareChunker(chunk_size=50, chunk_overlap=20, min_chunk_tokens=5)
    big = "word " * 300
    done, out = _finishes(lambda: chunker._pack_page_fixed(["A short opening sentence.", big, "A closing sentence."]))
    assert done and any("closing sentence" in c for c in out) and any(len(c.split()) >= 300 for c in out)


def test_the_shared_constructor_defaults_to_fixed_size_chunks_and_honours_the_config():
    assert PageAwareChunker.from_config({}).semantic is False
    c = PageAwareChunker.from_config({"semantic": True, "max_chunk_tokens": 123, "chunk_size": 200})
    assert (c.semantic, c.max_chunk_tokens, c.chunk_size) == (True, 123, 200)
