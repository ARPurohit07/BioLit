"""The reranker must survive a full GPU: smaller batch first, then CPU, never a failed request. Scripted OOMs, no models."""
from __future__ import annotations

import pytest
import torch

from backend.app.models.schemas import Chunk
from backend.app.retrieval.reranker import Reranker, _is_oom


def _oom() -> Exception:
    return torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 734.00 MiB.")


class _FakeModule:
    def __init__(self):
        self.calls: list[str] = []

    def to(self, device):
        self.calls.append(f"to({device})")
        return self

    def float(self):
        self.calls.append("float()")
        return self

    def half(self):
        self.calls.append("half()")
        return self


class _FakeCrossEncoder:
    """predict() replays scripted outcomes: an Exception is raised, anything else is returned as the scores."""

    def __init__(self, outcomes):
        self.outcomes, self.batch_sizes = list(outcomes), []
        self.model = _FakeModule()
        self._target_device = torch.device("cuda")

    def predict(self, pairs, batch_size=32, show_progress_bar=False):
        self.batch_sizes.append(batch_size)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _reranker(*outcomes, device="cuda", fallback=True, batch_size=16) -> tuple[Reranker, _FakeCrossEncoder]:
    r = object.__new__(Reranker)
    r.batch_size, r.fallback_to_cpu, r.fallback_reason = batch_size, fallback, None
    r.device, r.dtype = device, ("fp16" if device == "cuda" else "fp32")
    r.model = _FakeCrossEncoder(outcomes)
    return r, r.model


def _cands(n=4):
    return [(Chunk(chunk_id=f"c{i}", document_id="d", page_number=1, section="Results", text=f"text {i}", token_count=2), 0.0)
            for i in range(n)]


def test_normal_scoring_sorts_and_truncates_using_the_configured_batch_size():
    r, fake = _reranker([0.1, 0.9, 0.5, 0.7])
    out = r.rerank("q", _cands(), top_k=2)
    assert [c.chunk_id for c, _ in out] == ["c1", "c3"]
    assert fake.batch_sizes == [16]  # config batch_size is now honoured (it used to be ignored: library default 32)


def test_oom_is_retried_with_a_smaller_batch_and_stays_on_the_gpu():
    r, fake = _reranker(_oom(), [0.1, 0.9, 0.5, 0.7], batch_size=16)
    out = r.rerank("q", _cands(), top_k=1)
    assert out[0][0].chunk_id == "c1"
    assert fake.batch_sizes == [16, 4]
    assert r.device == "cuda" and r.fallback_reason is None


def test_repeated_oom_falls_back_to_the_cpu_permanently_and_still_answers():
    r, fake = _reranker(_oom(), _oom(), [0.1, 0.9, 0.5, 0.7], [0.2, 0.3, 0.1, 0.0])
    out = r.rerank("q", _cands(), top_k=1)
    assert out[0][0].chunk_id == "c1"                       # the request that hit the OOM still succeeded
    assert r.device == "cpu" and r.dtype == "fp32"
    assert "out of GPU memory while scoring" in r.fallback_reason
    assert fake._target_device == torch.device("cpu")        # predict() must stop moving the model back to the GPU
    assert "float()" in fake.model.calls and "to(cpu)" in fake.model.calls
    r.rerank("q", _cands(), top_k=1)                         # the next request goes straight to the CPU: one call, no OOM
    assert fake.batch_sizes == [16, 4, 16, 16]


def test_a_non_oom_error_is_not_swallowed():
    r, _ = _reranker(ValueError("bad input"))
    with pytest.raises(ValueError):
        r.rerank("q", _cands(), top_k=1)
    assert r.device == "cuda"


def test_fallback_can_be_disabled():
    r, fake = _reranker(_oom(), fallback=False)
    with pytest.raises(torch.cuda.OutOfMemoryError):
        r.rerank("q", _cands(), top_k=1)
    assert fake.batch_sizes == [16]


def test_no_candidates_means_no_model_call():
    r, fake = _reranker()
    assert r.rerank("q", [], top_k=5) == []
    assert fake.batch_sizes == []


def test_describe_reports_where_the_model_ended_up():
    r, _ = _reranker()
    assert r.describe() == "cuda (fp16)"
    r._fall_back_to_cpu("out of GPU memory placing the model on cuda")
    assert r.describe().startswith("cpu (fell back from GPU: out of GPU memory")
    cpu, _ = _reranker(device="cpu")
    assert cpu.describe() == "cpu"


def test_oom_detection_covers_the_typed_error_and_the_old_runtimeerror_form():
    assert _is_oom(_oom())
    assert _is_oom(RuntimeError("CUDA error: out of memory"))
    assert not _is_oom(RuntimeError("something else"))
    assert not _is_oom(ValueError("out of memory"))  # only RuntimeError-shaped errors count
