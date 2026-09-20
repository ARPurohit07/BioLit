"""Retrieval-layer tests using synthetic chunks and fake stores/models.

Heavy optional dependencies (faiss, rank_bm25, sentence-transformers) are
detected via try/except import and their tests skip gracefully when absent.
Live-model tests (real downloads) are opt-in via BIOLIT_TEST_LIVE_MODELS=1.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from backend.app.models.schemas import Chunk
from backend.app.retrieval.embeddings import _resolve_device
from backend.app.retrieval.hybrid import HybridRetriever, reciprocal_rank_fusion

try:
    import faiss  # noqa: F401
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

try:
    import rank_bm25  # noqa: F401
    HAS_RANK_BM25 = True
except ImportError:
    HAS_RANK_BM25 = False

try:
    import sentence_transformers  # noqa: F401
    HAS_SENTENCE_TRANSFORMERS = True
except Exception:
    HAS_SENTENCE_TRANSFORMERS = False

RUN_LIVE_MODEL_TESTS = os.environ.get("BIOLIT_TEST_LIVE_MODELS") == "1"


def make_chunk(chunk_id, document_id="doc1", page=1, section="Abstract", text="hello world"):
    return Chunk(chunk_id=chunk_id, document_id=document_id, page_number=page,
                 section=section, text=text, token_count=len(text.split()))


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

class TestReciprocalRankFusion:
    def test_fuses_and_dedupes(self):
        a, b, c = make_chunk("c1"), make_chunk("c2"), make_chunk("c3")
        list1 = [(a, 0.9), (b, 0.8)]
        list2 = [(b, 5.0), (c, 4.0)]
        fused = reciprocal_rank_fusion([list1, list2], k=60)
        ids = [chunk.chunk_id for chunk, _ in fused]
        assert ids == ["c2", "c1", "c3"]  # c2 appears in both lists -> highest fused score

    def test_empty_lists(self):
        assert reciprocal_rank_fusion([]) == []
        assert reciprocal_rank_fusion([[], []]) == []

    def test_single_list_preserves_order(self):
        a, b = make_chunk("c1"), make_chunk("c2")
        fused = reciprocal_rank_fusion([[(a, 1.0), (b, 0.5)]])
        assert [c.chunk_id for c, _ in fused] == ["c1", "c2"]


# ---------------------------------------------------------------------------
# HybridRetriever (fully stubbed dependencies)
# ---------------------------------------------------------------------------

class FakeVectorStore:
    def __init__(self, results):
        self._results = results

    def search(self, query_embedding, top_k, document_ids=None):
        results = self._results
        if document_ids is not None:
            results = [(c, s) for c, s in results if c.document_id in document_ids]
        return results[:top_k]


class FakeBM25:
    def __init__(self, results):
        self._results = results

    def search(self, query, top_k, document_ids=None):
        results = self._results
        if document_ids is not None:
            results = [(c, s) for c, s in results if c.document_id in document_ids]
        return results[:top_k]


class FakeEmbeddingModel:
    def encode(self, texts, batch_size=32):
        return np.zeros((len(texts), 4), dtype=np.float32)


CONFIG = {
    "modes": {
        "fast": {"use_bm25": False, "use_reranker": False, "top_k": 3},
        "balanced": {"use_bm25": True, "use_reranker": True, "top_k": 5},
    },
    "hybrid": {"dense_top_k": 5, "bm25_top_k": 5, "rrf_k": 60, "final_top_k": 4},
}


class TestHybridRetriever:
    def test_fast_mode_uses_dense_only(self):
        chunks = [make_chunk(f"c{i}") for i in range(5)]
        dense_results = [(c, 1.0 / (i + 1)) for i, c in enumerate(chunks)]
        retriever = HybridRetriever(FakeVectorStore(dense_results), FakeBM25([]), FakeEmbeddingModel(), CONFIG)

        results = retriever.retrieve("query", mode="fast")
        assert [c.chunk_id for c, _ in results] == ["c0", "c1", "c2"]

    def test_balanced_mode_fuses_dense_and_bm25(self):
        chunks = [make_chunk(f"c{i}") for i in range(5)]
        dense_results = [(chunks[0], 0.9), (chunks[1], 0.8)]
        bm25_results = [(chunks[1], 5.0), (chunks[2], 4.0)]
        retriever = HybridRetriever(FakeVectorStore(dense_results), FakeBM25(bm25_results),
                                     FakeEmbeddingModel(), CONFIG)

        results = retriever.retrieve("query", mode="balanced")
        ids = [c.chunk_id for c, _ in results]
        assert set(ids) == {"c0", "c1", "c2"}
        assert len(results) <= CONFIG["hybrid"]["final_top_k"]
        assert ids[0] == "c1"  # appears in both dense and bm25 lists

    def test_document_id_filter_propagates(self):
        chunks_doc1 = [make_chunk(f"a{i}", document_id="doc1") for i in range(3)]
        chunks_doc2 = [make_chunk(f"b{i}", document_id="doc2") for i in range(3)]
        dense_results = [(c, 1.0) for c in chunks_doc1 + chunks_doc2]
        retriever = HybridRetriever(FakeVectorStore(dense_results), FakeBM25([]), FakeEmbeddingModel(), CONFIG)

        results = retriever.retrieve("query", mode="fast", document_ids=["doc1"])
        assert results and all(c.document_id == "doc1" for c, _ in results)


# ---------------------------------------------------------------------------
# EmbeddingModel device resolution (no model download required)
# ---------------------------------------------------------------------------

def test_resolve_device_returns_valid_value():
    assert _resolve_device("auto") in ("cpu", "cuda")
    assert _resolve_device("cpu") == "cpu"
    assert _resolve_device("cuda") == "cuda"  # explicit device passed through untouched


# ---------------------------------------------------------------------------
# FAISSVectorStore (requires faiss)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_FAISS, reason="faiss not installed")
class TestFAISSVectorStore:
    def test_add_search_save_load(self, tmp_path):
        from backend.app.retrieval.vector_store import FAISSVectorStore

        dim = 4
        store = FAISSVectorStore(dim=dim, index_path=str(tmp_path / "faiss.index"),
                                  metadata_path=str(tmp_path / "meta.jsonl"))
        chunks = [make_chunk(f"c{i}", document_id="doc1" if i < 2 else "doc2") for i in range(4)]
        embeddings = np.eye(4, dtype=np.float32)
        store.add(chunks, embeddings)
        assert len(store) == 4

        results = store.search(embeddings[0], top_k=2)
        assert results[0][0].chunk_id == "c0"

        filtered = store.search(embeddings[0], top_k=4, document_ids=["doc2"])
        assert all(c.document_id == "doc2" for c, _ in filtered)

        store.save()
        store2 = FAISSVectorStore(dim=dim, index_path=str(tmp_path / "faiss.index"),
                                   metadata_path=str(tmp_path / "meta.jsonl"))
        store2.load()
        assert len(store2) == 4
        assert store2.search(embeddings[0], top_k=1)[0][0].chunk_id == "c0"

    def test_load_with_missing_files_is_empty(self, tmp_path):
        from backend.app.retrieval.vector_store import FAISSVectorStore

        store = FAISSVectorStore(dim=4, index_path=str(tmp_path / "missing.index"),
                                  metadata_path=str(tmp_path / "missing.jsonl"))
        store.load()
        assert len(store) == 0

    def test_remove_document(self, tmp_path):
        from backend.app.retrieval.vector_store import FAISSVectorStore

        store = FAISSVectorStore(dim=3, index_path=str(tmp_path / "f.index"),
                                  metadata_path=str(tmp_path / "m.jsonl"))
        chunks = [make_chunk("x1", document_id="docA"), make_chunk("x2", document_id="docB")]
        embeddings = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
        store.add(chunks, embeddings)

        store.remove_document("docA")
        assert len(store) == 1
        remaining = store.search(np.array([0, 1, 0], dtype=np.float32), top_k=5)
        assert remaining[0][0].chunk_id == "x2"


# ---------------------------------------------------------------------------
# BM25Index (requires rank_bm25)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_RANK_BM25, reason="rank_bm25 not installed")
class TestBM25Index:
    def test_add_search_save_load(self, tmp_path):
        from backend.app.retrieval.bm25 import BM25Index

        index = BM25Index(index_path=str(tmp_path / "bm25.pkl"))
        chunks = [
            make_chunk("c1", text="the cat sat on the mat"),
            make_chunk("c2", text="dogs are loyal animals"),
        ]
        index.add(chunks)
        results = index.search("cat mat", top_k=2)
        assert results[0][0].chunk_id == "c1"

        index.save()
        index2 = BM25Index(index_path=str(tmp_path / "bm25.pkl"))
        index2.load()
        assert len(index2) == 2

    def test_document_id_filter(self):
        from backend.app.retrieval.bm25 import BM25Index

        index = BM25Index(index_path="unused.pkl")
        chunks = [
            make_chunk("c1", document_id="doc1", text="cancer treatment therapy"),
            make_chunk("c2", document_id="doc2", text="cancer treatment therapy"),
        ]
        index.add(chunks)
        results = index.search("cancer", top_k=5, document_ids=["doc2"])
        assert results and all(c.document_id == "doc2" for c, _ in results)

    def test_remove_document(self):
        from backend.app.retrieval.bm25 import BM25Index

        index = BM25Index(index_path="unused.pkl")
        chunks = [make_chunk("c1", document_id="doc1"), make_chunk("c2", document_id="doc2")]
        index.add(chunks)
        index.remove_document("doc1")
        assert len(index) == 1


# ---------------------------------------------------------------------------
# Live model tests (opt-in only — require network + a real download)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not RUN_LIVE_MODEL_TESTS, reason="set BIOLIT_TEST_LIVE_MODELS=1 to run (network + model download)")
class TestEmbeddingModelLive:
    def test_encode_shape_and_normalization(self):
        from backend.app.retrieval.embeddings import EmbeddingModel

        model = EmbeddingModel("BAAI/bge-small-en-v1.5", device="cpu")
        vecs = model.encode(["hello world", "biomedical research"])
        assert vecs.shape[0] == 2
        assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-3)


@pytest.mark.skipif(not RUN_LIVE_MODEL_TESTS, reason="set BIOLIT_TEST_LIVE_MODELS=1 to run (network + model download)")
class TestRerankerLive:
    def test_rerank_orders_by_relevance(self):
        from backend.app.retrieval.reranker import Reranker

        reranker = Reranker("BAAI/bge-reranker-base", device="cpu")
        candidates = [
            (make_chunk("c1", text="unrelated text about cooking recipes"), 0.5),
            (make_chunk("c2", text="deep learning models for cancer diagnosis"), 0.4),
        ]
        results = reranker.rerank("cancer diagnosis with deep learning", candidates, top_k=2)
        assert results[0][0].chunk_id == "c2"


# ---------------------------------------------------------------- weighted rank fusion
def _chunk(cid):
    from backend.app.models.schemas import Chunk
    return Chunk(chunk_id=cid, document_id="d", page_number=1, section="Results", text=cid, token_count=3)


def test_weights_let_one_ranker_outrank_the_other():
    from backend.app.retrieval.hybrid import reciprocal_rank_fusion

    dense = [(_chunk("a"), 0.9), (_chunk("b"), 0.8)]
    sparse = [(_chunk("b"), 5.0), (_chunk("a"), 4.0)]

    equal = [c.chunk_id for c, _ in reciprocal_rank_fusion([dense, sparse], k=60)]
    assert set(equal) == {"a", "b"}                                   # a tie, decided by insertion order

    bm25_heavy = [c.chunk_id for c, _ in reciprocal_rank_fusion([dense, sparse], k=60, weights=[1.0, 2.0])]
    assert bm25_heavy[0] == "b"                                       # the sparse list's top result wins
    dense_heavy = [c.chunk_id for c, _ in reciprocal_rank_fusion([dense, sparse], k=60, weights=[2.0, 1.0])]
    assert dense_heavy[0] == "a"


def test_omitted_weights_behave_exactly_like_the_unweighted_default():
    from backend.app.retrieval.hybrid import reciprocal_rank_fusion

    lists = [[(_chunk("a"), 1.0), (_chunk("c"), 0.5)], [(_chunk("c"), 2.0), (_chunk("b"), 1.0)]]
    assert [(c.chunk_id, round(s, 6)) for c, s in reciprocal_rank_fusion(lists, k=60)] == \
           [(c.chunk_id, round(s, 6)) for c, s in reciprocal_rank_fusion(lists, k=60, weights=[1.0, 1.0])]
