"""Compare embedding models for the dense half of retrieval on the labelled questions, without touching the live index.

Each model embeds the same 1,476 indexed chunks into an in-memory index; BM25 and the reranker are the app's own. Reported per
half (tune / held-out) for dense alone, hybrid and hybrid + reranker, so a model is chosen on one half and confirmed on the other.

    python scripts/eval_embedders.py
"""
from __future__ import annotations

import json
import statistics as st
import sys
import time
from pathlib import Path

import faiss
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import eval_retrieval as er  # noqa: E402
from backend.app.retrieval.embeddings import EmbeddingModel  # noqa: E402
from backend.app.retrieval.hybrid import reciprocal_rank_fusion  # noqa: E402

MODELS = ["BAAI/bge-small-en-v1.5", "BAAI/bge-base-en-v1.5", "BAAI/bge-large-en-v1.5"]
EVAL = REPO_ROOT / "experiments" / "eval"


def load(name: str) -> list[dict]:
    return [json.loads(l) for l in (EVAL / name).read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> None:
    halves = {"tune": load("eval_set_dev.jsonl"), "held-out": load("eval_set_test.jsonl")}
    _, _, vs, bm, rr = er.build_components()
    chunks = vs._chunks
    texts = [c.text for c in chunks]
    out = {}
    for name in MODELS:
        t0 = time.time()
        model = EmbeddingModel(name, "auto", True)
        vecs = np.asarray(model.encode(texts, batch_size=32), dtype="float32")
        index = faiss.IndexFlatIP(vecs.shape[1])
        index.add(vecs)
        res = {}
        for half, items in halves.items():
            rows = {"dense": [], "hybrid": [], "hybrid+rerank": []}
            for item in items:
                q = np.asarray(model.encode([item["question"]]), dtype="float32")
                sc, ix = index.search(q, 20)
                dense = [(chunks[i], float(s)) for i, s in zip(ix[0], sc[0]) if i >= 0]
                fused = reciprocal_rank_fusion([dense, bm.search(item["question"], 20)], k=60)
                reranked = rr.rerank(item["question"], fused[:20], top_k=10)
                for arm, lst in (("dense", dense[:10]), ("hybrid", fused[:10]), ("hybrid+rerank", reranked)):
                    rows[arm].append(er.per_query(er.ranked_hits([c for c, _ in lst], item, "chunk")))
            res[half] = {arm: {m: round(st.fmean(r[m] for r in rs), 3) for m in ("recall@1", "recall@5", "recall@10", "mrr")}
                         for arm, rs in rows.items()}
        out[name] = res
        print(f"\n{name}  ({(time.time() - t0) / 60:.1f} min, dim {vecs.shape[1]})", flush=True)
        for half in halves:
            for arm in ("dense", "hybrid", "hybrid+rerank"):
                m = res[half][arm]
                print(f"  {half:<9}{arm:<15} R@1 {m['recall@1']:.2f}  R@5 {m['recall@5']:.2f}  R@10 {m['recall@10']:.2f}  MRR {m['mrr']:.2f}", flush=True)
        del model, index, vecs
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
    (EVAL / "embedder_compare.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
