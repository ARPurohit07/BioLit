"""Search retrieval settings against the labelled question set. No LLM: a full sweep takes minutes.

Settings swept: the weight given to the dense and BM25 rankings when they are fused, how many fused candidates the
reranker re-scores, and how deep each retriever goes.

Tuned and reported on different questions. The sweep picks its winner on half the questions (odd positions) and the
score that gets reported is measured on the other half, so a setting that only suits the tuning half is not mistaken
for an improvement.

    python scripts/sweep_retrieval.py                 # writes experiments/eval/retrieval_sweep.json
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

sys.path.insert(0, str(REPO_ROOT / "scripts"))
import eval_retrieval as er  # noqa: E402

from backend.app.retrieval.hybrid import reciprocal_rank_fusion  # noqa: E402

OUT = REPO_ROOT / "experiments" / "eval" / "retrieval_sweep.json"

# (dense_weight, bm25_weight, candidate_k, retriever_depth)
GRID = [
    (1.0, 1.0, 20, 20),    # what the app ships today
    (1.0, 1.5, 20, 20),
    (1.0, 2.0, 20, 20),
    (0.5, 1.0, 20, 20),
    (1.0, 1.5, 40, 40),
    (1.0, 2.0, 40, 40),
    (1.0, 1.0, 40, 40),
    (1.0, 1.5, 60, 60),
]


def evaluate(items, emb, vs, bm, rr, dense_w, bm25_w, cand_k, depth) -> dict:
    """Recall@k / MRR for one setting, at chunk level, after reranking."""
    rows = []
    for item in items:
        q = emb.encode([item["question"]])[0]
        dense = vs.search(q, depth)
        sparse = bm.search(item["question"], depth)
        fused = reciprocal_rank_fusion([dense, sparse], k=60, weights=[dense_w, bm25_w])
        reranked = rr.rerank(item["question"], fused[:cand_k], top_k=10)
        chunks = [c for c, _ in reranked]
        rows.append(er.per_query(er.ranked_hits(chunks, item, "chunk")))
    return {m: round(st.fmean(r[m] for r in rows), 4) for m in rows[0]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    items = [json.loads(l) for l in er.EVAL_SET.read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit:
        items = items[: args.limit]
    tune, test = items[1::2], items[0::2]
    _settings, emb, vs, bm, rr = er.build_components()
    print(f"{len(items)} questions: {len(tune)} to choose the setting, {len(test)} held out | reranker on {rr.describe()}", flush=True)

    results = []
    t0 = time.time()
    for dense_w, bm25_w, cand_k, depth in GRID:
        m = evaluate(tune, emb, vs, bm, rr, dense_w, bm25_w, cand_k, depth)
        results.append({"dense_weight": dense_w, "bm25_weight": bm25_w, "candidate_k": cand_k, "depth": depth, "tune": m})
        print(f"  dense={dense_w} bm25={bm25_w} cand_k={cand_k} depth={depth}  "
              f"R@1 {m['recall@1']:.2f} R@5 {m['recall@5']:.2f} R@10 {m['recall@10']:.2f} MRR {m['mrr']:.2f}"
              f"  | {(time.time() - t0) / 60:.1f} min", flush=True)

    best = max(results, key=lambda r: (r["tune"]["recall@5"], r["tune"]["mrr"]))
    shipped = results[0]
    print(f"\nbest on the tuning half: dense={best['dense_weight']} bm25={best['bm25_weight']} "
          f"cand_k={best['candidate_k']} depth={best['depth']}", flush=True)

    held = {}
    for name, cfg in (("shipped", shipped), ("best", best)):
        held[name] = evaluate(test, emb, vs, bm, rr, cfg["dense_weight"], cfg["bm25_weight"], cfg["candidate_k"], cfg["depth"])
        print(f"HELD-OUT {name:<8} R@1 {held[name]['recall@1']:.2f} R@5 {held[name]['recall@5']:.2f} "
              f"R@10 {held[name]['recall@10']:.2f} MRR {held[name]['mrr']:.2f} nDCG {held[name]['ndcg@10']:.2f}", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "n_questions": len(items), "n_tune": len(tune), "n_held_out": len(test),
        "grid": results, "chosen": {k: best[k] for k in ("dense_weight", "bm25_weight", "candidate_k", "depth")},
        "held_out": held,
        "note": "Chunk level, after reranking. The setting was chosen on the tuning half; held_out is measured on the "
                "other half. Recall with one labelled chunk per question equals hit rate.",
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT.relative_to(REPO_ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
