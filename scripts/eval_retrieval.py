"""Score retrieval against the labelled evaluation set (experiments/eval/eval_set.jsonl). No LLM involved.

Four strategies are compared so the value of each component is visible, not assumed:
    dense           bge-small embeddings, FAISS
    bm25            BM25 over the same chunks
    hybrid          dense + BM25 fused with reciprocal rank fusion
    hybrid+rerank   top-20 fused candidates re-scored by the cross-encoder (what Balanced mode uses)

Metrics (binary relevance, one labelled source chunk per question):
    Recall@k / Hit@k   is the labelled chunk in the top k?   (k = 1, 3, 5, 10)
    MRR                mean reciprocal rank of the labelled chunk
    nDCG@10            rank-discounted gain
  reported at two levels:
    chunk level        the exact source chunk (strict: another chunk that also answers counts as a miss)
    paper level        any chunk from the source paper (did we at least find the right paper?)
  and broken down by question type (text / table / figure) and by difficulty (questions that share many words with
  their source passage are "easy" for lexical search; the low-overlap half is "hard"). 95% bootstrap intervals are
  given because 150 questions is a small sample.

    python scripts/eval_retrieval.py                     # audited set; writes experiments/eval/retrieval_eval.json
"""
from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.app.config.settings import get_settings  # noqa: E402
from backend.app.evaluation.retrieval_metrics import _mrr, _ndcg_at_k  # noqa: E402
from backend.app.ingestion.chunker import count_tokens  # noqa: E402
from backend.app.retrieval.bm25 import BM25Index  # noqa: E402
from backend.app.retrieval.embeddings import EmbeddingModel  # noqa: E402
from backend.app.retrieval.hybrid import reciprocal_rank_fusion  # noqa: E402
from backend.app.retrieval.reranker import Reranker  # noqa: E402
from backend.app.retrieval.vector_store import FAISSVectorStore  # noqa: E402

EVAL_SET = REPO_ROOT / "experiments" / "eval" / "eval_set_final.jsonl"
OUT = REPO_ROOT / "experiments" / "eval" / "retrieval_eval.json"
KS = (1, 3, 5, 10)

# Chunk-id-free metrics: comparable across chunking strategies since they never look at chunk_id.
_STOPWORDS = {
    "the", "and", "for", "are", "was", "were", "with", "that", "this", "these", "those", "from",
    "have", "has", "had", "not", "but", "can", "which", "when", "where", "who", "whom", "how",
    "what", "why", "does", "did", "using", "used", "use", "into", "onto", "than", "then", "them",
    "their", "there", "here", "its", "his", "her", "our", "your", "you", "they", "she", "him",
    "been", "being", "will", "would", "could", "should", "also", "such", "over", "under", "each",
    "any", "all", "both", "more", "most", "some", "one", "two", "via", "per", "out", "about",
}
_WORD_RE = re.compile(r"[a-z0-9]+")
_NUM_RE = re.compile(r"\d+\.?\d*")


def _content_words(text: str) -> set[str]:
    return {w for w in _WORD_RE.findall(text.lower()) if len(w) >= 3 and w not in _STOPWORDS}


def granularity_neutral_score(chunks: list, item: dict) -> dict:
    """Score the top-5 retrieved chunks against the reference answer, without touching chunk_id."""
    top5 = chunks[:5]
    context = "\n".join(c.text for c in top5)
    context_words = set(_WORD_RE.findall(context.lower()))

    words = _content_words(item["reference_answer"])
    answer_coverage = (sum(1 for w in words if w in context_words) / len(words)) if words else 1.0

    ref_nums = _NUM_RE.findall(item["reference_answer"])
    number_covered = None
    if ref_nums:
        number_covered = all(re.search(rf"\b{re.escape(n)}\b", context) for n in ref_nums)

    return {
        "answer_coverage": answer_coverage,
        "number_covered": number_covered,
        "context_tokens": count_tokens(context),
    }


def summarise_granularity_neutral(gn_rows: list[dict]) -> dict:
    coverages = [r["answer_coverage"] for r in gn_rows]
    num_rows = [r["number_covered"] for r in gn_rows if r["number_covered"] is not None]
    tokens = [r["context_tokens"] for r in gn_rows]
    return {
        "answer_coverage": round(statistics.fmean(coverages), 4),
        "answer_covered@0.8": round(sum(1 for c in coverages if c >= 0.8) / len(coverages), 4),
        "number_coverage": {"mean": round(statistics.fmean(num_rows), 4) if num_rows else None, "n": len(num_rows)},
        "context_tokens": round(statistics.fmean(tokens), 1),
    }


def build_components():
    s = get_settings()
    emb_cfg, rr_cfg = s.models_config.get("embedding_model", {}), s.models_config.get("reranker", {})
    vs_cfg, bm_cfg = s.retrieval_config.get("vector_store", {}), s.retrieval_config.get("bm25", {})
    emb = EmbeddingModel(emb_cfg.get("name", "BAAI/bge-small-en-v1.5"), emb_cfg.get("device", "auto"), emb_cfg.get("normalize", True))
    vs = FAISSVectorStore(emb_cfg.get("dim", 384), str(s.repo_root / vs_cfg.get("index_path", "data/index/faiss.index")),
                          str(s.repo_root / vs_cfg.get("metadata_path", "data/index/chunk_metadata.jsonl")))
    vs.load()
    bm = BM25Index(str(s.repo_root / bm_cfg.get("index_path", "data/index/bm25_index.pkl")), bm_cfg.get("k1", 1.5), bm_cfg.get("b", 0.75))
    bm.load()
    rr = Reranker(rr_cfg.get("name", "BAAI/bge-reranker-base"), rr_cfg.get("device", "auto"), rr_cfg.get("max_length", 512),
                  rr_cfg.get("batch_size", 16), rr_cfg.get("half_precision", True), rr_cfg.get("fallback_to_cpu", True))
    return s, emb, vs, bm, rr


def rank_all(question: str, emb, vs, bm, rr, cfg: dict) -> dict[str, tuple[list, float]]:
    """Ranked chunk lists (best first, up to 10) for every strategy, with each strategy's wall time in ms."""
    h = cfg.get("hybrid", {})
    t = time.perf_counter()
    q = emb.encode([question])[0]
    embed_ms = (time.perf_counter() - t) * 1000
    out = {}

    t = time.perf_counter()
    dense = vs.search(q, 20)
    out["dense"] = ([c for c, _ in dense[:10]], embed_ms + (time.perf_counter() - t) * 1000)

    t = time.perf_counter()
    sparse = bm.search(question, 20)
    out["bm25"] = ([c for c, _ in sparse[:10]], (time.perf_counter() - t) * 1000)

    t = time.perf_counter()
    fused = reciprocal_rank_fusion([dense, sparse], k=h.get("rrf_k", 60))
    fuse_ms = (time.perf_counter() - t) * 1000
    out["hybrid"] = ([c for c, _ in fused[:10]], out["dense"][1] + out["bm25"][1] + fuse_ms)

    t = time.perf_counter()
    reranked = rr.rerank(question, fused[: cfg.get("reranker", {}).get("candidate_k", 20)], top_k=10)
    out["hybrid+rerank"] = ([c for c, _ in reranked], out["hybrid"][1] + (time.perf_counter() - t) * 1000)
    return out


def ranked_hits(ranked_chunks: list, item: dict, level: str) -> list[bool]:
    if level == "chunk":
        relevant = set(item["chunk_ids"]) if item.get("chunk_ids") else {item["chunk_id"]}
        return [c.chunk_id in relevant for c in ranked_chunks]
    seen, hits = set(), []                      # paper level: score each paper once, at its best-ranked chunk
    for c in ranked_chunks:
        if c.document_id in seen:
            continue
        seen.add(c.document_id)
        hits.append(c.document_id == item["document_id"])
    return hits


def per_query(hits: list[bool]) -> dict[str, float]:
    row = {f"recall@{k}": float(any(hits[:k])) for k in KS}            # one relevant item: recall == hit rate
    row["mrr"] = _mrr(hits)
    row["ndcg@10"] = _ndcg_at_k(hits, 10)
    return row


def summarise(rows: list[dict[str, float]], seed: int = 7) -> dict:
    if not rows:
        return {"n": 0}
    rng = random.Random(seed)
    out = {"n": len(rows)}
    for metric in rows[0]:
        vals = [r[metric] for r in rows]
        boots = sorted(statistics.fmean(rng.choices(vals, k=len(vals))) for _ in range(1000))
        out[metric] = {"mean": round(statistics.fmean(vals), 4), "ci95": [round(boots[25], 4), round(boots[974], 4)]}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=0, help="only the first N questions (smoke test)")
    ap.add_argument("--eval-set", type=Path, default=EVAL_SET, help="labelled questions (default: the audited final set)")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    items = [json.loads(l) for l in args.eval_set.read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit:
        items = items[: args.limit]
    settings, emb, vs, bm, rr = build_components()
    print(f"{len(items)} questions | corpus: {len(vs)} chunks | reranker on {rr.describe()}", flush=True)

    overlap_median = statistics.median(i["question_chunk_overlap"] for i in items)
    rows: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))      # arm -> group -> [per-query dict]
    latency: dict[str, list[float]] = defaultdict(list)
    gn_rows: dict[str, list[dict]] = defaultdict(list)
    for n, item in enumerate(items, 1):
        ranked = rank_all(item["question"], emb, vs, bm, rr, settings.retrieval_config)
        for arm, (chunks, ms) in ranked.items():
            latency[arm].append(ms)
            gn_rows[arm].append(granularity_neutral_score(chunks, item))
            for level in ("chunk", "paper"):
                row = per_query(ranked_hits(chunks, item, level))
                rows[arm][f"{level}/all"].append(row)
                if level == "chunk":
                    rows[arm][f"chunk/type={item['type']}"].append(row)
                    rows[arm][f"chunk/{'hard' if item['question_chunk_overlap'] <= overlap_median else 'easy'}"].append(row)
        if n % 25 == 0:
            print(f"[{n}/{len(items)}]", flush=True)

    arms = {}
    for arm in rows:
        arms[arm] = {group: summarise(r) for group, r in rows[arm].items()}
        arms[arm]["latency_ms_median"] = round(statistics.median(latency[arm]), 1)
        arms[arm]["granularity_neutral"] = summarise_granularity_neutral(gn_rows[arm])

    balanced = arms["hybrid+rerank"]["chunk/all"]
    result = {
        "eval_set": {"n": len(items), "by_type": {t: sum(i["type"] == t for i in items) for t in ("text", "table", "figure")},
                     "papers": len({i["document_id"] for i in items}), "difficulty_split_at_overlap": overlap_median},
        "corpus_chunks": len(vs),
        "arms": arms,
        # The shape the /api/evaluation endpoint already reads: Balanced mode (hybrid + reranker), chunk level.
        "retrieval_metrics": {
            "recall_at_1": balanced["recall@1"]["mean"], "recall_at_3": balanced["recall@3"]["mean"],
            "recall_at_5": balanced["recall@5"]["mean"], "recall_at_10": balanced["recall@10"]["mean"],
            "mrr": balanced["mrr"]["mean"], "ndcg_at_10": balanced["ndcg@10"]["mean"], "num_queries": len(items),
        },
        "caveats": [
            "Each question has one labelled source chunk; another chunk that also answers counts as a miss, so chunk-level "
            "recall is a lower bound (paper-level recall is the looser view).",
            "Questions were written by a language model from the very chunk they are scored against, so they share "
            "vocabulary with it; this favours lexical search. The 'hard' half (below-median word overlap) is the fairer test.",
            "With ~125 questions the 95% bootstrap intervals are wide; differences smaller than the intervals are noise.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"\n{'strategy':<15}{'R@1':>7}{'R@5':>7}{'R@10':>7}{'MRR':>7}{'nDCG@10':>9}   paper R@1 / R@5   median ms")
    for arm in ("bm25", "dense", "hybrid", "hybrid+rerank"):
        a, p = arms[arm]["chunk/all"], arms[arm]["paper/all"]
        print(f"{arm:<15}{a['recall@1']['mean']:>7.2f}{a['recall@5']['mean']:>7.2f}{a['recall@10']['mean']:>7.2f}"
              f"{a['mrr']['mean']:>7.2f}{a['ndcg@10']['mean']:>9.2f}   {p['recall@1']['mean']:.2f} / {p['recall@5']['mean']:.2f}"
              f"          {arms[arm]['latency_ms_median']:.0f}")

    print(f"\n{'strategy':<15}{'ans_cov':>9}{'cov@0.8':>9}{'num_cov':>9}{'ctx_tok':>9}")
    for arm in ("bm25", "dense", "hybrid", "hybrid+rerank"):
        g = arms[arm]["granularity_neutral"]
        num_cov = "n/a" if g["number_coverage"]["mean"] is None else f"{g['number_coverage']['mean']:.2f}"
        print(f"{arm:<15}{g['answer_coverage']:>9.2f}{g['answer_covered@0.8']:>9.2f}{num_cov:>9}{g['context_tokens']:>9.0f}")

    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
