"""Re-point eval_set.jsonl labels at a new chunk index after re-chunking the corpus.

Re-chunking changes every chunk_id, which would silently invalidate the 144 labelled questions in
experiments/eval/eval_set.jsonl. This script recovers each label's original source text from a snapshot of the OLD
chunk_metadata.jsonl and finds the chunk(s) in the NEW chunk_metadata.jsonl that cover the same text, so retrieval
can keep being scored against the same source passages.

Method: for each eval item, restrict candidates to chunks from the same document_id (and, when available, the same
page), then rank them by token overlap coefficient against the old chunk's text (see overlap_coefficient() for why
this is used instead of F1). The single best match is written to chunk_id; every new chunk whose overlap clears
--threshold is written to chunk_ids, so a chunk the new strategy split into several pieces is still fully credited.
The original id is kept in chunk_id_old.

    python scripts/remap_eval_labels.py                  # writes experiments/eval/eval_set_remapped.jsonl

Self-test (old snapshot == current index): every item must remap to its own chunk_id with overlap 1.0.
    cp data/index/chunk_metadata.jsonl experiments/eval/chunk_metadata_old.jsonl
    python scripts/remap_eval_labels.py --new data/index/chunk_metadata.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

EVAL_SET = REPO_ROOT / "experiments" / "eval" / "eval_set.jsonl"
OLD_META = REPO_ROOT / "experiments" / "eval" / "chunk_metadata_old.jsonl"
NEW_META = REPO_ROOT / "data" / "index" / "chunk_metadata.jsonl"
OUT = REPO_ROOT / "experiments" / "eval" / "eval_set_remapped.jsonl"
# This corpus repeats page headers/footers verbatim in every chunk of a page, which alone can push containment past
# 0.5 for two otherwise unrelated same-page chunks; 0.8 was chosen (see remap_eval_labels self-test in the module
# docstring) as high enough to reject that boilerplate-driven overlap while still catching genuine sub-chunk splits,
# whose containment is at or very near 1.0 because the re-chunker only moves boundaries, not the text itself.
THRESHOLD = 0.8

_TOKEN = re.compile(r"\w+")


def tokens(text: str) -> Counter:
    return Counter(t.lower() for t in _TOKEN.findall(text))


def overlap_scores(a: Counter, b: Counter) -> tuple[float, float]:
    """(containment, jaccard) for two token bags.

    Containment (Szymkiewicz-Simpson: |A n B| / min(|A|, |B|)) is the primary ranking score, picked over F1 because
    re-chunking can split one old chunk into several small new ones, or merge several old chunks into one big new
    one. Either way the smaller bag of tokens is (almost) fully contained in the larger, but the two chunks' sizes
    differ a lot -- which drags F1 down even for a perfect sub-chunk match. Normalising by the smaller side instead
    keeps a full split fragment, or a full merge, scoring near 1.0.

    Containment alone cannot break ties, though: a short chunk (e.g. a two-word caption fragment) is trivially "1.0
    contained" in almost any longer chunk that happens to share those few tokens, tying with the chunk it actually
    came from. Jaccard (|A n B| / |A u B|) is used as the tiebreak because it falls sharply as the size mismatch
    grows, so it favours the same-sized, truly matching chunk over a coincidental subset match.
    """
    if not a or not b:
        return 0.0, 0.0
    intersection = sum((a & b).values())
    if intersection == 0:
        return 0.0, 0.0
    containment = intersection / min(sum(a.values()), sum(b.values()))
    jaccard = intersection / sum((a | b).values())
    return containment, jaccard


def load_metadata(path: Path) -> dict[str, dict]:
    by_id: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            by_id[row["chunk_id"]] = row
    return by_id


def group_by_document(chunks: dict[str, dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    for c in chunks.values():
        out[c["document_id"]].append(c)
    return out


def best_matches(old_text: str, candidates: list[dict]) -> list[tuple[str, float]]:
    old_tokens = tokens(old_text)
    scored = [(c["chunk_id"], *overlap_scores(old_tokens, tokens(c["text"]))) for c in candidates]
    scored.sort(key=lambda row: (-row[1], -row[2]))
    return [(chunk_id, containment) for chunk_id, containment, _ in scored]


def remap_item(item: dict, old_by_id: dict, new_by_doc: dict, threshold: float) -> tuple[dict, str]:
    """Return (updated item, status), status in {"confident", "ambiguous", "failed"}."""
    out = dict(item)
    out["chunk_id_old"] = item["chunk_id"]

    old_chunk = old_by_id.get(item["chunk_id"])
    doc_candidates = new_by_doc.get(item["document_id"], [])
    if old_chunk is None or not doc_candidates:
        out["chunk_ids"] = []
        return out, "failed"

    same_page = [c for c in doc_candidates if c.get("page_number") == old_chunk.get("page_number")]
    scored = best_matches(old_chunk["text"], same_page or doc_candidates)
    if not scored:
        out["chunk_ids"] = []
        return out, "failed"

    best_id, best_score = scored[0]
    out["chunk_id"] = best_id
    out["chunk_ids"] = [cid for cid, score in scored if score >= threshold] or [best_id]
    out["remap_overlap"] = round(best_score, 4)
    return out, "confident" if best_score >= threshold else "ambiguous"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-set", type=Path, default=EVAL_SET, help="labelled questions to remap")
    ap.add_argument("--old", type=Path, default=OLD_META, help="snapshot of chunk_metadata.jsonl taken before re-chunking")
    ap.add_argument("--new", type=Path, default=NEW_META, help="chunk_metadata.jsonl produced by the new chunking run")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--threshold", type=float, default=THRESHOLD, help="overlap coefficient below which a match is ambiguous")
    args = ap.parse_args()

    items = [json.loads(l) for l in args.eval_set.read_text(encoding="utf-8").splitlines() if l.strip()]
    old_by_id = load_metadata(args.old)
    new_by_doc = group_by_document(load_metadata(args.new))

    remapped, counts, low_confidence = [], Counter(), []
    for item in items:
        updated, status = remap_item(item, old_by_id, new_by_doc, args.threshold)
        remapped.append(updated)
        counts[status] += 1
        if status != "confident":
            low_confidence.append((updated, status))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for row in remapped:
            f.write(json.dumps(row) + "\n")

    print(f"{len(items)} items | confident {counts['confident']} | ambiguous {counts['ambiguous']} | failed {counts['failed']}")
    if low_confidence:
        print(f"\nlow-confidence examples (overlap below {args.threshold}):")
        for row, status in low_confidence[:5]:
            print(f"  [{status}] {row['qid']} doc={row['document_id']} old={row['chunk_id_old']} "
                  f"-> best={row['chunk_id']} overlap={row.get('remap_overlap', 0.0)}")
    print(f"\nwrote {args.out.relative_to(REPO_ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
