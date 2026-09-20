"""Build a labelled evaluation set: questions whose source chunk is known, so retrieval can be scored for real.

Each item is generated from one indexed chunk (text, table or figure): a local LLM writes a question that the chunk
answers plus a short reference answer. The source chunk and paper are the ground-truth relevance label.

Filters keep the set honest rather than merely large:
  * self-contained: no "this paper"/"the text" style references, because a real user's question has no such context;
  * grounded reference: the reference answer must be supported by the chunk (numbers included), checked with the same
    deterministic grounding test the app uses;
  * no copying: the question may not reuse a run of words from the passage, and its word overlap with the passage is
    recorded so the "easy" (high overlap) and "hard" (low overlap) halves can be reported separately.

Limits, stated in the README: the labels are single-chunk, so other chunks that also answer count as misses
(chunk-level recall is a lower bound; paper-level recall is reported too), and the generator is a 3B model.

    python scripts/build_eval_set.py --n 150            # writes experiments/eval/eval_set.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.app.verification.grounding import NONE, content_words, estimate  # noqa: E402

PROCESSED = REPO_ROOT / "data" / "processed"
OUT = REPO_ROOT / "experiments" / "eval" / "eval_set.jsonl"
OLLAMA = "http://localhost:11434/api/chat"
MODEL = "qwen2.5:3b"
SKIP_SECTIONS = {"References", "Abstract", "Acknowledgements", "Acknowledgments", "Unknown"}
MIX = {"text": 0.65, "table": 0.20, "figure": 0.15}   # share of the set by chunk type

_META_REF = re.compile(
    r"\b(this|the|that) (paper|passage|text|study|article|excerpt|section|authors?|work|document|context)\b|"
    r"\b(above|following|given) (text|passage|table|figure|excerpt)\b|\baccording to (the|this)\b", re.IGNORECASE)
_LABEL_REF = re.compile(r"\b(table|figure|fig\.)\s*\d+", re.IGNORECASE)
# A usable question names something: a model or dataset (CamelCase, acronym, hyphenated code), a number, or a proper
# noun after the first word. "How do researchers usually evaluate new techniques?" has no single right passage.
_SPECIFIC = re.compile(r"\b[A-Z][a-z]*[A-Z0-9][A-Za-z0-9\-]*\b|\b[A-Z]{2,}|\d|(?<=\s)[A-Z][a-z]{3,}")

PROMPTS = {
    "text": (
        "Below is a passage from the research paper \"{title}\".\n\nPASSAGE:\n{chunk}\n\n"
        "Write ONE question that a researcher could ask and that this passage answers. Rules: the question must make "
        "sense on its own to someone who has not read the paper, so it MUST contain the exact name of the specific model, "
        "method, dataset or drug it is about (for example 'MSDA', 'GDSCv2', 'SiamDTI') instead of saying 'this paper', 'the "
        "model' or 'the authors'; put it in your own words (do not copy phrases of four or more words from the passage); "
        "ask about a concrete fact, not a yes/no. Then give a 1-2 sentence answer using ONLY facts stated in the passage.\n"
        'Reply as JSON: {{"question": "...", "answer": "..."}}'
    ),
    "table": (
        "Below is a table from the research paper \"{title}\" (Markdown).\n\nTABLE:\n{chunk}\n\n"
        "Write ONE question about a specific value or comparison in this table, naming the exact method, metric and dataset "
        "so that it makes sense on its own (do not say 'the table above'). Then give the answer, quoting the number(s) "
        "exactly as in the table.\n"
        'Reply as JSON: {{"question": "...", "answer": "..."}}'
    ),
    "figure": (
        "Below is a figure caption from the research paper \"{title}\".\n\nCAPTION:\n{chunk}\n\n"
        "Write ONE question that this caption answers. It MUST contain the exact name of the specific model, method or "
        "dataset it is about so it makes sense on its own. Then give a 1-2 sentence answer using ONLY what the caption states.\n"
        'Reply as JSON: {{"question": "...", "answer": "..."}}'
    ),
}


def load_chunks() -> dict[str, dict]:
    docs = {}
    for f in sorted(PROCESSED.glob("*.json")):
        rec = json.loads(f.read_text(encoding="utf-8"))
        if rec.get("status") == "scanned_needs_ocr":
            continue
        for c in rec["chunks"]:
            c["paper_title"] = rec.get("title") or rec["document_id"]
            docs.setdefault(rec["document_id"], []).append(c)
    return docs


def eligible(c: dict) -> bool:
    if c["section"] in SKIP_SECTIONS:
        return False
    kind = c.get("chunk_type", "text")
    if kind == "text":
        words = c["text"].split()
        digits = sum(ch.isdigit() for ch in c["text"]) / max(len(c["text"]), 1)
        return 90 <= len(words) <= 420 and digits < 0.12        # prose, not a residual table or a reference list
    if kind == "table":
        return c["text"].count("|") >= 12 and c["token_count"] <= 500
    return len(c["text"].split()) >= 12                            # figure: a real caption


def sample(docs: dict[str, list[dict]], n: int, seed: int) -> list[dict]:
    """Stratified: spread over papers first (round robin), split by chunk type per MIX."""
    rng = random.Random(seed)
    picked: list[dict] = []
    for kind, share in MIX.items():
        want = round(n * share * 2.6)                              # over-sample: many candidates fail the filters
        pools = {d: [c for c in cs if c.get("chunk_type", "text") == kind and eligible(c)] for d, cs in docs.items()}
        for pool in pools.values():
            rng.shuffle(pool)
        got: list[dict] = []
        while len(got) < want and any(pools.values()):
            for d in sorted(pools, key=lambda _: rng.random()):
                if pools[d] and len(got) < want:
                    got.append(pools[d].pop())
        picked += got
    rng.shuffle(picked)
    return picked


def ask(prompt: str) -> dict | None:
    try:
        r = requests.post(OLLAMA, json={
            "model": MODEL, "stream": False, "format": "json", "keep_alive": "10m",
            "options": {"temperature": 0.3, "num_predict": 220},
            "messages": [{"role": "user", "content": prompt}],
        }, timeout=180)
        data = json.loads(r.json()["message"]["content"])
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _ngrams(text: str, n: int) -> set[tuple]:
    toks = re.findall(r"[a-z0-9]+", text.lower())
    return {tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)}


def check(item: dict, chunk: dict) -> str | None:
    """Return the reason an item is rejected, or None when it passes."""
    q, a, kind = item.get("question", ""), item.get("answer", ""), chunk.get("chunk_type", "text")
    if not isinstance(q, str) or not isinstance(a, str) or not q.strip() or not a.strip():
        return "empty"
    if not q.strip().endswith("?") or not 7 <= len(q.split()) <= 40:
        return "bad question shape"
    if _META_REF.search(q):
        return "refers to the paper/text"
    if kind == "text" and _LABEL_REF.search(q):
        return "refers to a table/figure label"
    if not _SPECIFIC.search(q):
        return "generic question (names nothing specific)"
    if _ngrams(q, 5) & _ngrams(chunk["text"], 5):
        return "copies the passage"
    numbers = re.findall(r"\d+(?:\.\d+)?", a)
    if kind == "table" and numbers:          # a table answer is often just a value: it must be in the table
        return None if all(n in chunk["text"] for n in numbers) else "answer numbers not in the table"
    if len(a.split()) < 3:
        return "answer too short"
    g = estimate(a, [chunk["text"]])
    if g.label == NONE:
        return f"answer not grounded in the passage ({g.reason})"
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=150, help="target number of questions")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    docs = load_chunks()
    flat = {c["chunk_id"]: c for cs in docs.values() for c in cs}
    print(f"{len(docs)} papers, {len(flat)} chunks: {dict(Counter(c.get('chunk_type', 'text') for c in flat.values()))}", flush=True)
    candidates = sample(docs, args.n, args.seed)
    print(f"{len(candidates)} candidate chunks; generating with {MODEL}", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    # Resumable: accepted questions are appended as they are made, so an interrupted run loses nothing.
    kept: list[dict] = [json.loads(l) for l in OUT.read_text(encoding="utf-8").splitlines() if l.strip()] if OUT.exists() else []
    generic = [k for k in kept if not _SPECIFIC.search(k["question"])]
    if generic:                                    # saved before the specificity filter existed: drop and renumber
        kept = [k for k in kept if _SPECIFIC.search(k["question"])]
        for n, k in enumerate(kept):
            k["qid"] = f"q{n:03d}"
        OUT.write_text("".join(json.dumps(k, ensure_ascii=False) + "\n" for k in kept), encoding="utf-8")
        print(f"dropped {len(generic)} generic questions saved earlier", flush=True)
    done_chunks = {k["chunk_id"] for k in kept}
    rejected = Counter()
    seen_q: set[str] = {k["question"].strip().lower() for k in kept}
    per_type = Counter(k["type"] for k in kept)
    if kept:
        print(f"resuming with {len(kept)} questions already saved {dict(per_type)}", flush=True)
    quota = {k: round(args.n * v) for k, v in MIX.items()}
    t0 = time.time()
    for i, chunk in enumerate(candidates, 1):
        kind = chunk.get("chunk_type", "text")
        if per_type[kind] >= quota[kind] or chunk["chunk_id"] in done_chunks:
            continue
        item = ask(PROMPTS[kind].format(chunk=chunk["text"][:2600], title=chunk.get("paper_title", "")[:120]))
        why = "no/invalid JSON" if item is None else check(item, chunk)
        if why is None and item["question"].strip().lower() in seen_q:
            why = "duplicate question"
        if why:
            rejected[why.split(" (")[0]] += 1
        else:
            seen_q.add(item["question"].strip().lower())
            q_words = content_words(item["question"])
            overlap = len(q_words & content_words(chunk["text"])) / max(len(q_words), 1)
            kept.append({
                "qid": f"q{len(kept):03d}", "type": kind, "question": item["question"].strip(),
                "reference_answer": item["answer"].strip(), "chunk_id": chunk["chunk_id"],
                "document_id": chunk["document_id"], "paper_title": chunk["paper_title"],
                "page": chunk["page_number"], "section": chunk["section"], "label": chunk.get("label"),
                "question_chunk_overlap": round(overlap, 3),
            })
            per_type[kind] += 1
            with open(OUT, "a", encoding="utf-8") as f:
                f.write(json.dumps(kept[-1], ensure_ascii=False) + "\n")
        if i % 10 == 0:
            print(f"[{i}/{len(candidates)}] kept {len(kept)} {dict(per_type)} | {(time.time() - t0) / 60:.1f} min", flush=True)
        if all(per_type[k] >= quota[k] for k in quota):
            break

    papers = len({k["document_id"] for k in kept})
    ov = sorted(k["question_chunk_overlap"] for k in kept)
    print(f"\nkept {len(kept)} questions from {papers} papers {dict(per_type)}")
    print(f"question/passage word overlap: median {ov[len(ov) // 2]:.2f}, max {ov[-1]:.2f}")
    print("rejected:", dict(rejected))
    print(f"wrote {OUT.relative_to(REPO_ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
