"""Builds a citation-grounded SFT dataset in the exact format the RAG pipeline uses at inference.

Why this exists (vs. training/prepare_dataset.py)
--------------------------------------------------
prepare_dataset.py trains on `instruction + Context:` prompts whose contexts are whole papers
(3k-32k tokens), so at max_seq_length=2048 the *response* is truncated away entirely, and its
responses deliberately contain no [n] citations. The pipeline, however, calls the model with a
system prompt + a numbered-EVIDENCE user prompt and needs inline [n] markers in the answer.

This builder instead:
  1. picks questions (LLM-written from real chunks, plus templated summarize / limitations /
     compare / research-gap tasks),
  2. runs the REAL retriever + reranker + prompt builders (backend/app/generation) restricted to
     each split's own papers (paper-level splits — no evidence leaks across train/val/test),
  3. drafts the answer with a local Ollama "teacher" model (default qwen2.5:3b — no cloud APIs),
  4. keeps an example only if it passes deterministic checks: every [n] refers to a real evidence
     block, most content sentences are cited, numbers appear in the cited evidence, and content
     words are largely present in the cited evidence. Everything else is rejected and logged.

The teacher is a small model and the filters are heuristics: spot-check a sample before training.
Every attempt (accepted or rejected, with reason) is appended to a cache file so a run can resume.

    python -u training/build_cited_dataset.py --processed-dir data/processed --output-dir data/
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
for p in (str(REPO_ROOT), str(REPO_ROOT / "training")):
    if p not in sys.path:
        sys.path.insert(0, p)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from prepare_dataset import split_papers  # noqa: E402

QA_SEED_MIN_TOKENS = 80
K_CHOICES = (5, 4, 3)  # evidence blocks per prompt; the pipeline uses 5, fewer only to fit the token budget

_STOP = set(
    "that this with from were have been which their these those also than then such using used "
    "into over both each other more most only some when while where about between within across "
    "based paper study studies results method methods approach model models proposed show shown "
    "shows however thus therefore they them does not are and the for can may our its it's".split()
)

# Appended to the pipeline's system prompt at GENERATION time only (context distillation): the stored
# training example keeps the original pipeline prompt, so the fine-tuned model learns to answer in this
# style from the prompt it will actually receive. Contains no domain facts — only placeholder wording.
STYLE_HINT = (
    "\n\nSTYLE REQUIREMENTS (follow exactly):\n"
    "- Write 1 to 5 sentences, one per line.\n"
    "- Start each line with its label: SUPPORTED CLAIM:, INTERPRETATION: or LIMITATION:.\n"
    "- Put the [n] citation marker(s) inside the sentence, immediately before its final period, "
    "e.g. 'SUPPORTED CLAIM: Method A improves recall over baseline B [2].'\n"
    "- Every line must contain at least one [n] citation.\n"
    "- State the finding itself. Do not comment on the evidence (no 'this is directly stated', "
    "'this is supported by', 'classified as').\n"
    "- Give the specific detail that answers the question (a number, name, method, dataset or reason). "
    "Do not repeat the question's wording back as the answer."
)

QUESTION_PROMPT = (
    "Below is a passage from a scientific paper. Write ONE specific question a researcher might ask "
    "that this passage helps answer. Focus on {focus}.\n"
    "Rules: the question must be self-contained and name the concepts involved; do NOT say "
    "'the passage', 'this paper' or 'the text'; do NOT state the answer inside the question and do NOT "
    "copy a sentence from the passage — the question should only be answerable by reading the passage; "
    "output only the question, on one line.\n\n"
    "PASSAGE:\n{passage}\n\nQUESTION:"
)
# One focus per question index, so a chunk's 3 questions differ instead of paraphrasing each other.
QUESTION_FOCUS = (
    "a specific result, number, or dataset detail",
    "how a method or component works",
    "why something was done, a limitation, or a comparison",
)


# ---------------------------------------------------------------------------
# Answer quality checks (pure functions — unit-testable)
# ---------------------------------------------------------------------------

_LIST_PREFIX = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")
_LABEL_PREFIX = re.compile(r"^\s*(?:\*\*)?(?:SUPPORTED CLAIM|INTERPRETATION|LIMITATION)(?:\*\*)?\s*:?\s*(?:\*\*)?\s*", re.I)
_CITE = re.compile(r"\[(\d+)\]")
_TRAILING_CITES = re.compile(r"^(?:\[\d+\])+(?:\s+(?:SUPPORTED CLAIM|INTERPRETATION|LIMITATION))?\W*$", re.I)


_CITE_ONLY_LINE = re.compile(r"^\s*(?:(?:SUPPORTED CLAIM|INTERPRETATION|LIMITATION)\s*:?\s*)?((?:\[\d+\]\s*)+)\W*$", re.I)
_META = re.compile(
    r"\b(?:directly stated|is supported by|as indicated by|classified as|provides? evidence|"
    r"based on the (?:provided )?(?:information|evidence)|the (?:provided )?evidence (?:states|shows|indicates))\b"
    r"|\bthis (?:information|can be)\b",
    re.I,
)


_ABSTAIN = re.compile(
    r"\binsufficient (?:evidence|information)\b"
    r"|\b(?:does|do|did) not (?:explicitly |directly |specifically )?"
    r"(?:specify|provide|mention|state|address|describe|demonstrate|contain|include|discuss|report|cover)\b"
    r"|\bnot (?:explicitly |directly |specifically )?(?:provided|specified|mentioned|stated|described|"
    r"demonstrated|addressed|covered|reported)\b",
    re.I,
)


def normalize_answer(answer: str) -> str:
    """Repair the one formatting slip that loses no content: a line holding only citation markers
    (e.g. 'SUPPORTED CLAIM: [3][4]') is folded onto the end of the sentence above it."""
    lines: list[str] = []
    for line in answer.strip().splitlines():
        m = _CITE_ONLY_LINE.match(line)
        prev = next((i for i in range(len(lines) - 1, -1, -1) if lines[i].strip()), None)
        if m and prev is not None:
            marks = "".join(re.findall(r"\[\d+\]", m.group(1)))
            base, tail = lines[prev].rstrip(), "."
            if base and base[-1] in ".!?":
                base, tail = base[:-1], base[-1]
            lines[prev] = f"{base} {marks}{tail}"
        else:
            lines.append(line)
    return "\n".join(lines).strip()


def content_sentences(answer: str) -> list[str]:
    """Sentences long enough to be factual statements (headings / labels / fragments dropped)."""
    out: list[str] = []
    for line in answer.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        line = _LIST_PREFIX.sub("", line)
        line = _LABEL_PREFIX.sub("", line)
        pieces: list[str] = []
        for s in re.split(r"(?<=[.!?])\s+(?=[A-Z\[(\"'*])", line):
            s = s.strip(" *_")
            # The model often writes "... 19.3%. [1][4] SUPPORTED CLAIM": the trailing markers belong
            # to the sentence before them, not to a new sentence.
            if pieces and _TRAILING_CITES.match(s):
                pieces[-1] += " " + s
            elif s:
                pieces.append(s)
        out.extend(s for s in pieces if len(re.findall(r"\w+", _CITE.sub("", s))) >= 6)
    return out


def _content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z][a-z0-9\-]{3,}", text.lower()) if w not in _STOP}


_SPECIFIC_NUM = re.compile(r"\d+\.\d+|\d{3,}")
MIN_NUMBER_CONTEXT = 0.4  # share of a sentence's content words that must sit near its number in the evidence
MIN_NOVELTY = 0.35        # share of a QA answer's content words that are not already in the question
MIN_CITE_OVERLAP = 0.20   # share of a sentence's content words that EACH block it cites must contain


def number_context_recall(sentence: str, evidence: str, window: int = 30) -> Optional[float]:
    """For the sentence's specific numbers (decimals, 3+ digits), the best share of its content words found
    within +/-`window` tokens of that number in the evidence. Catches 'right number, wrong setting'
    (e.g. a transductive result reported as inductive). None if the sentence has no such number."""
    nums = _SPECIFIC_NUM.findall(sentence)
    words = _content_words(sentence)
    if not nums or not words:
        return None
    tokens = [t.strip(".-") for t in re.findall(r"[a-z0-9.\-]+", evidence.lower())]
    worst = 1.0
    for num in nums:
        best = 0.0
        for i, tok in enumerate(tokens):
            if tok == num:
                near = set(tokens[max(0, i - window): i + window + 1])
                best = max(best, len(words & near) / len(words))
        worst = min(worst, best)
    return worst


def novelty(answer_sentences: list[str], question: str) -> float:
    words = _content_words(_CITE.sub("", " ".join(answer_sentences)))
    return len(words - _content_words(question)) / len(words) if words else 0.0


def check_answer(
    answer: str, evidence_texts: dict[int, str], min_sentences: int = 2, question: Optional[str] = None
) -> tuple[bool, str, dict[str, float]]:
    """Returns (accepted, reason, metrics). `evidence_texts` maps citation id -> evidence text.
    `min_sentences` is 1 for direct factual questions, where a single cited sentence is a complete answer.
    Pass `question` for direct questions to also reject answers that merely restate it."""
    metrics: dict[str, float] = {}
    if not answer.strip():
        return False, "empty", metrics
    if "unsupported — could not be verified" in answer:
        return False, "contains_unsupported_note", metrics

    cited_ids = [int(m) for m in _CITE.findall(answer)]
    if not cited_ids:
        return False, "no_citations", metrics
    if any(i not in evidence_texts for i in cited_ids):
        return False, "invalid_citation_id", metrics

    all_sentences = content_sentences(answer)
    if any(_META.search(s) for s in all_sentences):
        return False, "meta_commentary", metrics
    # "The evidence does not specify X" is the behavior the pipeline prompt asks for when evidence is
    # insufficient, so an uncited abstention is not an unsupported claim. It doesn't count toward the
    # factual sentences below, and the answer still needs cited content (a pure abstention can't be
    # verified, and might be a false abstention).
    sentences = [s for s in all_sentences if _CITE.search(s) or not _ABSTAIN.search(s)]
    if len(sentences) < min_sentences:
        return False, "too_few_sentences", metrics

    cited_sentences = [s for s in sentences if _CITE.search(s)]
    coverage = len(cited_sentences) / len(sentences)
    metrics["coverage"] = round(coverage, 3)
    if coverage < 0.7:
        return False, "low_citation_coverage", metrics

    recalls: list[float] = []
    for s in cited_sentences:
        ids = [int(m) for m in _CITE.findall(s)]
        ev = " ".join(evidence_texts[i] for i in ids)
        bare = _CITE.sub("", s)
        for num in re.findall(r"\d+(?:\.\d+)?", bare):
            if num not in ev:
                return False, f"number_not_in_evidence:{num}", metrics
        ctx = number_context_recall(bare, ev)
        if ctx is not None and ctx < MIN_NUMBER_CONTEXT:
            return False, "number_context_mismatch", metrics
        words = _content_words(bare)
        if words:
            recalls.append(len(words & _content_words(ev)) / len(words))
            # Citation precision: each cited block must itself share some of the sentence's words,
            # otherwise a relevant block can be padded with an unrelated [n] and still pass the pooled
            # check above. Kept low (0.20): legitimate [1][3] citations each cover only part of a sentence.
            if not _ABSTAIN.search(s):
                for i in ids:
                    if len(words & _content_words(evidence_texts[i])) / len(words) < MIN_CITE_OVERLAP:
                        return False, "irrelevant_citation", metrics
    if not recalls:
        return False, "nothing_to_ground", metrics
    mean_recall = sum(recalls) / len(recalls)
    weak_frac = sum(r < 0.4 for r in recalls) / len(recalls)
    metrics["grounding"] = round(mean_recall, 3)
    metrics["weak_frac"] = round(weak_frac, 3)
    if mean_recall < 0.55 or weak_frac > 0.25:
        return False, "weak_grounding", metrics

    if question is not None:
        nov = novelty(sentences, question)
        metrics["novelty"] = round(nov, 3)
        if nov < MIN_NOVELTY:
            return False, "restates_question", metrics

    lines = [l.strip() for l in answer.splitlines() if l.strip()]
    if len(lines) != len(set(lines)) and len(lines) > 4:
        return False, "repeated_lines", metrics
    return True, "ok", metrics


# ---------------------------------------------------------------------------
# Candidate construction
# ---------------------------------------------------------------------------

def load_papers(processed_dir: Path) -> list[dict[str, Any]]:
    papers = []
    for path in sorted(processed_dir.glob("*.json")):
        rec = json.loads(path.read_text(encoding="utf-8"))
        if rec.get("chunks") and rec.get("status") != "scanned_needs_ocr":
            papers.append(rec)
    return papers


def build_candidates(
    papers_by_split: dict[str, list[dict[str, Any]]], qpc: dict[str, int], rng: random.Random
) -> list[dict[str, Any]]:
    """Ordered candidate list. QA questions are generated lazily (need the LLM); templated ones are ready."""
    cands: list[dict[str, Any]] = []
    for split, papers in papers_by_split.items():
        ids = [p["document_id"] for p in papers]
        split_cands: list[dict[str, Any]] = []
        for paper in papers:
            title = paper.get("title") or paper["document_id"]
            split_cands.append({"id": f"{split}:summarize:{paper['document_id']}", "split": split, "kind": "summarize",
                                "question": f"Summarize the key findings of: {title}", "document_ids": [paper["document_id"]]})
            split_cands.append({"id": f"{split}:limitations:{paper['document_id']}", "split": split, "kind": "limitations",
                                "question": title, "document_ids": [paper["document_id"]]})
            for c in paper["chunks"]:
                if c.get("section") == "References" or (c.get("token_count") or 0) < QA_SEED_MIN_TOKENS:
                    continue
                for q in range(qpc[split]):
                    split_cands.append({"id": f"{split}:qa:{c['chunk_id']}:{q}", "split": split, "kind": "qa",
                                        "seed_text": c["text"], "seed_index": q, "document_ids": ids})
        for i in range(len(papers)):
            for j in range(i + 1, len(papers)):
                pair = [papers[i]["document_id"], papers[j]["document_id"]]
                for kind, qtext in (("compare_methodology", "Compare the methodology used across these papers."),
                                    ("compare_results", "Compare the results and findings reported across these papers.")):
                    split_cands.append({"id": f"{split}:{kind}:{pair[0]}:{pair[1]}", "split": split, "kind": kind,
                                        "question": qtext, "document_ids": pair})
        if len(papers) >= 2:
            split_cands.append({"id": f"{split}:research_gaps:all", "split": split, "kind": "research_gaps",
                                "question": "Identify research gaps across these papers.", "document_ids": ids})
        rng.shuffle(split_cands)
        cands.extend(split_cands)
    return cands


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def fmt_eta(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m" if seconds >= 3600 else f"{seconds // 60}m{seconds % 60:02d}s"


def load_cache(path: Path) -> dict[str, dict[str, Any]]:
    cache: dict[str, dict[str, Any]] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                cache[rec["id"]] = rec
    return cache


def evidence_from_context(context: str) -> dict[int, str]:
    """Inverse of the '[n] text' joined context stored in each example."""
    return {int(m.group(1)): m.group(2) for m in re.finditer(r"\[(\d+)\] (.*?)(?=\n\n\[\d+\] |\Z)", context, re.S)}


def revalidate(example: dict[str, Any]) -> tuple[bool, str]:
    """Re-run the CURRENT answer checks on a stored example, so tightening a filter never needs regeneration."""
    is_qa = example["task_type"] == "qa"
    ok, reason, _ = check_answer(
        example["response"], evidence_from_context(example["context"]),
        min_sentences=1 if is_qa else 2, question=example["instruction"] if is_qa else None,
    )
    return ok, reason


def finalize(cache: dict[str, dict[str, Any]], output_dir: Path) -> dict[str, int]:
    targets = {"train": output_dir / "training" / "train.jsonl",
               "val": output_dir / "validation" / "val.jsonl",
               "test": output_dir / "test" / "test.jsonl"}
    counts = {}
    dropped: dict[str, int] = {}
    for split, path in targets.items():
        rows = []
        for r in cache.values():
            if r["split"] != split or r["status"] != "accepted":
                continue
            ok, reason = revalidate(r["example"])
            if ok:
                rows.append(r["example"])
            else:
                dropped[reason] = dropped.get(reason, 0) + 1
        if dropped:
            print(f"[build] re-validation with current checks dropped: {dropped}", flush=True)
        backup = path.with_suffix(".jsonl.v1_uncited.bak")
        if path.exists() and not backup.exists():
            path.replace(backup)  # keep the previous (uncited, truncated) dataset instead of clobbering it
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        counts[split] = len(rows)
    return counts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--processed-dir", default="data/processed")
    ap.add_argument("--output-dir", default="data/")
    ap.add_argument("--cache", default="data/training/cited_attempts.jsonl")
    ap.add_argument("--teacher-model", default=None, help="Ollama model that drafts answers (default: configured serving model)")
    ap.add_argument("--train-questions-per-chunk", type=int, default=2)
    ap.add_argument("--eval-questions-per-chunk", type=int, default=1)
    ap.add_argument("--prompt-budget", type=int, default=2200, help="max tokens for system+user prompt")
    ap.add_argument("--max-answer-tokens", type=int, default=600)
    ap.add_argument("--limit", type=int, default=None, help="only attempt this many NEW candidates (smoke test)")
    ap.add_argument("--finalize-only", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    cache_path = Path(args.cache)
    cache = load_cache(cache_path)

    if args.finalize_only:
        print("Wrote:", finalize(cache, output_dir))
        return

    from transformers import AutoTokenizer

    from backend.app.config.settings import get_settings
    from backend.app.generation.ollama_client import OllamaClient
    from backend.app.generation.rag_pipeline import RAGPipeline
    from backend.app.models.schemas import QueryType
    from backend.app.retrieval.bm25 import BM25Index
    from backend.app.retrieval.embeddings import EmbeddingModel
    from backend.app.retrieval.hybrid import HybridRetriever
    from backend.app.retrieval.reranker import Reranker
    from backend.app.retrieval.vector_store import FAISSVectorStore
    from backend.app.verification.claims import ClaimExtractor

    settings = get_settings()
    rcfg, mcfg = settings.retrieval_config, settings.models_config
    teacher = args.teacher_model or settings.ollama_model

    print(f"[build] teacher={teacher} host={settings.ollama_host} prompt_budget={args.prompt_budget} "
          f"max_answer_tokens={args.max_answer_tokens}", flush=True)
    tok = AutoTokenizer.from_pretrained(mcfg.get("base_model", {}).get("name", "Qwen/Qwen2.5-1.5B-Instruct"))

    def ntok(text: str) -> int:
        return len(tok(text)["input_ids"])

    # Retrieval on CPU: leaves the 4GB GPU to Ollama (and later, training).
    emb_cfg, rr_cfg = mcfg.get("embedding_model", {}), mcfg.get("reranker", {})
    embedder = EmbeddingModel(emb_cfg.get("name", "BAAI/bge-small-en-v1.5"), device="cpu", normalize=emb_cfg.get("normalize", True))
    vs_cfg, bm_cfg = rcfg.get("vector_store", {}), rcfg.get("bm25", {})
    vstore = FAISSVectorStore(dim=emb_cfg.get("dim", 384),
                              index_path=str(settings.repo_root / vs_cfg.get("index_path", "data/index/faiss.index")),
                              metadata_path=str(settings.repo_root / vs_cfg.get("metadata_path", "data/index/chunk_metadata.jsonl")))
    vstore.load()
    bm25 = BM25Index(index_path=str(settings.repo_root / bm_cfg.get("index_path", "data/index/bm25_index.pkl")),
                     k1=bm_cfg.get("k1", 1.5), b=bm_cfg.get("b", 0.75))
    bm25.load()
    retriever = HybridRetriever(vstore, bm25, embedder, rcfg)
    reranker = Reranker(rr_cfg.get("name", "BAAI/bge-reranker-base"), device="cpu", max_length=rr_cfg.get("max_length", 512))
    client = OllamaClient(settings.ollama_host, teacher, timeout_s=300)
    pipeline = RAGPipeline(retriever, reranker, client, ClaimExtractor(), None, rcfg)  # type: ignore[arg-type]
    output_k = rcfg.get("reranker", {}).get("output_k", 5)

    papers = load_papers(Path(args.processed_dir))
    train_ids, val_ids, test_ids = split_papers([p["document_id"] for p in papers], 0.1, 0.1, args.seed)
    by_id = {p["document_id"]: p for p in papers}
    papers_by_split = {"val": [by_id[i] for i in val_ids], "test": [by_id[i] for i in test_ids],
                       "train": [by_id[i] for i in train_ids]}
    print(f"[build] paper-level split: {len(train_ids)} train / {len(val_ids)} val / {len(test_ids)} test papers", flush=True)

    rng = random.Random(args.seed)
    qpc = {"train": args.train_questions_per_chunk, "val": args.eval_questions_per_chunk, "test": args.eval_questions_per_chunk}
    candidates = build_candidates(papers_by_split, qpc, rng)
    todo = [c for c in candidates if c["id"] not in cache]
    n_cached = len(candidates) - len(todo)
    if args.limit:
        todo = todo[: args.limit]
    per_split = {s: sum(c["split"] == s for c in candidates) for s in papers_by_split}
    print(f"[build] {len(candidates)} candidates {per_split} | {n_cached} already cached | "
          f"{len(todo)} to attempt now", flush=True)

    query_types = {"summarize": QueryType.SUMMARIZE, "limitations": QueryType.LIMITATIONS,
                   "compare_methodology": QueryType.COMPARE_METHODOLOGY, "compare_results": QueryType.COMPARE_RESULTS,
                   "research_gaps": QueryType.RESEARCH_GAPS, "qa": QueryType.QUESTION_ANSWERING}
    seen_questions: set[str] = {r["example"]["instruction"].lower() for r in cache.values() if r.get("example")}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    stats = {"accepted": sum(r["status"] == "accepted" for r in cache.values()),
             "rejected": sum(r["status"] == "rejected" for r in cache.values())}
    reasons: dict[str, int] = {}
    t_start = time.time()

    def record(cand: dict[str, Any], status: str, reason: str, example: Optional[dict[str, Any]] = None, **extra: Any) -> None:
        rec = {"id": cand["id"], "split": cand["split"], "kind": cand["kind"], "status": status, "reason": reason, **extra}
        if example is not None:
            rec["example"] = example
        cache[cand["id"]] = rec
        with open(cache_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        stats[status] += 1
        if status == "rejected":
            key = reason.split(":")[0]
            reasons[key] = reasons.get(key, 0) + 1

    try:
        for n, cand in enumerate(todo, start=1):
            tag = f"[{n:>3}/{len(todo)}] {cand['split']:<5} {cand['kind']:<19}"

            # 1. question
            if cand["kind"] == "qa":
                temp = 0.3 if cand["seed_index"] == 0 else 0.9
                focus = QUESTION_FOCUS[cand["seed_index"] % len(QUESTION_FOCUS)]
                question = client.generate(
                    QUESTION_PROMPT.format(passage=cand["seed_text"], focus=focus), temperature=temp
                ).strip().splitlines()[0].strip(' "*')
                if len(question) < 15 or not question.endswith("?"):
                    record(cand, "rejected", "bad_question", question=question)
                    print(f"{tag} REJECT bad_question: {question[:70]!r}", flush=True)
                    continue
                if question.lower() in seen_questions:
                    record(cand, "rejected", "duplicate_question", question=question)
                    print(f"{tag} REJECT duplicate_question", flush=True)
                    continue
                seen_questions.add(question.lower())
            else:
                question = cand["question"]

            # 2. real retrieval + rerank, restricted to this split's papers
            pool = retriever.retrieve(question, mode="balanced", document_ids=cand["document_ids"])
            ranked = reranker.rerank(question, pool, top_k=output_k)
            full_evidence = pipeline._build_evidence(ranked)
            if len(full_evidence) < 3:
                record(cand, "rejected", "too_little_evidence", question=question)
                print(f"{tag} REJECT too_little_evidence ({len(full_evidence)} chunks)", flush=True)
                continue

            # 3. fit the pipeline's prompt into the token budget by dropping the lowest-ranked evidence
            chosen = None
            for k in K_CHOICES:
                if k > len(full_evidence):
                    continue
                evidence = full_evidence[:k]
                system, user = pipeline._build_prompt(query_types[cand["kind"]], question, evidence)
                prompt_tokens = ntok(system) + ntok(user) + 25
                if prompt_tokens <= args.prompt_budget:
                    chosen = (evidence, system, user, prompt_tokens)
                    break
            if chosen is None:
                record(cand, "rejected", "prompt_too_long", question=question)
                print(f"{tag} REJECT prompt_too_long (>{args.prompt_budget} tok even with 3 chunks)", flush=True)
                continue
            evidence, system, user, prompt_tokens = chosen

            # 4. draft with the teacher, then filter
            t_gen = time.time()
            answer = normalize_answer(client.generate(user, system=system + STYLE_HINT))
            gen_s = time.time() - t_gen
            answer_tokens = ntok(answer)
            if answer_tokens > args.max_answer_tokens:
                ok, reason, metrics = False, "answer_too_long", {}
            else:
                ok, reason, metrics = check_answer(
                    answer, {e.citation_id: e.text for e in evidence},
                    min_sentences=1 if cand["kind"] == "qa" else 2,
                    question=question if cand["kind"] == "qa" else None,
                )

            info = f"k={len(evidence)} prompt={prompt_tokens}tok ans={answer_tokens}tok gen={gen_s:.0f}s"
            if ok:
                example = {
                    "system": system, "user": user, "response": answer,
                    "instruction": question, "context": "\n\n".join(f"[{e.citation_id}] {e.text}" for e in evidence),
                    "provenance": [{"paper_id": e.document_id, "page": e.page_number, "section": e.section} for e in evidence],
                    "task_type": cand["kind"], "teacher": teacher, "checks": metrics,
                }
                record(cand, "accepted", "ok", example=example, question=question)
                print(f"{tag} ACCEPT  {info} cov={metrics['coverage']} grounded={metrics['grounding']}", flush=True)
            else:
                record(cand, "rejected", reason, question=question, answer=answer, checks=metrics)
                print(f"{tag} REJECT  {info} -> {reason}", flush=True)

            done_time = time.time() - t_start
            if n % 5 == 0 or n == len(todo):
                eta = done_time / n * (len(todo) - n)
                top = ", ".join(f"{k}={v}" for k, v in sorted(reasons.items(), key=lambda kv: -kv[1])[:4]) or "-"
                print(f"    ---- progress {n}/{len(todo)} | accepted {stats['accepted']} rejected {stats['rejected']} "
                      f"| this run rejects: {top} | elapsed {fmt_eta(done_time)} ETA {fmt_eta(eta)}", flush=True)
    except KeyboardInterrupt:
        print("\n[build] interrupted — progress is cached; re-run to resume.", flush=True)

    if args.limit:
        print(f"\n[build] --limit run: NOT writing train/val/test files (cache: {cache_path}). "
              "Re-run without --limit to continue and finalize.", flush=True)
        return
    counts = finalize(cache, output_dir)
    print(f"\n[build] wrote accepted examples: {counts} (cache: {cache_path})", flush=True)


if __name__ == "__main__":
    main()
