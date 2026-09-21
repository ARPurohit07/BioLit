"""Check every eval_set.jsonl reference_answer against its own source chunk, using a strong external judge.

experiments/eval/eval_set.jsonl holds 144 questions whose reference_answer was written by a local 3B model and never
audited. scripts/audit_eval_set.py catches a reference whose numbers contradict its source, but not one that is
fluent, numerically consistent, and still wrong about what the paper says. This asks openai/gpt-oss-120b (via
OpenRouter) to read the question, the reference answer, and the source chunk text, and say whether the reference is
actually correct: correct, incorrect, unsupported (chunk doesn't contain the answer), ambiguous (more than one
reasonable answer), or not_substantive (the chunk itself is bibliography/header/footer/toc text, not paper prose --
build_index.py's References filter is known to leak some of these through, and matching words inside a citation
title do not count as support).

Needs OPENROUTER_KEY in the environment, or a .env file at the repo root with OPENROUTER_KEY=... . The key is never
printed or written anywhere.

    python scripts/verify_references.py --dry-run          # print the payload for the first item, no network call
    python scripts/verify_references.py --limit 2           # verify just the first 2 (unverified) items
    python scripts/verify_references.py                     # verify everything not already in the output file

Resumable: appends to experiments/eval/reference_verification.jsonl, skipping qids already present there.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_SET = REPO_ROOT / "experiments" / "eval" / "eval_set.jsonl"
CHUNK_META = REPO_ROOT / "data" / "index" / "chunk_metadata.jsonl"
PROCESSED = REPO_ROOT / "data" / "processed"
OUT_PATH = REPO_ROOT / "experiments" / "eval" / "reference_verification.jsonl"
ENV_PATH = REPO_ROOT / ".env"

API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "openai/gpt-oss-120b"
VERDICTS = ("correct", "incorrect", "unsupported", "ambiguous", "not_substantive")

SYSTEM_PROMPT = (
    "You are auditing a biomedical-literature QA dataset. You will be given a QUESTION, a REFERENCE ANSWER, and the "
    "SOURCE TEXT the reference answer is supposed to come from. Decide whether the reference answer is accurate and "
    "supported by the source text.\n\n"
    "First check whether SOURCE TEXT is substantive paper content (body prose: abstract, methods, results, "
    "discussion, conclusion, etc.) as opposed to a reference/bibliography list, a page header or footer, a table of "
    "contents, or an author/affiliation block. A phrase that merely appears inside a citation entry in a "
    "bibliography (e.g. matching words from a cited paper's title) does NOT count as support for a claim about the "
    "paper's own content — such an item must be verdict not_substantive, never correct, even if the wording matches.\n\n"
    "Respond with strict JSON only, no markdown fences, no extra text, matching exactly this shape:\n"
    '{"verdict": "correct|incorrect|unsupported|ambiguous|not_substantive", "corrected_answer": "...", '
    '"reason": "..."}\n\n'
    "verdict meanings:\n"
    "- correct: the source text is substantive paper content, and the reference answer is accurate and supported "
    "by it.\n"
    "- incorrect: the source text is substantive paper content, but the reference answer is contradicted by it; "
    "corrected_answer must give the right answer, drawn only from the source text.\n"
    "- unsupported: the source text is substantive paper content, but it does not contain the answer to the "
    "question at all.\n"
    "- ambiguous: the source text is substantive paper content, and the question admits more than one reasonable "
    "answer from it.\n"
    "- not_substantive: the source text itself is not substantive paper content (bibliography/reference list, "
    "header/footer, table of contents, author/affiliation block, etc.), so it cannot properly support or refute "
    "any claim about the paper, regardless of whether matching words appear in it.\n\n"
    "For correct, unsupported, ambiguous, or not_substantive verdicts, corrected_answer may be an empty string. "
    "Keep reason to one or two sentences."
)


def load_env_key() -> str:
    import os

    key = os.environ.get("OPENROUTER_KEY")
    if key:
        return key
    if ENV_PATH.exists():
        for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            if k.strip() == "OPENROUTER_KEY":
                v = v.strip().strip('"').strip("'")
                if v:
                    return v
    sys.exit("OPENROUTER_KEY not set: export it or add OPENROUTER_KEY=... to a .env file at the repo root.")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_chunk_text(chunk_id: str, document_id: str) -> str | None:
    for row in read_jsonl(CHUNK_META):
        if row.get("chunk_id") == chunk_id:
            return row.get("text")
    for path in PROCESSED.glob(f"{document_id}*.json"):
        doc = json.loads(path.read_text(encoding="utf-8"))
        for c in doc.get("chunks", []):
            if c.get("chunk_id") == chunk_id:
                return c.get("text")
    return None


def build_payload(item: dict, chunk_text: str) -> dict:
    user_content = (
        f"QUESTION:\n{item['question']}\n\n"
        f"REFERENCE ANSWER:\n{item['reference_answer']}\n\n"
        f"SOURCE TEXT:\n{chunk_text}"
    )
    return {
        "model": MODEL,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
    }


def call_openrouter(payload: dict, api_key: str, retries: int = 2) -> dict | None:
    import requests

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    attempt = 0
    while True:
        try:
            resp = requests.post(API_URL, headers=headers, json=payload, timeout=120)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:  # noqa: BLE001
            if attempt >= retries:
                print(f"  request failed after {attempt + 1} attempts: {exc}")
                return None
            wait = 2 ** attempt
            print(f"  request failed ({exc}), retrying in {wait}s...")
            time.sleep(wait)
            attempt += 1


def parse_verdict(raw_content: str) -> dict | None:
    text = raw_content.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            parsed = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    if not isinstance(parsed, dict) or parsed.get("verdict") not in VERDICTS:
        return None
    return {
        "verdict": parsed["verdict"],
        "corrected_answer": parsed.get("corrected_answer", ""),
        "reason": parsed.get("reason", ""),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="print the payload for the first item and exit")
    ap.add_argument("--limit", type=int, default=None, help="verify at most this many (unverified) items")
    args = ap.parse_args()

    items = read_jsonl(EVAL_SET)
    if not items:
        sys.exit(f"no items found in {EVAL_SET}")

    if args.dry_run:
        item = items[0]
        chunk_text = load_chunk_text(item["chunk_id"], item["document_id"])
        if chunk_text is None:
            sys.exit(f"could not find chunk text for {item['chunk_id']}")
        payload = build_payload(item, chunk_text)
        print(json.dumps(payload, indent=2))
        return

    api_key = load_env_key()

    done_qids = {row["qid"] for row in read_jsonl(OUT_PATH)}
    todo = [item for item in items if item["qid"] not in done_qids]
    if args.limit is not None:
        todo = todo[: args.limit]

    if not todo:
        print("nothing to do (all items already verified)")
        return

    verdict_counts: Counter = Counter()
    prompt_tokens_total = 0
    completion_tokens_total = 0
    n_done = 0

    with OUT_PATH.open("a", encoding="utf-8") as out_f:
        for item in todo:
            qid = item["qid"]
            chunk_text = load_chunk_text(item["chunk_id"], item["document_id"])
            if chunk_text is None:
                record = {"qid": qid, "verdict": "error", "reason": f"chunk text not found for {item['chunk_id']}"}
                out_f.write(json.dumps(record) + "\n")
                out_f.flush()
                verdict_counts["error"] += 1
                n_done += 1
                print(f"[{n_done}/{len(todo)}] {qid}: error (no chunk text)")
                continue

            payload = build_payload(item, chunk_text)
            response = call_openrouter(payload, api_key)

            if response is None:
                record = {"qid": qid, "verdict": "error", "reason": "request failed after retries"}
            else:
                try:
                    content = response["choices"][0]["message"]["content"]
                except (KeyError, IndexError):
                    content = ""
                parsed = parse_verdict(content)
                usage = response.get("usage", {})
                prompt_tokens_total += usage.get("prompt_tokens", 0)
                completion_tokens_total += usage.get("completion_tokens", 0)
                if parsed is None:
                    record = {"qid": qid, "verdict": "error", "reason": "could not parse model reply as JSON"}
                else:
                    record = {"qid": qid, **parsed}

            out_f.write(json.dumps(record) + "\n")
            out_f.flush()
            verdict_counts[record["verdict"]] += 1
            n_done += 1
            print(f"[{n_done}/{len(todo)}] {qid}: {record['verdict']}")

    print("\nverdict counts:")
    for v in (*VERDICTS, "error"):
        if verdict_counts.get(v):
            print(f"  {v}: {verdict_counts[v]}")
    print("\nby type:")
    type_by_qid = {item["qid"]: item.get("type", "unknown") for item in items}
    breakdown: dict[str, Counter] = {}
    for row in read_jsonl(OUT_PATH):
        t = type_by_qid.get(row["qid"], "unknown")
        breakdown.setdefault(t, Counter())[row["verdict"]] += 1
    for t, counts in breakdown.items():
        print(f"  {t}: {dict(counts)}")
    print(f"\ntokens this run: prompt={prompt_tokens_total} completion={completion_tokens_total}")


if __name__ == "__main__":
    main()
