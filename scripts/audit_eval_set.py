"""Flag unreliable references in the labelled eval set, deterministically (no LLM).

experiments/eval/eval_set.jsonl has model-written reference_answer strings tied to a source chunk_id. Some of those
references are wrong, e.g. q091 asks for the AUC at N1=5, N2=10 in Table 4 and answers 0.9721 -- the value one cell
away (row N1=10, col N2=5, i.e. the table read transposed). Optimising the generator against a broken reference
wastes effort, so this checks every item against its own source chunk:

    numbers_in_source    every number in the reference appears in the source chunk; for table questions naming two
                          axes ("N1=5 and N2=10"), the actual cell is parsed and compared, so a transposed-but-present
                          number still fails
    term_grounding       share of the reference answer's content words present in the source chunk
    question_answerable  share of the QUESTION's content words present in the source chunk (if the question's own
                          subject isn't in its source chunk, it's probably mislabelled)
    ambiguity            other chunks in the same document that also contain every reference number

    ok        no issues
    suspect   one weak signal (low term/question grounding, or an ambiguous number match elsewhere)
    broken    a reference number is missing from its source chunk, or lands in the wrong table cell

    python scripts/audit_eval_set.py     # writes experiments/eval/eval_audit.json and eval_set_clean.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_answers_fast import NUMBER, words  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_SET = REPO_ROOT / "experiments" / "eval" / "eval_set.jsonl"
CHUNK_META = REPO_ROOT / "data" / "index" / "chunk_metadata.jsonl"
PROCESSED = REPO_ROOT / "data" / "processed"
OUT_AUDIT = REPO_ROOT / "experiments" / "eval" / "eval_audit.json"
OUT_CLEAN = REPO_ROOT / "experiments" / "eval" / "eval_set_clean.jsonl"

TERM_GROUNDING_MIN = 0.5
QUESTION_ANSWERABLE_MIN = 0.4
VAR_EQ = re.compile(r"[A-Za-z][A-Za-z0-9_]*\s*=\s*(-?\d+(?:\.\d+)?)")


def load_chunks(meta_path: Path) -> dict[str, dict]:
    chunks = {}
    for line in meta_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            chunks[row["chunk_id"]] = row
    return chunks


def find_in_processed(chunk_id: str, document_id: str) -> dict | None:
    for path in PROCESSED.glob(f"{document_id}*.json"):
        doc = json.loads(path.read_text(encoding="utf-8"))
        for c in doc.get("chunks", []):
            if c.get("chunk_id") == chunk_id:
                return c
    return None


def parse_table_grid(text: str) -> tuple[dict[str, int], dict[str, list[str]]]:
    """Pull a {column label: index} map and a {row label: row values} map out of a markdown pipe table."""
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if all(re.fullmatch(r"-+", c or "-") for c in cells):
            continue  # separator row
        rows.append(cells)
    if len(rows) < 2:
        return {}, {}
    header, *data_rows = rows
    col_index = {c: i for i, c in enumerate(header[1:]) if c}
    row_values: dict[str, list[str]] = {}
    for r in data_rows:
        if len(r) < 2 or not r[0] or not any(r[1:]):
            continue  # blank or label-only row
        row_values[r[0]] = r[1:]
    return col_index, row_values


def table_cell(col_index: dict, row_values: dict, row_label: str, col_label: str) -> str | None:
    r = row_values.get(row_label)
    c = col_index.get(col_label)
    if r is None or c is None or c >= len(r) or not r[c]:
        return None
    return r[c]


def check_table_cell(question: str, reference_answer: str, chunk_text: str) -> dict | None:
    """For a table question naming two `var=value` axes, verify against the actual cell rather than a plain
    substring search, which a transposed-but-present number would still pass."""
    values = VAR_EQ.findall(question)
    if len(values) != 2:
        return None
    col_index, row_values = parse_table_grid(chunk_text)
    if not col_index or not row_values:
        return None
    v1, v2 = values
    cell = table_cell(col_index, row_values, v1, v2)
    if cell is None:
        return None
    ref_nums = NUMBER.findall(reference_answer)
    if cell in ref_nums:
        return {"match": True, "cell": cell}
    swapped = table_cell(col_index, row_values, v2, v1)
    if swapped is not None and swapped in ref_nums:
        return {"match": False, "cell": cell,
                "reason": f"reference {ref_nums} matches the transposed cell ({swapped}), not the requested one ({cell})"}
    return {"match": False, "cell": cell, "reason": f"parsed table cell is {cell}, reference has {ref_nums}"}


def base_fields(item: dict) -> dict:
    return {
        "qid": item["qid"],
        "type": item["type"],
        "question": item["question"],
        "reference_answer": item["reference_answer"],
        "chunk_id": item["chunk_id"],
    }


def audit_item(item: dict, chunks: dict[str, dict]) -> dict:
    chunk = chunks.get(item["chunk_id"]) or find_in_processed(item["chunk_id"], item["document_id"])
    if chunk is None:
        r = base_fields(item)
        r.update(numbers_in_source=False, missing_numbers=None, table_cell_check=None, term_grounding=0.0,
                  question_answerable=0.0, ambiguity=0, verdict="broken",
                  reasons=["source chunk not found in chunk_metadata.jsonl or data/processed/"], _badness=1000)
        return r
    text = chunk.get("text", "")

    ref_nums = NUMBER.findall(item["reference_answer"])
    missing = [n for n in ref_nums if n not in text]

    table_check = None
    if item["type"] == "table" and not missing:
        table_check = check_table_cell(item["question"], item["reference_answer"], text)

    ref_words = words(item["reference_answer"])
    source_words = words(text)
    term_grounding = len(ref_words & source_words) / len(ref_words) if ref_words else 1.0

    q_words = words(item["question"])
    question_answerable = len(q_words & source_words) / len(q_words) if q_words else 1.0

    doc_chunks = [c for cid, c in chunks.items()
                  if c["document_id"] == item["document_id"] and cid != item["chunk_id"]]
    ambiguity = sum(1 for c in doc_chunks if ref_nums and all(n in c.get("text", "") for n in ref_nums))

    reasons, verdict = [], "ok"
    if missing:
        reasons.append(f"reference numbers not in source chunk: {missing}")
        verdict = "broken"
    if table_check and not table_check["match"]:
        reasons.append(table_check["reason"])
        verdict = "broken"

    weak = []
    if term_grounding < TERM_GROUNDING_MIN:
        weak.append(f"low term grounding ({term_grounding:.2f})")
    if question_answerable < QUESTION_ANSWERABLE_MIN:
        weak.append(f"question subject barely in source chunk ({question_answerable:.2f})")
    if ambiguity:
        weak.append(f"{ambiguity} other chunk(s) in the document also contain every reference number")

    if verdict == "ok" and weak:
        verdict = "suspect"
    reasons.extend(weak)

    badness = (100 if verdict == "broken" else 10 if verdict == "suspect" else 0)
    badness += len(missing) * 5 + len(weak) + round((1 - term_grounding) + (1 - question_answerable), 2)

    r = base_fields(item)
    r.update(numbers_in_source=not missing, missing_numbers=missing, table_cell_check=table_check,
              term_grounding=round(term_grounding, 3), question_answerable=round(question_answerable, 3),
              ambiguity=ambiguity, verdict=verdict, reasons=reasons, _badness=badness)
    return r


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-set", type=Path, default=EVAL_SET)
    ap.add_argument("--chunk-metadata", type=Path, default=CHUNK_META)
    ap.add_argument("--out-audit", type=Path, default=OUT_AUDIT)
    ap.add_argument("--out-clean", type=Path, default=OUT_CLEAN)
    ap.add_argument("--worst", type=int, default=10)
    args = ap.parse_args()

    items = [json.loads(l) for l in args.eval_set.read_text(encoding="utf-8").splitlines() if l.strip()]
    chunks = load_chunks(args.chunk_metadata)
    results = [audit_item(item, chunks) for item in items]

    counts: dict = defaultdict(int)
    by_type: dict = defaultdict(lambda: defaultdict(int))
    for r in results:
        counts[r["verdict"]] += 1
        by_type[r["type"]][r["verdict"]] += 1
    summary = {"total": len(results), "counts": dict(counts),
               "by_type": {t: dict(v) for t, v in by_type.items()}}

    audit_out = {"summary": summary, "items": [{k: v for k, v in r.items() if k != "_badness"} for r in results]}
    args.out_audit.parent.mkdir(parents=True, exist_ok=True)
    args.out_audit.write_text(json.dumps(audit_out, indent=2), encoding="utf-8")

    clean_qids = {r["qid"] for r in results if r["verdict"] == "ok"}
    with args.out_clean.open("w", encoding="utf-8") as f:
        for item in items:
            if item["qid"] in clean_qids:
                f.write(json.dumps(item) + "\n")

    print(f"n={summary['total']}  " + "  ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    print("\nby type:")
    for t, v in sorted(by_type.items()):
        print(f"  {t:<8} " + "  ".join(f"{k}={v2}" for k, v2 in sorted(v.items())))

    worst = sorted((r for r in results if r["verdict"] != "ok"), key=lambda r: -r["_badness"])[: args.worst]
    print(f"\n{len(worst)} worst items:")
    for r in worst:
        print(f"\n[{r['verdict']}] {r['qid']} ({r['type']})")
        print(f"  Q: {r['question']}")
        print(f"  A: {r['reference_answer']}")
        print(f"  reason: {'; '.join(r['reasons'])}")

    print(f"\nwrote {args.out_audit} and {args.out_clean} ({len(clean_qids)} ok items)")


if __name__ == "__main__":
    main()
