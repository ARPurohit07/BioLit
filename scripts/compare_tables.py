"""Re-answer the table questions with the current index and compare with the saved answers from the previous index.

Same prompts, same generator, same judge: only the index differs. Value-only references are checked directly for the value;
faithfulness is scored with the validated judge.

    python scripts/compare_tables.py --tag v2
"""
from __future__ import annotations

import argparse
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_ragas as er  # noqa: E402
import diagnose_correctness as dc  # noqa: E402
from compare_variant import faithfulness, generate, value_present  # noqa: E402
from eval_prompt_ab import paired  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="v2", help="the saved answers to compare against")
    a = ap.parse_args()
    qs = [q for h in ("dev", "test") for q in er.read_jsonl(er.EVAL_DIR / f"eval_set_{h}.jsonl") if q["type"] == "table"]
    qids = {q["qid"] for q in qs}
    old = {r["qid"]: r for h in ("dev", "test") for r in er.read_jsonl(er.EVAL_DIR / f"answers_{a.tag}_{h}.jsonl") if r["qid"] in qids}
    newp = er.EVAL_DIR / "answers_tables_new.jsonl"
    generate(qs, newp)
    new = {r["qid"]: r for r in er.read_jsonl(newp)}
    faith_old = {r["qid"]: r["faithfulness"] for h in ("dev", "test") for r in er.read_jsonl(er.EVAL_DIR / f"faith_{h}_{a.tag}.jsonl") if r["qid"] in qids}
    faith_new = faithfulness(list(new.values()), er.EVAL_DIR / "faith_tables_new.jsonl")
    print(f"\ntable questions: {len(qids)} | answered before {len(old)}, now {len(new)}")
    value = [q for q in qids if q in old and q in new and dc.is_value(old[q]["reference_answer"])]
    ok_o = {q: value_present(old[q]["reference_answer"], old[q]["answer"]) for q in value}
    ok_n = {q: value_present(new[q]["reference_answer"], new[q]["answer"]) for q in value}
    print(f"value questions ({len(value)}): value present  before {sum(ok_o.values())}  ->  now {sum(ok_n.values())}  "
          f"(gained {sum(ok_n[q] and not ok_o[q] for q in value)}, lost {sum(ok_o[q] and not ok_n[q] for q in value)})")
    ab = lambda d: sum("insufficient evidence" in r["answer"].lower() for r in d.values())
    print(f"abstentions: before {ab({q: old[q] for q in old})}  now {ab(new)}")
    fo, fn = [x for x in faith_old.values() if x is not None], [x for x in faith_new.values() if x is not None]
    print(f"faithfulness: before {st.fmean(fo):.3f} ({len(fo)})  now {st.fmean(fn):.3f} ({len(fn)})")
    print("paired faithfulness:", paired({q: {"m": v} for q, v in faith_old.items()}, {q: {"m": v} for q, v in faith_new.items()}, "m"))
    for q in value:
        if ok_n[q] != ok_o[q]:
            print(f"  [{q}] {'GAINED' if ok_n[q] else 'LOST  '} ref {old[q]['reference_answer'][:22]!r} | now: {er.clean_answer(new[q]['answer'])[:80]!r}")


if __name__ == "__main__":
    main()
