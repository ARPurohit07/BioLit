"""Rewrite each extracted table from its page image with a vision model, keeping only transcriptions that cannot add a number.

The PDF text extractor leaves empty cells in about 59% of tables (values split across columns, merged headers), and answers
about those tables lose accuracy. A vision model reads the same region as a person would. Because a model rewriting numbers
is a hazard in a medical tool, a transcription is accepted only if every number in it also appears in the raw text of that
page region: it may repair the layout, it cannot invent a value. Rejected tables keep their original extraction.
Results are cached in data/table_cache/<document_id>.json and picked up by ingestion (PDFLoader.load_structured).

    python scripts/transcribe_tables.py
"""
from __future__ import annotations

import base64
import json
import re
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import fitz
import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from backend.app.ingestion.structure import StructureParser, rows_from_markdown  # noqa: E402
from ingest import make_document_id  # noqa: E402

MODEL = "gemma4:31b-cloud"
PROMPT = ("Transcribe this table from a research paper into a Markdown table. Copy every number exactly as printed, "
          "including parenthesised or plus/minus values, in the same cell as the value it belongs to. Merge multi-row headers "
          "into one header row. Do not add, drop, round or correct anything. Output only the table.")
NUM = re.compile(r"\d+(?:\.\d+)?")


def digits(tokens: list[str]) -> set[str]:
    return {re.sub(r"\D", "", t).lstrip("0") or "0" for t in tokens}


def transcribe(png: bytes) -> str:
    body = {"model": MODEL, "stream": False, "prompt": PROMPT, "images": [base64.b64encode(png).decode()],
            "options": {"num_predict": 3000}}
    for attempt in range(3):
        try:
            r = requests.post("http://127.0.0.1:11434/api/generate", json=body, timeout=300)
            r.raise_for_status()
            return r.json().get("response", "")
        except Exception:
            if attempt == 2:
                return ""
    return ""


def jobs() -> list[tuple[Path, str, int, int, object, object]]:
    out = []
    tmp = Path(tempfile.mkdtemp())
    parser = StructureParser(tmp, REPO_ROOT)
    for pdf in sorted((REPO_ROOT / "data" / "raw").glob("*.pdf")):
        doc_id = make_document_id(pdf)
        with fitz.open(str(pdf)) as doc:
            for n, page in enumerate(doc, 1):
                s = parser.parse_page(page, doc_id, n)
                for idx, tb in enumerate(s.tables):
                    clip = (fitz.Rect(*tb.bbox) + (-6, -6, 6, 6)) & page.rect
                    out.append((pdf, doc_id, n, idx, tb, (page.get_pixmap(clip=clip, dpi=170).tobytes("png"), page.get_text("text", clip=clip))))
    return out


def main() -> None:
    cache_dir = REPO_ROOT / "data" / "table_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    todo = jobs()
    print(f"{len(todo)} tables found", flush=True)

    def work(job):
        pdf, doc_id, n, idx, tb, (png, truth) = job
        md = transcribe(png)
        rows = rows_from_markdown(md)
        extra = digits(NUM.findall(md)) - digits(NUM.findall(truth))
        ok = len(rows) >= 2 and max((len(r) for r in rows), default=0) >= 2 and not extra
        return doc_id, f"p{n}_t{idx}", {"markdown": md, "accepted": ok, "invented_numbers": sorted(extra)[:8], "caption": tb.caption}

    results: dict[str, dict] = {}
    with ThreadPoolExecutor(3) as pool:
        for i, (doc_id, key, entry) in enumerate(pool.map(work, todo), 1):
            results.setdefault(doc_id, {})[key] = entry
            if i % 10 == 0:
                print(f"  {i}/{len(todo)}", flush=True)
    for doc_id, entries in results.items():
        (cache_dir / f"{doc_id}.json").write_text(json.dumps(entries, indent=1, ensure_ascii=False), encoding="utf-8")
    acc = sum(e["accepted"] for d in results.values() for e in d.values())
    print(f"accepted {acc} of {len(todo)}; rejected tables keep their original extraction", flush=True)


if __name__ == "__main__":
    main()
