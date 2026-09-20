"""Download a topical corpus of open-access arXiv papers into data/raw/ and record exactly which ones.

The original corpus was six hand-picked papers. A retrieval system is only meaningfully tested when the right paper
has to be found among many similar ones, so this grows the corpus to a few dozen on the same subject (machine
learning for drug discovery). It uses the public arXiv API, waits between requests as the API terms ask, skips
papers already present, and writes configs/corpus_arxiv.json so the same corpus can be re-fetched.

    python scripts/fetch_arxiv.py --target 50        # top up data/raw to 50 papers
    python scripts/fetch_arxiv.py --from-manifest    # re-fetch exactly the papers listed in the manifest
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW = REPO_ROOT / "data" / "raw"
MANIFEST = REPO_ROOT / "configs" / "corpus_arxiv.json"
API = "https://export.arxiv.org/api/query"
ATOM = {"a": "http://www.w3.org/2005/Atom"}
DELAY_S = 3.5          # arXiv asks for no more than one request every 3 seconds
MAX_PDF_MB = 12        # skip very large PDFs: they are mostly supplementary material and slow to parse
MIN_YEAR = 2019

# Same subject as the original six papers, split so no single sub-topic dominates the corpus.
QUERIES = [
    ("drug response prediction", 'abs:"drug response prediction"'),
    ("drug-target interaction", 'abs:"drug-target interaction"'),
    ("drug repurposing", 'abs:"drug repurposing" OR abs:"drug repositioning"'),
    ("drug synergy", 'abs:"drug synergy" OR abs:"drug combination"'),
    ("drug-drug interaction", 'abs:"drug-drug interaction"'),
    ("molecular property prediction", 'abs:"molecular property prediction"'),
    ("binding affinity", 'abs:"binding affinity" AND abs:"deep learning"'),
    ("virtual screening", 'abs:"virtual screening" AND abs:"neural network"'),
]


def _get(url: str, **kw) -> requests.Response:
    time.sleep(DELAY_S)
    return requests.get(url, timeout=60, headers={"User-Agent": "BioLit-research-corpus/1.0"}, **kw)


def search(query: str, n: int) -> list[dict]:
    params = {"search_query": query, "start": 0, "max_results": n, "sortBy": "relevance"}
    root = ET.fromstring(_get(f"{API}?{urllib.parse.urlencode(params)}").content)
    papers = []
    for e in root.findall("a:entry", ATOM):
        raw_id = e.findtext("a:id", "", ATOM).rsplit("/abs/", 1)[-1]
        arxiv_id = re.sub(r"v\d+$", "", raw_id)
        papers.append({
            "id": arxiv_id,
            "title": " ".join(e.findtext("a:title", "", ATOM).split()),
            "year": int(e.findtext("a:published", "0000", ATOM)[:4]),
            "abstract": " ".join(e.findtext("a:summary", "", ATOM).split()),
        })
    return papers


def download(arxiv_id: str) -> tuple[bool, str]:
    dest = RAW / f"arxiv_{arxiv_id}.pdf"
    r = _get(f"https://arxiv.org/pdf/{arxiv_id}")
    if r.status_code != 200 or not r.content.startswith(b"%PDF"):
        return False, f"HTTP {r.status_code}, not a PDF"
    if len(r.content) > MAX_PDF_MB * 1024 * 1024:
        return False, f"{len(r.content) / 1e6:.0f} MB, over the {MAX_PDF_MB} MB cap"
    dest.write_bytes(r.content)
    return True, f"{len(r.content) / 1e6:.1f} MB"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", type=int, default=50, help="total number of papers to have in data/raw")
    ap.add_argument("--per-query", type=int, default=12, help="candidates requested per topic query")
    ap.add_argument("--from-manifest", action="store_true", help="re-fetch the papers listed in the manifest instead of searching")
    args = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)

    have = {re.sub(r"^arxiv_", "", p.stem) for p in RAW.glob("arxiv_*.pdf")}
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else {"papers": []}
    known = {p["id"]: p for p in manifest["papers"]}

    if args.from_manifest:
        for pid, meta in known.items():
            if pid in have:
                continue
            ok, why = download(pid)
            print(f"[{'ok' if ok else 'skip'}] {pid} {meta['title'][:60]} ({why})", flush=True)
        return 0

    print(f"{len(have)} papers already in data/raw; target {args.target}", flush=True)
    seen: set[str] = set(have)
    candidates: list[tuple[str, dict]] = []
    for topic, q in QUERIES:
        try:
            found = search(q, args.per_query)
        except Exception as exc:  # network hiccup: keep what the other queries give
            print(f"[warn] search '{topic}' failed: {exc}", file=sys.stderr, flush=True)
            continue
        fresh = [p for p in found if p["id"] not in seen and p["year"] >= MIN_YEAR]
        seen.update(p["id"] for p in fresh)
        candidates += [(topic, p) for p in fresh]
        print(f"[search] {topic}: {len(found)} results, {len(fresh)} new (>= {MIN_YEAR})", flush=True)

    # Round-robin over topics so the corpus is balanced rather than filled by the first query.
    by_topic: dict[str, list[dict]] = {}
    for topic, p in candidates:
        by_topic.setdefault(topic, []).append(p)
    order = []
    while any(by_topic.values()):
        for topic in list(by_topic):
            if by_topic[topic]:
                order.append((topic, by_topic[topic].pop(0)))

    for topic, p in order:
        if len(have) >= args.target:
            break
        ok, why = download(p["id"])
        print(f"[{'ok' if ok else 'skip'}] {p['id']} {p['title'][:70]} ({why})", flush=True)
        if ok:
            have.add(p["id"])
            known[p["id"]] = {"id": p["id"], "title": p["title"], "year": p["year"], "topic": topic}

    manifest["papers"] = sorted(known.values(), key=lambda x: x["id"])
    manifest["note"] = ("Papers fetched by scripts/fetch_arxiv.py. The six originals predate the script and are not listed "
                        "unless re-fetched; data/raw is the source of truth for what was indexed.")
    MANIFEST.parent.mkdir(exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"done: {len(have)} papers in data/raw ({len(known)} recorded in {MANIFEST.relative_to(REPO_ROOT).as_posix()})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
