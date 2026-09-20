"""Smoke-test every backend endpoint against a RUNNING server and validate what comes back.

Beyond status codes it checks the content: evidence ids run 1..n, every claim's citations point at real evidence,
metrics are in range and consistent with the claims, latency fields make sense, retrieval respects document filters,
error paths return 4xx (never 500), and an upload -> index -> query -> delete round trip leaves the index as it found it.

    python scripts/smoke_test_api.py                 # everything (about 6-8 minutes with a 3B model on a laptop GPU)
    python scripts/smoke_test_api.py --quick         # skips High-Faithfulness and literature review
    python scripts/smoke_test_api.py --skip-lifecycle  # do not upload/index/delete a synthetic PDF

Exit code 0 = no FAIL. WARN lines are answer-quality observations (e.g. an answer with no citation), not defects.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import time
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
STATUSES = {"NOT_VERIFIED", "SUPPORTED", "PARTIALLY_SUPPORTED", "UNSUPPORTED", "CONTRADICTED"}
DOC_STATUSES = {"uploaded", "parsing", "indexing", "indexed", "failed", "scanned_needs_ocr"}
EXPECTED_PATHS = ["/api/health", "/api/documents/upload", "/api/documents/", "/api/documents/{document_id}",
                  "/api/documents/index", "/api/query", "/api/compare", "/api/literature-review", "/api/verify",
                  "/api/evaluation/"]

results: list[tuple[str, str, str]] = []  # (PASS|FAIL|WARN, name, detail)


def record(status: str, name: str, detail: str = "", secs: float | None = None) -> None:
    results.append((status, name, detail))
    t = f"{secs:>5.1f}s" if secs is not None else "      "
    print(f"[{status:<4}] {t}  {name}" + (f"  -- {detail}" if detail else ""), flush=True)


class Client:
    def __init__(self, base: str):
        self.base = base.rstrip("/")

    def call(self, method: str, path: str, timeout: int = 600, **kw):
        t0 = time.time()
        try:
            r = requests.request(method, f"{self.base}{path}", timeout=timeout, **kw)
        except Exception as exc:  # timeout / connection refused
            return None, time.time() - t0, f"{type(exc).__name__}: {exc}"
        return r, time.time() - t0, ""


def _json(r):
    try:
        return r.json()
    except Exception:
        return None


def expect(c: Client, name: str, method: str, path: str, want: set[int], timeout: int = 600, report: bool = False, **kw):
    """Call, require a status in `want`; returns (json_or_None, secs). Never treats a 5xx as acceptable.
    Pass report=True for checks that are only about the status code, so a pass is printed and counted."""
    r, secs, err = c.call(method, path, timeout=timeout, **kw)
    if r is None:
        record("FAIL", name, err, secs)
        return None, secs
    body = _json(r)
    if r.status_code not in want:
        detail = f"HTTP {r.status_code}, expected {sorted(want)}"
        if isinstance(body, dict) and "detail" in body:
            detail += f" | detail: {str(body['detail'])[:260]}"
        record("FAIL", name, detail, secs)
        return None, secs
    if report:
        record("PASS", name, f"HTTP {r.status_code}", secs)
    return body, secs


def validate_query_response(d: dict, mode: str | None = None, doc_ids: list[str] | None = None) -> list[str]:
    p: list[str] = []
    ans = d.get("answer_markdown", "")
    if not ans.strip():
        p.append("empty answer")
    if "The following statements from that answer" in ans:
        p.append("regeneration-prompt echo leaked into the answer")
    ev = d.get("evidence", [])
    ids = [e["citation_id"] for e in ev]
    if not ev:
        p.append("no evidence returned")
    if ids != list(range(1, len(ids) + 1)):
        p.append(f"evidence citation ids are not 1..n: {ids}")
    if d.get("num_sources") != len(ev):
        p.append(f"num_sources {d.get('num_sources')} != len(evidence) {len(ev)}")
    for e in ev:
        if not e["text"].strip() or not e["document_id"] or e["page_number"] < 1:
            p.append(f"evidence [{e['citation_id']}] has empty text/document or page < 1")
            break
    if doc_ids is not None:
        stray = {e["document_id"] for e in ev} - set(doc_ids)
        if stray:
            p.append(f"evidence from documents outside the filter: {sorted(stray)}")
    claims = d.get("claims", [])
    valid, seen = set(ids), set()
    for c in claims:
        if c["claim_id"] in seen:
            p.append(f"duplicate claim_id {c['claim_id']}")
        seen.add(c["claim_id"])
        if c["status"] not in STATUSES:
            p.append(f"claim {c['claim_id']} has unknown status {c['status']}")
        if not set(c["citation_ids"]) <= valid:
            p.append(f"claim {c['claim_id']} cites blocks {c['citation_ids']} not in evidence {sorted(valid)}")
        if not c["text"].strip():
            p.append(f"claim {c['claim_id']} is empty")
        if c.get("grounding") not in (None, "strong", "weak", "none"):
            p.append(f"claim {c['claim_id']} has unknown grounding {c.get('grounding')!r}")
        if not c["citation_ids"] and c.get("grounding") not in (None, "none"):
            p.append(f"claim {c['claim_id']} cites nothing but was rated grounded ({c.get('grounding')})")
    m = d.get("citation_metrics", {})
    n_flagged = sum(c.get("grounding") == "none" for c in claims)
    if claims and m.get("flagged_claims") != n_flagged:
        p.append(f"flagged_claims {m.get('flagged_claims')} != {n_flagged} claims with grounding 'none'")
    n_verified = sum(c["status"] != "NOT_VERIFIED" for c in claims)
    if m.get("verified_claims") != n_verified:
        p.append(f"verified_claims {m.get('verified_claims')} != {n_verified} claims with a real status")
    if not 0.0 <= m.get("citation_coverage", -1) <= 1.0:
        p.append(f"citation_coverage={m.get('citation_coverage')} outside [0,1]")
    for k in ("citation_precision", "faithfulness", "unsupported_claim_rate"):
        v = m.get(k, "missing")
        if n_verified == 0 and claims:
            if v is not None:
                p.append(f"{k}={v} reported although no claim was verified (should be null = not measured)")
        elif not (isinstance(v, (int, float)) and 0.0 <= v <= 1.0):
            p.append(f"metric {k}={v} is not a number in [0,1]")
    if m.get("total_claims") != len(claims):
        p.append(f"total_claims {m.get('total_claims')} != {len(claims)} claims")
    if m.get("total_citations") != sum(len(c["citation_ids"]) for c in claims):
        p.append("total_citations does not match the claims' citation ids")
    if claims:
        cov = sum(bool(c["citation_ids"]) for c in claims) / len(claims)
        if abs(cov - m.get("citation_coverage", -1)) > 1e-6:
            p.append(f"citation_coverage {m.get('citation_coverage'):.3f} != recomputed {cov:.3f}")
    lat = d.get("latency", {})
    if any(v < 0 for v in lat.values() if isinstance(v, (int, float))):
        p.append("negative latency value")
    if lat.get("total_latency_ms", 0) + 1 < lat.get("generation_latency_ms", 0):
        p.append("total latency is smaller than generation latency")
    if mode:
        if d.get("mode") != mode:
            p.append(f"mode echoed as {d.get('mode')}, sent {mode}")
        if mode == "high_faithfulness" and claims and lat.get("verification_latency_ms", 0) <= 0:
            p.append("high_faithfulness reported no verification time")
        if mode == "high_faithfulness" and n_verified != len(claims):
            p.append(f"high_faithfulness left {len(claims) - n_verified} claims NOT_VERIFIED")
        if mode in ("fast", "balanced") and n_verified != 0:
            p.append(f"{mode} mode produced {n_verified} verified claims (it should not verify)")
        if mode != "high_faithfulness" and lat.get("verification_latency_ms", 0) != 0:
            p.append(f"{mode} mode reported verification time (it should not verify)")
    return p


def check_query_result(name: str, body, secs: float, **kw) -> None:
    if body is None:
        return
    problems = validate_query_response(body, **kw)
    n_claims = len(body.get("claims", []))
    cited = sum(bool(c["citation_ids"]) for c in body.get("claims", []))
    info = f"{len(body['evidence'])} evidence, {n_claims} claims ({cited} cited)"
    if problems:
        record("FAIL", name, "; ".join(problems)[:400], secs)
    else:
        record("PASS", name, info, secs)
        if n_claims and cited == 0:
            record("WARN", f"{name}: answer has no citations", "quality observation, not a defect")


def run(args) -> int:
    c = Client(args.base)

    # ------------------------------------------------------------------ infrastructure
    r, secs, err = c.call("GET", "/api/health", timeout=30)
    if r is None:
        print(f"Backend not reachable at {args.base}: {err}\nStart it first (see README), then re-run.")
        return 2
    h = _json(r) or {}
    ok = r.status_code == 200 and h.get("status") == "ok" and h.get("ollama_available") is True \
        and h.get("num_indexed_documents", 0) > 0 and h.get("num_indexed_chunks", 0) > 0
    record("PASS" if ok else "FAIL", "GET /api/health", str(h), secs)
    if h.get("num_indexed_documents", 0) == 0:
        print("No documents indexed; the query tests need at least two. Aborting.")
        return 2
    base_docs, base_chunks = h["num_indexed_documents"], h["num_indexed_chunks"]

    spec, secs = expect(c, "GET /openapi.json", "GET", "/openapi.json", {200}, timeout=30)
    if spec is not None:
        missing = [p for p in EXPECTED_PATHS if p not in spec.get("paths", {})]
        record("FAIL" if missing else "PASS", "OpenAPI lists every expected route", f"missing: {missing}" if missing else f"{len(spec['paths'])} paths", secs)
    r, secs, err = c.call("GET", "/docs", timeout=30)
    record("PASS" if r is not None and r.status_code == 200 else "FAIL", "GET /docs (Swagger UI)", err, secs)

    # ------------------------------------------------------------------ documents
    body, secs = expect(c, "GET /api/documents/", "GET", "/api/documents/", {200}, timeout=30)
    docs: list[dict] = []
    if body is not None:
        docs = body["documents"]
        problems = []
        if body["total"] != len(docs):
            problems.append(f"total {body['total']} != {len(docs)} documents")
        for d in docs:
            if d["status"] not in DOC_STATUSES:
                problems.append(f"{d['document_id']}: unknown status {d['status']}")
            if d["status"] == "indexed" and (d["num_chunks"] <= 0 or not d["title"].strip()):
                problems.append(f"{d['document_id']}: indexed but no chunks/title")
        if len(docs) != base_docs:
            problems.append(f"list has {len(docs)} documents but /api/health says {base_docs}")
        record("FAIL" if problems else "PASS", "documents list is valid and matches /api/health", "; ".join(problems) or f"{len(docs)} documents", secs)
    body, secs = expect(c, "GET /api/documents (no trailing slash, as the UI calls it)", "GET", "/api/documents", {200}, report=True, timeout=30)

    expect(c, "upload a non-PDF is rejected with 400", "POST", "/api/documents/upload", {400}, report=True, timeout=30,
           files={"file": ("notes.txt", b"hello", "text/plain")})
    expect(c, "upload an empty PDF is rejected with 400", "POST", "/api/documents/upload", {400}, report=True, timeout=30,
           files={"file": ("empty.pdf", b"", "application/pdf")})
    expect(c, "delete an unknown document returns 404", "DELETE", "/api/documents/does-not-exist", {404}, report=True, timeout=30)
    expect(c, "index an unknown document id does not crash", "POST", "/api/documents/index", {200}, report=True, timeout=60,
           json={"document_ids": ["does-not-exist"]})

    ids = [d["document_id"] for d in docs if d["status"] == "indexed"]
    if len(ids) < 2:
        record("FAIL", "need at least 2 indexed documents for the query tests", f"found {len(ids)}")
        return 1
    doc_a, doc_b = ids[0], ids[1]
    q = "How does the MSDA plug-in improve zero-shot drug response prediction?"

    # ------------------------------------------------------------------ query
    expect(c, "query without a question is rejected with 422", "POST", "/api/query", {422}, report=True, timeout=30, json={"mode": "fast"})
    expect(c, "query with an invalid mode is rejected with 422", "POST", "/api/query", {422}, report=True, timeout=30, json={"question": q, "mode": "turbo"})
    expect(c, "query with an invalid query_type is rejected with 422", "POST", "/api/query", {422}, report=True, timeout=30, json={"question": q, "query_type": "nonsense"})

    for mode in ("fast", "balanced") + (() if args.quick else ("high_faithfulness",)):
        body, secs = expect(c, f"POST /api/query mode={mode}", "POST", "/api/query", {200}, json={"question": q, "mode": mode})
        check_query_result(f"POST /api/query mode={mode} returns a valid answer", body, secs, mode=mode)

    body, secs = expect(c, "POST /api/query restricted to one document", "POST", "/api/query", {200},
                        json={"question": "What methods are used?", "mode": "balanced", "document_ids": [doc_a]})
    check_query_result("query with document_ids filter only returns that document's evidence", body, secs, mode="balanced", doc_ids=[doc_a])

    body, secs = expect(c, "POST /api/query with an unknown document id", "POST", "/api/query", {200, 404, 422},
                        json={"question": "What methods are used?", "mode": "fast", "document_ids": ["does-not-exist"]})
    if body is not None:
        record("PASS", "query with an unknown document id is handled without a 5xx", f"answer: {body.get('answer_markdown', '')[:70]!r}", secs)

    for qtype in ("summarize", "limitations", "research_gaps", "structured_table", "conflict_detection", "question_answering"):
        body, secs = expect(c, f"POST /api/query type={qtype}", "POST", "/api/query", {200},
                            json={"question": "drug repurposing with machine learning", "mode": "balanced", "query_type": qtype})
        check_query_result(f"query_type={qtype} returns a valid answer", body, secs, mode="balanced")

    # ------------------------------------------------------------------ compare (the UI's request shape)
    expect(c, "compare with an invalid aspect is rejected with 422", "POST", "/api/compare", {422}, report=True, timeout=30,
           json={"document_ids": [doc_a, doc_b], "aspect": "nonsense", "mode": "balanced"})
    expect(c, "compare without document_ids is rejected with 422", "POST", "/api/compare", {422}, report=True, timeout=30,
           json={"aspect": "compare_methodology", "mode": "balanced"})
    combos = [("compare_methodology", [doc_a, doc_b]), ("compare_results", [doc_a, doc_b]), ("compare_papers", [doc_a, doc_b]),
              ("compare_methodology", ids)]
    if args.quick:
        combos = combos[:1]
    for aspect, sel in combos:
        name = f"POST /api/compare {aspect} on {len(sel)} papers"
        body, secs = expect(c, name, "POST", "/api/compare", {200}, json={"document_ids": sel, "aspect": aspect, "mode": "balanced"})
        check_query_result(f"{name} returns a valid comparison", body, secs, mode="balanced", doc_ids=sel)
    body, secs = expect(c, "POST /api/compare with a single document (UI requires 2)", "POST", "/api/compare", {200, 400, 422},
                        json={"document_ids": [doc_a], "aspect": "compare_methodology", "mode": "fast"})
    if body is not None and "answer_markdown" in body:
        check_query_result("compare with 1 document still returns a valid result", body, secs, mode="fast")
    if not args.quick:
        body, secs = expect(c, "POST /api/compare in high_faithfulness mode", "POST", "/api/compare", {200},
                            json={"document_ids": [doc_a, doc_b], "aspect": "compare_results", "mode": "high_faithfulness"})
        check_query_result("compare in high_faithfulness mode returns a valid comparison", body, secs, mode="high_faithfulness", doc_ids=[doc_a, doc_b])

    # ------------------------------------------------------------------ literature review
    expect(c, "literature-review without a topic is rejected with 422", "POST", "/api/literature-review", {422}, report=True, timeout=30, json={"mode": "fast"})
    if not args.quick:
        body, secs = expect(c, "POST /api/literature-review", "POST", "/api/literature-review", {200},
                            json={"topic": "drug repurposing with machine learning", "mode": "balanced", "max_papers": 6})
        check_query_result("literature review returns a valid, cited review", body, secs, mode="balanced")
        if body is not None:
            record("PASS" if len(body["answer_markdown"]) > 400 else "FAIL", "literature review is a substantial document",
                   f"{len(body['answer_markdown'])} chars")

    # ------------------------------------------------------------------ verify
    body, secs = expect(c, "POST /api/query (to get real chunk ids for /verify)", "POST", "/api/query", {200},
                        json={"question": q, "mode": "fast"})
    if body is not None and body["evidence"]:
        e0 = body["evidence"][0]
        sentence = next((s.strip() for s in e0["text"].replace("\n", " ").split(". ") if 60 < len(s.strip()) < 240), None)
        if sentence:
            v, secs = expect(c, "POST /api/verify verbatim sentence", "POST", "/api/verify", {200},
                             json={"claim_text": sentence, "evidence_chunk_ids": [e0["chunk_id"]]})
            if v is not None:
                record("PASS" if v["status"] == "SUPPORTED" and v["rationale"] else "FAIL",
                       "verify: a sentence copied from the evidence is SUPPORTED", f"{v['status']}: {v['rationale'][:120]}", secs)
        v, secs = expect(c, "POST /api/verify invented claim", "POST", "/api/verify", {200},
                         json={"claim_text": "The study reported that the drug cured every patient within one week.", "evidence_chunk_ids": [e0["chunk_id"]]})
        if v is not None:
            record("PASS" if v["status"] in ("UNSUPPORTED", "CONTRADICTED") else "FAIL",
                   "verify: an invented claim is not SUPPORTED", v["status"], secs)
    v, secs = expect(c, "POST /api/verify with unknown chunk ids", "POST", "/api/verify", {200},
                     json={"claim_text": "Anything at all.", "evidence_chunk_ids": ["nope"]}, timeout=60)
    if v is not None:
        record("PASS" if v["status"] == "UNSUPPORTED" and v["rationale"] else "FAIL",
               "verify: unknown chunk ids give UNSUPPORTED with an explanation", v["status"], secs)
    expect(c, "verify without claim_text is rejected with 422", "POST", "/api/verify", {422}, report=True, timeout=30, json={"evidence_chunk_ids": ["x"]})

    # ------------------------------------------------------------------ evaluation
    body, secs = expect(c, "GET /api/evaluation (follows the redirect, as the UI does)", "GET", "/api/evaluation", {200}, timeout=30)
    if body is not None:
        problems = []
        cc = body.get("citation_comparison")
        if not body.get("evaluated"):
            problems.append("evaluated is false")
        if cc:
            for cfg in cc["configs"]:
                if not (0 <= cfg["cites_any"] <= cfg["n"] and 0 <= cfg["passes_checker"] <= cfg["n"]
                        and cfg["valid_ids"] <= cfg["cites_any"] and 0 <= cfg["avg_sentence_coverage"] <= 1
                        and cfg["val_n"] + cfg["test_n"] == cfg["n"]
                        and cfg["val_passes"] + cfg["test_passes"] == cfg["passes_checker"]):
                    problems.append(f"inconsistent counts in {cfg['name']}")
        else:
            problems.append("no citation_comparison served")
        record("FAIL" if problems else "PASS", "evaluation summary is populated and internally consistent",
               "; ".join(problems) or f"{len(cc['configs'])} configurations", secs)

    # ------------------------------------------------------------------ CORS (what the browser sends)
    r, secs, err = c.call("OPTIONS", "/api/compare", timeout=30, headers={
        "Origin": "http://localhost:5173", "Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": "content-type"})
    cors_ok = r is not None and r.status_code == 200 and r.headers.get("access-control-allow-origin") == "http://localhost:5173"
    record("PASS" if cors_ok else "FAIL", "CORS preflight from the frontend origin", err or (f"HTTP {r.status_code}" if r is not None else ""), secs)

    # ------------------------------------------------------------------ upload -> index -> query -> delete
    if not args.skip_lifecycle:
        lifecycle(c, base_docs, base_chunks)
    return 0


def lifecycle(c: Client, base_docs: int, base_chunks: int) -> None:
    try:
        import fitz  # PyMuPDF
    except ImportError:
        record("WARN", "upload lifecycle skipped", "PyMuPDF is not installed in this environment")
        return
    backup = Path(tempfile.mkdtemp(prefix="biolit_index_backup_"))
    shutil.copytree(REPO_ROOT / "data" / "index", backup / "index")
    shutil.copy2(REPO_ROOT / "data" / "metadata" / "biolit.db", backup / "biolit.db")
    print(f"       (index + database backed up to {backup} before the lifecycle test)", flush=True)

    pdf = fitz.open()
    text = ("Zebrafish caudal fin regeneration depends on a population of blastema cells that dedifferentiate after amputation. "
            "Researchers amputated fins of adult zebrafish and measured regrowth over fourteen days using calibrated microscopy. "
            "Regrowth rate was highest in the first four days and declined thereafter, reaching 0.42 millimetres per day on day three. "
            "Treatment with a retinoic acid inhibitor reduced regrowth by 61 percent relative to vehicle controls in this cohort. ") * 4
    for _ in range(2):
        page = pdf.new_page()
        page.insert_textbox(fitz.Rect(50, 50, 550, 780), "Results\n" + text, fontsize=10)
    pdf_bytes = pdf.tobytes()
    pdf.close()

    doc_id = None
    try:
        up, secs = expect(c, "lifecycle: upload a synthetic PDF", "POST", "/api/documents/upload", {200}, timeout=60,
                          files={"file": ("zebrafish_smoke_test.pdf", pdf_bytes, "application/pdf")})
        if up is None:
            return
        doc_id = up["document_id"]
        record("PASS" if up["status"] == "uploaded" else "FAIL", "lifecycle: upload reports status 'uploaded'", up["status"], secs)
        idx, secs = expect(c, "lifecycle: index the uploaded PDF", "POST", "/api/documents/index", {200}, timeout=300, json={"document_ids": [doc_id]})
        if idx is not None:
            res = idx["results"][0] if idx["results"] else {}
            record("PASS" if res.get("status") == "indexed" else "FAIL", "lifecycle: document became 'indexed'", str(res)[:200], secs)
        h = _json(c.call("GET", "/api/health", timeout=30)[0]) or {}
        record("PASS" if h.get("num_indexed_documents") == base_docs + 1 and h.get("num_indexed_chunks") > base_chunks else "FAIL",
               "lifecycle: health counts grew after indexing", f"docs {base_docs}->{h.get('num_indexed_documents')}, chunks {base_chunks}->{h.get('num_indexed_chunks')}")
        body, secs = expect(c, "lifecycle: query the new document", "POST", "/api/query", {200}, timeout=300,
                            json={"question": "How fast did zebrafish fins regrow and what reduced regrowth?", "mode": "balanced", "document_ids": [doc_id]})
        check_query_result("lifecycle: query over the new document returns its evidence", body, secs, mode="balanced", doc_ids=[doc_id])
        if body is not None:
            joined = " ".join(e["text"] for e in body["evidence"]).lower()
            record("PASS" if "zebrafish" in joined and "0.42" in body["answer_markdown"] + joined else "FAIL",
                   "lifecycle: retrieved evidence is from the uploaded text", "")
    finally:
        if doc_id:
            expect(c, "lifecycle: delete the synthetic document", "DELETE", f"/api/documents/{doc_id}", {200}, report=True, timeout=60)
            h = _json(c.call("GET", "/api/health", timeout=30)[0]) or {}
            restored = h.get("num_indexed_documents") == base_docs and h.get("num_indexed_chunks") == base_chunks
            record("PASS" if restored else "FAIL", "lifecycle: index returned to its original size after delete",
                   f"docs {h.get('num_indexed_documents')} (was {base_docs}), chunks {h.get('num_indexed_chunks')} (was {base_chunks})")
            if not restored:
                shutil.rmtree(REPO_ROOT / "data" / "index")
                shutil.copytree(backup / "index", REPO_ROOT / "data" / "index")
                shutil.copy2(backup / "biolit.db", REPO_ROOT / "data" / "metadata" / "biolit.db")
                record("WARN", "index files restored from backup", "RESTART the backend so it reloads the restored index")
            else:
                shutil.rmtree(backup, ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--quick", action="store_true", help="skip High-Faithfulness and literature review")
    ap.add_argument("--skip-lifecycle", action="store_true", help="do not upload/index/delete a synthetic PDF")
    args = ap.parse_args()
    code = run(args)
    fails = [r for r in results if r[0] == "FAIL"]
    warns = [r for r in results if r[0] == "WARN"]
    print(f"\n=== {sum(r[0] == 'PASS' for r in results)} passed, {len(fails)} failed, {len(warns)} warnings ===")
    for _, name, detail in fails:
        print(f"  FAIL: {name}  -- {detail}")
    return code or (1 if fails else 0)


if __name__ == "__main__":
    sys.exit(main())
