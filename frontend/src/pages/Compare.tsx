import { useEffect, useState } from "react";
import { listDocuments, runCompare } from "../api/client";
import type { DocumentMetadata, QueryResponse, QueryType, RAGMode } from "../types/api";
import ModeSelector from "../components/ModeSelector";
import LoadingSpinner from "../components/LoadingSpinner";
import ErrorBanner from "../components/ErrorBanner";
import QueryResultPanel from "../components/QueryResultPanel";
import "../styles/results.css";

const ASPECTS: { value: QueryType; label: string }[] = [
  { value: "compare_methodology", label: "Compare methodology" },
  { value: "compare_results", label: "Compare results" },
  { value: "compare_papers", label: "Compare papers (general)" },
  { value: "conflict_detection", label: "Conflict detection" },
];

export default function Compare() {
  const [documents, setDocuments] = useState<DocumentMetadata[]>([]);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [aspect, setAspect] = useState<QueryType>("compare_methodology");
  const [mode, setMode] = useState<RAGMode>("balanced");
  const [result, setResult] = useState<QueryResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    listDocuments()
      .then((res) => setDocuments(res.documents))
      .catch(() => undefined);
  }, []);

  const toggle = (id: string) => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const handleSubmit = async () => {
    if (selected.size < 2) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await runCompare({
        document_ids: Array.from(selected),
        aspect,
        mode,
      });
      setResult(res);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="page">
      <div className="page-header">
        <h1>Paper Comparison</h1>
        <p>Compare methodology, results, or detect conflicts across two or more papers.</p>
      </div>

      {error && <ErrorBanner message={error} />}

      <div className="card query-form-card">
        <div className="form-field">
          <label>Documents to compare (select at least 2)</label>
          <div className="doc-scope-list">
            {documents.length === 0 && <p className="form-hint">No documents indexed yet.</p>}
            {documents.map((doc) => (
              <label key={doc.document_id} className="doc-scope-item">
                <input
                  type="checkbox"
                  checked={selected.has(doc.document_id)}
                  onChange={() => toggle(doc.document_id)}
                />
                {doc.title || doc.filename}
              </label>
            ))}
          </div>
          <p className="form-hint">{selected.size} selected</p>
        </div>

        <div className="form-field">
          <label htmlFor="aspect">Aspect</label>
          <select id="aspect" value={aspect} onChange={(e) => setAspect(e.target.value as QueryType)}>
            {ASPECTS.map((a) => (
              <option key={a.value} value={a.value}>
                {a.label}
              </option>
            ))}
          </select>
        </div>

        <div className="form-field">
          <label>Mode</label>
          <ModeSelector value={mode} onChange={setMode} />
        </div>

        <button
          className="btn btn-primary"
          onClick={handleSubmit}
          disabled={loading || selected.size < 2}
        >
          {loading ? "Comparing…" : "Compare"}
        </button>
        {loading && (
          <span className="inline-loading">
            <LoadingSpinner label="Running comparison" />
          </span>
        )}
      </div>

      {result && <QueryResultPanel result={result} />}
    </div>
  );
}
