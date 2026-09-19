import { useEffect, useRef, useState } from "react";
import { listDocuments, runQuery } from "../api/client";
import type { DocumentMetadata, QueryResponse, QueryType, RAGMode } from "../types/api";
import ModeSelector from "../components/ModeSelector";
import LoadingSpinner from "../components/LoadingSpinner";
import ErrorBanner from "../components/ErrorBanner";
import QueryResultPanel from "../components/QueryResultPanel";
import "../styles/results.css";

const QUERY_TYPES: { value: QueryType; label: string }[] = [
  { value: "question_answering", label: "Question answering" },
  { value: "summarize", label: "Summarize" },
  { value: "compare_papers", label: "Compare papers" },
  { value: "compare_methodology", label: "Compare methodology" },
  { value: "compare_results", label: "Compare results" },
  { value: "limitations", label: "Limitations" },
  { value: "research_gaps", label: "Research gaps" },
  { value: "literature_review", label: "Literature review" },
  { value: "structured_table", label: "Structured table" },
  { value: "conflict_detection", label: "Conflict detection" },
];

export default function Query() {
  const [documents, setDocuments] = useState<DocumentMetadata[]>([]);
  const [question, setQuestion] = useState("");
  const [queryType, setQueryType] = useState<QueryType>("question_answering");
  const [mode, setMode] = useState<RAGMode>("balanced");
  const [scope, setScope] = useState<Set<string>>(new Set());
  const [result, setResult] = useState<QueryResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    listDocuments()
      .then((res) => setDocuments(res.documents))
      .catch(() => undefined);
  }, []);

  useEffect(() => {
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, []);

  const toggleScope = (id: string) => {
    setScope((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const handleSubmit = async () => {
    if (!question.trim()) return;
    setLoading(true);
    setError(null);
    setResult(null);
    setElapsed(0);
    const start = Date.now();
    timerRef.current = setInterval(() => setElapsed((Date.now() - start) / 1000), 250);
    try {
      const res = await runQuery({
        question: question.trim(),
        mode,
        query_type: queryType,
        document_ids: scope.size > 0 ? Array.from(scope) : undefined,
      });
      setResult(res);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      if (timerRef.current) clearInterval(timerRef.current);
      setLoading(false);
    }
  };

  return (
    <div className="page">
      <div className="page-header">
        <h1>Research Query</h1>
        <p>Ask a question against your indexed corpus and get a cited answer.</p>
      </div>

      {error && <ErrorBanner message={error} />}

      <div className="card query-form-card">
        <div className="form-field">
          <label htmlFor="question">Question</label>
          <textarea
            id="question"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="e.g. What sample sizes were used across the included trials?"
          />
        </div>

        <div className="form-field">
          <label htmlFor="query-type">Query type</label>
          <select
            id="query-type"
            value={queryType}
            onChange={(e) => setQueryType(e.target.value as QueryType)}
          >
            {QUERY_TYPES.map((t) => (
              <option key={t.value} value={t.value}>
                {t.label}
              </option>
            ))}
          </select>
        </div>

        <div className="form-field">
          <label>Mode</label>
          <ModeSelector value={mode} onChange={setMode} />
        </div>

        <div className="form-field">
          <label>Document scope (optional — leave empty to search all indexed documents)</label>
          <div className="doc-scope-list">
            {documents.length === 0 && <p className="form-hint">No documents indexed yet.</p>}
            {documents.map((doc) => (
              <label key={doc.document_id} className="doc-scope-item">
                <input
                  type="checkbox"
                  checked={scope.has(doc.document_id)}
                  onChange={() => toggleScope(doc.document_id)}
                />
                {doc.title || doc.filename}
              </label>
            ))}
          </div>
        </div>

        <button className="btn btn-primary" onClick={handleSubmit} disabled={loading || !question.trim()}>
          {loading ? "Generating…" : "Generate"}
        </button>
        {loading && (
          <span className="inline-loading">
            <LoadingSpinner label="Running RAG pipeline" elapsedSeconds={elapsed} />
          </span>
        )}
      </div>

      {result && <QueryResultPanel result={result} />}
    </div>
  );
}
