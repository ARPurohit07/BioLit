import { useEffect, useState } from "react";
import { listDocuments, runLiteratureReview } from "../api/client";
import type { DocumentMetadata, QueryResponse, RAGMode } from "../types/api";
import ModeSelector from "../components/ModeSelector";
import LoadingSpinner from "../components/LoadingSpinner";
import ErrorBanner from "../components/ErrorBanner";
import MarkdownRenderer from "../components/MarkdownRenderer";
import LatencyTable from "../components/LatencyTable";
import { useEvidenceViewer } from "../components/EvidenceViewerContext";
import "../styles/results.css";

export default function LiteratureReview() {
  const [documents, setDocuments] = useState<DocumentMetadata[]>([]);
  const [topic, setTopic] = useState("");
  const [scope, setScope] = useState<Set<string>>(new Set());
  const [mode, setMode] = useState<RAGMode>("high_faithfulness");
  const [maxPapers, setMaxPapers] = useState(15);
  const [result, setResult] = useState<QueryResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const { openCitation } = useEvidenceViewer();

  useEffect(() => {
    listDocuments()
      .then((res) => setDocuments(res.documents))
      .catch(() => undefined);
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
    if (!topic.trim()) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await runLiteratureReview({
        topic: topic.trim(),
        document_ids: scope.size > 0 ? Array.from(scope) : undefined,
        mode,
        max_papers: maxPapers,
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
        <h1>Literature Review</h1>
        <p>Generate a synthesized review across your corpus for a given topic.</p>
      </div>

      {error && <ErrorBanner message={error} />}

      <div className="card query-form-card">
        <div className="form-field">
          <label htmlFor="topic">Topic</label>
          <input
            id="topic"
            type="text"
            value={topic}
            onChange={(e) => setTopic(e.target.value)}
            placeholder="e.g. CRISPR-based gene therapy delivery mechanisms"
          />
        </div>

        <div className="form-field">
          <label htmlFor="max-papers">Max papers</label>
          <input
            id="max-papers"
            type="number"
            min={1}
            max={50}
            value={maxPapers}
            onChange={(e) => setMaxPapers(Number(e.target.value) || 1)}
          />
        </div>

        <div className="form-field">
          <label>Mode</label>
          <ModeSelector value={mode} onChange={setMode} />
        </div>

        <div className="form-field">
          <label>Document scope (optional — leave empty to include all indexed documents)</label>
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

        <button className="btn btn-primary" onClick={handleSubmit} disabled={loading || !topic.trim()}>
          {loading ? "Generating…" : "Generate Review"}
        </button>
        {loading && (
          <span className="inline-loading">
            <LoadingSpinner label="Synthesizing review" />
          </span>
        )}
      </div>

      {result && (
        <div className="result-layout">
          <div className="card result-answer-card">
            <MarkdownRenderer
              markdown={result.answer_markdown}
              evidence={result.evidence}
              claims={result.claims}
            />

            <div className="result-section-title lit-review-references-title">
              References
            </div>
            <ol className="lit-review-references">
              {result.evidence.map((ev) => (
                <li key={ev.citation_id}>
                  <button
                    className="lit-review-ref-btn mono"
                    onClick={() => openCitation(ev.citation_id, result.evidence, result.claims)}
                  >
                    [{ev.citation_id}]
                  </button>{" "}
                  {ev.document_title} — page {ev.page_number}
                  {ev.section ? `, ${ev.section}` : ""}
                </li>
              ))}
              {result.evidence.length === 0 && (
                <p className="result-empty">No references returned.</p>
              )}
            </ol>
          </div>

          <div className="result-sidebar">
            <div className="card">
              <div className="result-section-title">Latency</div>
              <LatencyTable latency={result.latency} />
            </div>
            <div className="card">
              <div className="result-section-title">Summary</div>
              <div className="result-stats">
                <div className="stat-chip">
                  <span className="stat-label">Sources</span>
                  <span className="stat-value">{result.num_sources}</span>
                </div>
                {result.citation_metrics && (
                  <div className="stat-chip">
                    <span className="stat-label">Faithfulness</span>
                    <span className="stat-value">
                      {(result.citation_metrics.faithfulness * 100).toFixed(0)}%
                    </span>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
