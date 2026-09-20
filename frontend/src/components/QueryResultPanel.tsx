import type { QueryResponse } from "../types/api";
import MarkdownRenderer from "./MarkdownRenderer";
import LatencyTable from "./LatencyTable";
import { useEvidenceViewer } from "./EvidenceViewerContext";
import { formatPercent, NOT_VERIFIED_HINT } from "../utils/format";
import FlaggedClaimsChip from "./FlaggedClaimsChip";
import "../styles/results.css";

export default function QueryResultPanel({ result }: { result: QueryResponse }) {
  const { openCitation } = useEvidenceViewer();

  return (
    <div className="result-layout">
      <div className="card result-answer-card">
        <div className="result-stats">
          <div className="stat-chip">
            <span className="stat-label">Sources</span>
            <span className="stat-value">{result.num_sources}</span>
          </div>
          {result.tokens_per_second !== undefined && (
            <div className="stat-chip">
              <span className="stat-label">Tokens/sec</span>
              <span className="stat-value">{result.tokens_per_second.toFixed(1)}</span>
            </div>
          )}
          {result.citation_metrics && (
            <>
              <div
                className="stat-chip"
                title={result.citation_metrics.citation_precision === null ? NOT_VERIFIED_HINT : undefined}
              >
                <span className="stat-label">Citation precision</span>
                <span className="stat-value">
                  {formatPercent(result.citation_metrics.citation_precision)}
                </span>
              </div>
              <div className="stat-chip">
                <span className="stat-label">Citation coverage</span>
                <span className="stat-value">
                  {formatPercent(result.citation_metrics.citation_coverage)}
                </span>
              </div>
              <div
                className="stat-chip"
                title={result.citation_metrics.faithfulness === null ? NOT_VERIFIED_HINT : undefined}
              >
                <span className="stat-label">Faithfulness</span>
                <span className="stat-value">
                  {formatPercent(result.citation_metrics.faithfulness)}
                </span>
              </div>
              <FlaggedClaimsChip metrics={result.citation_metrics} />
            </>
          )}
        </div>
        <MarkdownRenderer
          markdown={result.answer_markdown}
          evidence={result.evidence}
          claims={result.claims}
        />
      </div>

      <div className="result-sidebar">
        <div className="card">
          <div className="result-section-title">Latency</div>
          <LatencyTable latency={result.latency} />
        </div>
        <div className="card">
          <div className="result-section-title">Evidence ({result.evidence.length})</div>
          <div className="evidence-list">
            {result.evidence.map((ev) => (
              <button
                key={ev.citation_id}
                className="evidence-list-item"
                onClick={() => openCitation(ev.citation_id, result.evidence, result.claims)}
              >
                <span className="evidence-list-item-title">
                  [{ev.citation_id}] {ev.document_title}
                </span>
                <span className="evidence-list-item-meta">
                  Page {ev.page_number} · {ev.section || "—"}
                </span>
              </button>
            ))}
            {result.evidence.length === 0 && (
              <p className="result-empty">No evidence returned.</p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
