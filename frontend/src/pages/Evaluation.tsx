import { useEffect, useState } from "react";
import { getEvaluation } from "../api/client";
import type { EvaluationSummary } from "../types/api";
import LoadingSpinner from "../components/LoadingSpinner";
import ErrorBanner from "../components/ErrorBanner";
import BarChart from "../components/BarChart";
import { formatPercent } from "../utils/format";
import "./Evaluation.css";

const MODE_LABEL: Record<string, string> = {
  fast: "Fast",
  balanced: "Balanced",
  high_faithfulness: "High Faithfulness",
};

export default function Evaluation() {
  const [summary, setSummary] = useState<EvaluationSummary | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const load = () => {
    setLoading(true);
    setError(null);
    getEvaluation()
      .then(setSummary)
      .catch((err) => setError(err instanceof Error ? err.message : String(err)))
      .finally(() => setLoading(false));
  };

  useEffect(load, []);

  return (
    <div className="page">
      <div className="page-header">
        <h1>Evaluation</h1>
        <p>
          Citation behaviour of the model options, generation faithfulness, and latency across RAG
          modes.
        </p>
      </div>

      {error && <ErrorBanner message={error} onRetry={load} />}
      {loading && <LoadingSpinner label="Loading evaluation summary" />}

      {summary && !summary.evaluated && (
        <div className="eval-not-evaluated card">
          <h3>Not yet evaluated</h3>
          <p>
            {summary.note ??
              "No evaluation has been run yet. Run the evaluation pipeline to populate these metrics."}
          </p>
        </div>
      )}

      {summary && summary.evaluated && (
        <>
          {summary.variant !== "unknown" && summary.variant !== "none" && (
            <p className="eval-variant mono">variant: {summary.variant}</p>
          )}

          {summary.citation_comparison && (
            <div className="eval-section">
              <h3>Citation behaviour on held-out prompts</h3>
              <div className="card eval-chart-card">
                <table className="eval-latency-table">
                  <thead>
                    <tr>
                      <th>Configuration</th>
                      <th>Cites [n]</th>
                      <th>Valid ids</th>
                      <th>Passes checks</th>
                      <th>Val</th>
                      <th>Test</th>
                      <th>Sentences cited</th>
                      <th>Avg words</th>
                    </tr>
                  </thead>
                  <tbody>
                    {summary.citation_comparison.configs.map((c) => (
                      <tr key={c.name} title={c.description}>
                        <td>{c.name}</td>
                        <td className="mono">{c.cites_any}/{c.n}</td>
                        <td className="mono">{c.valid_ids}/{c.n}</td>
                        <td className="mono">{c.passes_checker}/{c.n}</td>
                        <td className="mono">{c.val_passes}/{c.val_n}</td>
                        <td className="mono">{c.test_passes}/{c.test_n}</td>
                        <td className="mono">{(c.avg_sentence_coverage * 100).toFixed(0)}%</td>
                        <td className="mono">{c.avg_answer_words.toFixed(0)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <p className="eval-note">{summary.citation_comparison.prompts}</p>
              <ul className="eval-caveats">
                {summary.citation_comparison.caveats.map((c) => (
                  <li key={c}>{c}</li>
                ))}
              </ul>
            </div>
          )}

          {!summary.retrieval_metrics && (
            <p className="eval-note">
              Retrieval quality (Recall@k, MRR, nDCG) has not been measured: it needs labelled
              relevance judgements, which this corpus does not have yet.
            </p>
          )}

          {summary.retrieval_metrics && (
            <div className="eval-section">
              <h3>Retrieval</h3>
              <div className="eval-stats-grid">
                <div className="stat-chip">
                  <span className="stat-label">Recall@1</span>
                  <span className="stat-value">
                    {(summary.retrieval_metrics.recall_at_1 * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="stat-chip">
                  <span className="stat-label">Recall@5</span>
                  <span className="stat-value">
                    {(summary.retrieval_metrics.recall_at_5 * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="stat-chip">
                  <span className="stat-label">Recall@10</span>
                  <span className="stat-value">
                    {(summary.retrieval_metrics.recall_at_10 * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="stat-chip">
                  <span className="stat-label">MRR</span>
                  <span className="stat-value">{summary.retrieval_metrics.mrr.toFixed(3)}</span>
                </div>
                {summary.retrieval_metrics.ndcg_at_10 !== undefined && (
                  <div className="stat-chip">
                    <span className="stat-label">nDCG@10</span>
                    <span className="stat-value">
                      {summary.retrieval_metrics.ndcg_at_10.toFixed(3)}
                    </span>
                  </div>
                )}
                <div className="stat-chip">
                  <span className="stat-label">Queries</span>
                  <span className="stat-value">{summary.retrieval_metrics.num_queries}</span>
                </div>
              </div>
            </div>
          )}

          {summary.generation_metrics && (
            <div className="eval-section">
              <h3>Generation</h3>
              <div className="eval-stats-grid">
                <div className="stat-chip">
                  <span className="stat-label">Citation precision</span>
                  <span className="stat-value">
                    {formatPercent(summary.generation_metrics.citation_precision, 1)}
                  </span>
                </div>
                <div className="stat-chip">
                  <span className="stat-label">Citation coverage</span>
                  <span className="stat-value">
                    {formatPercent(summary.generation_metrics.citation_coverage, 1)}
                  </span>
                </div>
                <div className="stat-chip">
                  <span className="stat-label">Faithfulness</span>
                  <span className="stat-value">
                    {formatPercent(summary.generation_metrics.faithfulness, 1)}
                  </span>
                </div>
                <div className="stat-chip">
                  <span className="stat-label">Unsupported claim rate</span>
                  <span className="stat-value">
                    {formatPercent(summary.generation_metrics.unsupported_claim_rate, 1)}
                  </span>
                </div>
                {summary.generation_metrics.answer_relevance_approx !== undefined && (
                  <div className="stat-chip">
                    <span className="stat-label">Answer relevance (approx.)</span>
                    <span className="stat-value">
                      {(summary.generation_metrics.answer_relevance_approx * 100).toFixed(1)}%
                    </span>
                  </div>
                )}
              </div>
              <p className="eval-note">{summary.generation_metrics.note}</p>
            </div>
          )}

          {summary.latency_by_mode.length > 0 && (
            <div className="eval-section">
              <h3>Latency by mode</h3>
              <div className="card eval-chart-card">
                <BarChart
                  data={summary.latency_by_mode.map((m) => ({
                    label: MODE_LABEL[m.mode] ?? m.mode,
                    value: m.total_latency_ms,
                  }))}
                  unit=" ms"
                  formatValue={(v) => v.toFixed(0)}
                />
              </div>

              {summary.latency_by_mode.some((m) => m.tokens_per_second !== undefined) && (
                <>
                  <h3 className="eval-subheading">Tokens/sec by mode</h3>
                  <div className="card eval-chart-card">
                    <BarChart
                      data={summary.latency_by_mode.map((m) => ({
                        label: MODE_LABEL[m.mode] ?? m.mode,
                        value: m.tokens_per_second ?? 0,
                      }))}
                      formatValue={(v) => v.toFixed(1)}
                      colorVar="--status-amber"
                    />
                  </div>
                </>
              )}

              <table className="eval-latency-table">
                <thead>
                  <tr>
                    <th>Mode</th>
                    <th>Retrieval</th>
                    <th>Generation</th>
                    <th>Verification</th>
                    <th>Total</th>
                    <th>Tokens/sec</th>
                  </tr>
                </thead>
                <tbody>
                  {summary.latency_by_mode.map((m) => (
                    <tr key={m.mode}>
                      <td>{MODE_LABEL[m.mode] ?? m.mode}</td>
                      <td className="mono">{m.retrieval_latency_ms.toFixed(0)} ms</td>
                      <td className="mono">{m.generation_latency_ms.toFixed(0)} ms</td>
                      <td className="mono">{m.verification_latency_ms.toFixed(0)} ms</td>
                      <td className="mono">{m.total_latency_ms.toFixed(0)} ms</td>
                      <td className="mono">
                        {m.tokens_per_second !== undefined ? m.tokens_per_second.toFixed(1) : "—"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <p className="eval-note">
                Faithfulness is reported in aggregate above, not broken out per mode — the
                backend's <span className="mono">LatencyBenchmarkEntry</span> schema does not
                carry a per-mode faithfulness value. See Settings for where this may change.
              </p>
            </div>
          )}

          {summary.note && <p className="eval-note eval-note-footer">{summary.note}</p>}
        </>
      )}
    </div>
  );
}
