import type { MetricWithCI, RagEval, RetrievalGroup } from "../types/api";

const STRATEGIES: { key: string; label: string; note: string }[] = [
  { key: "bm25", label: "BM25", note: "keyword search" },
  { key: "dense", label: "Dense", note: "bge-small embeddings" },
  { key: "hybrid", label: "Hybrid", note: "dense + BM25, rank fusion" },
  { key: "hybrid+rerank", label: "Hybrid + reranker", note: "what Balanced mode uses" },
];

const RAGAS_ROWS: { key: string; label: string; hint: string }[] = [
  { key: "faithfulness", label: "Faithfulness", hint: "Share of the answer's statements supported by the retrieved context." },
  { key: "answer_relevancy", label: "Answer relevancy", hint: "Does the answer address the question." },
  { key: "context_precision", label: "Context precision", hint: "Are the useful passages ranked at the top of what was retrieved." },
  { key: "context_recall", label: "Context recall", hint: "Does the retrieved context contain what the reference answer needs." },
  { key: "factual_correctness", label: "Factual correctness", hint: "Agreement of the answer's claims with the reference answer." },
];

const MODE_LABEL: Record<string, string> = {
  fast: "Fast",
  balanced: "Balanced",
  high_faithfulness: "High Faithfulness",
};

const pct = (v: number | null | undefined) => (v === null || v === undefined ? "—" : `${(v * 100).toFixed(0)}%`);
const dec = (v: number | null | undefined) => (v === null || v === undefined ? "—" : v.toFixed(2));

function withCI(m: MetricWithCI | undefined) {
  if (!m) return "—";
  return (
    <>
      {m.mean.toFixed(2)}{" "}
      <span className="eval-ci">
        [{m.ci95[0].toFixed(2)}–{m.ci95[1].toFixed(2)}]
      </span>
    </>
  );
}

const group = (arm: Record<string, RetrievalGroup | number> | undefined, name: string) => {
  const g = arm?.[name];
  return typeof g === "object" ? g : undefined;
};

export default function RagEvalSections({ data }: { data: RagEval }) {
  const { retrieval, ragas } = data;
  const best = retrieval?.arms["hybrid+rerank"];

  return (
    <>
      {retrieval && (
        <div className="eval-section">
          <h3>Retrieval on a labelled question set</h3>
          <p className="eval-note">
            {retrieval.eval_set.n} questions from {retrieval.eval_set.papers} papers (
            {Object.entries(retrieval.eval_set.by_type)
              .map(([k, v]) => `${v} ${k}`)
              .join(", ")}
            ) over {retrieval.corpus_chunks.toLocaleString()} indexed chunks. Each question has one known source chunk.
            Brackets are 95% bootstrap intervals.
          </p>
          <div className="card eval-chart-card">
            <table className="eval-latency-table">
              <thead>
                <tr>
                  <th>Strategy</th>
                  <th>Recall@1</th>
                  <th>Recall@5</th>
                  <th>Recall@10</th>
                  <th>MRR</th>
                  <th>nDCG@10</th>
                  <th title="Right paper found, ignoring which chunk">Paper Recall@5</th>
                  <th>Median ms</th>
                </tr>
              </thead>
              <tbody>
                {STRATEGIES.filter((s) => retrieval.arms[s.key]).map((s) => {
                  const arm = retrieval.arms[s.key];
                  const chunk = group(arm, "chunk/all");
                  const paper = group(arm, "paper/all");
                  return (
                    <tr key={s.key} title={s.note}>
                      <td>{s.label}</td>
                      <td className="mono">{withCI(chunk?.["recall@1"])}</td>
                      <td className="mono">{withCI(chunk?.["recall@5"])}</td>
                      <td className="mono">{withCI(chunk?.["recall@10"])}</td>
                      <td className="mono">{withCI(chunk?.mrr)}</td>
                      <td className="mono">{withCI(chunk?.["ndcg@10"])}</td>
                      <td className="mono">{withCI(paper?.["recall@5"])}</td>
                      <td className="mono">{typeof arm.latency_ms_median === "number" ? arm.latency_ms_median.toFixed(0) : "—"}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          {best && (
            <>
              <h3 className="eval-subheading">Hybrid + reranker, by kind of question</h3>
              <div className="card eval-chart-card">
                <table className="eval-latency-table">
                  <thead>
                    <tr>
                      <th>Slice</th>
                      <th>n</th>
                      <th>Recall@1</th>
                      <th>Recall@5</th>
                      <th>MRR</th>
                    </tr>
                  </thead>
                  <tbody>
                    {[
                      ["chunk/type=text", "Text passages"],
                      ["chunk/type=table", "Tables"],
                      ["chunk/type=figure", "Figures (caption)"],
                      ["chunk/easy", "Easy: question shares many words with the passage"],
                      ["chunk/hard", "Hard: little wording in common"],
                    ]
                      .map(([k, label]) => ({ k, label, g: group(best, k) }))
                      .filter((r) => r.g && r.g.n > 0)
                      .map(({ k, label, g }) => (
                        <tr key={k}>
                          <td>{label}</td>
                          <td className="mono">{g!.n}</td>
                          <td className="mono">{withCI(g!["recall@1"])}</td>
                          <td className="mono">{withCI(g!["recall@5"])}</td>
                          <td className="mono">{withCI(g!.mrr)}</td>
                        </tr>
                      ))}
                  </tbody>
                </table>
              </div>
            </>
          )}
          <ul className="eval-caveats">
            {retrieval.caveats.map((c) => (
              <li key={c}>{c}</li>
            ))}
          </ul>
        </div>
      )}

      {ragas && Object.keys(ragas.modes).length > 0 && (
        <div className="eval-section">
          <h3>Answer quality (RAGAS)</h3>
          <p className="eval-note">
            Judged by the local model <span className="mono">{ragas.judge}</span>, so scores are noisy. “Scored” counts the
            answers the judge could evaluate; the rest failed to parse and are left out, not counted as zero.
          </p>
          <div className="card eval-chart-card">
            <table className="eval-latency-table">
              <thead>
                <tr>
                  <th>Metric</th>
                  {Object.keys(ragas.modes).map((m) => (
                    <th key={m}>
                      {MODE_LABEL[m] ?? m} <span className="eval-ci">(n={ragas.modes[m].n_questions})</span>
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {RAGAS_ROWS.map((row) => {
                  const v = ragas.judge_validity?.[row.key];
                  return (
                  <tr key={row.key} title={row.hint}>
                    <td>
                      {row.label}
                      {v && !v.separates && (
                        <span
                          className="eval-warn"
                          title="Judge check failed: this judge scored the right case and a deliberately wrong case too close together (see below). Treat this number as unreliable."
                        >
                          {" "}
                          unreliable
                        </span>
                      )}
                      {!v && (
                        <span className="eval-ci" title="No judge-validity control was run for this metric.">
                          {" "}
                          unchecked
                        </span>
                      )}
                    </td>
                    {Object.entries(ragas.modes).map(([m, s]) => {
                      const r = s.ragas[row.key];
                      return (
                        <td key={m} className="mono">
                          {dec(r?.mean)} <span className="eval-ci">({r?.n_scored ?? 0} scored)</span>
                        </td>
                      );
                    })}
                  </tr>
                  );
                })}
                <tr title="Whether the labelled source chunk was among the chunks handed to the model">
                  <td>Source chunk in context</td>
                  {Object.values(ragas.modes).map((s, i) => (
                    <td key={i} className="mono">{pct(s.retrieval.source_chunk_in_context)}</td>
                  ))}
                </tr>
                <tr title="Share of claims that cite a retrieved block">
                  <td>Citation coverage</td>
                  {Object.values(ragas.modes).map((s, i) => (
                    <td key={i} className="mono">{pct(s.citations.citation_coverage)}</td>
                  ))}
                </tr>
                <tr title="Claims the fast text-overlap check flags for a manual look">
                  <td>Claims flagged “check source”</td>
                  {Object.values(ragas.modes).map((s, i) => (
                    <td key={i} className="mono">{pct(s.citations.flagged_share)}</td>
                  ))}
                </tr>
              </tbody>
            </table>
          </div>

          {ragas.judge_validity && (
            <>
              <h3 className="eval-subheading">Can the judge be trusted?</h3>
              <p className="eval-note">
                Control test: each metric scores a case where everything is right and a case where one input is made
                wrong (an unrelated context, question or reference). A metric whose two scores are less than 0.4 apart
                is marked unreliable.
              </p>
              <div className="card eval-chart-card">
                <table className="eval-latency-table">
                  <thead>
                    <tr>
                      <th>Metric</th>
                      <th>Wrong input</th>
                      <th>Right case</th>
                      <th>Wrong case</th>
                      <th>Gap</th>
                      <th>Separates</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(ragas.judge_validity).map(([k, v]) => (
                      <tr key={k}>
                        <td>{k.replace("_", " ")}</td>
                        <td>{v.negative_control}</td>
                        <td className="mono">{dec(v.positive_mean)}</td>
                        <td className="mono">{dec(v.negative_mean)}</td>
                        <td className="mono">{dec(v.gap)}</td>
                        <td className="mono">{v.separates ? "yes" : "no"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          )}
          <ul className="eval-caveats">
            {ragas.caveats.map((c) => (
              <li key={c}>{c}</li>
            ))}
          </ul>
        </div>
      )}
    </>
  );
}
