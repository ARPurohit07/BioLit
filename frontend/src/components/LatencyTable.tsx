import type { LatencyBreakdown } from "../types/api";
import "./LatencyTable.css";

const ROWS: { key: keyof LatencyBreakdown; label: string }[] = [
  { key: "retrieval_latency_ms", label: "Retrieval" },
  { key: "dense_latency_ms", label: "Dense search" },
  { key: "bm25_latency_ms", label: "BM25 search" },
  { key: "rerank_latency_ms", label: "Rerank" },
  { key: "generation_latency_ms", label: "Generation" },
  { key: "verification_latency_ms", label: "Verification" },
  { key: "total_latency_ms", label: "Total" },
];

export default function LatencyTable({ latency }: { latency: LatencyBreakdown }) {
  return (
    <table className="latency-table">
      <tbody>
        {ROWS.map(({ key, label }) => (
          <tr key={key} className={key === "total_latency_ms" ? "latency-total" : undefined}>
            <td>{label}</td>
            <td className="mono latency-value">{latency[key].toFixed(0)} ms</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
