import type { CitationMetrics } from "../types/api";
import { GROUNDING_FLAG_HINT } from "../utils/format";

/** Shown only when faithfulness was not measured: how many claims the fast heuristic flags for a manual check. */
export default function FlaggedClaimsChip({ metrics }: { metrics: CitationMetrics }) {
  if (metrics.faithfulness !== null || metrics.flagged_claims === undefined) return null;
  return (
    <div className="stat-chip" title={GROUNDING_FLAG_HINT}>
      <span className="stat-label">Flagged to check</span>
      <span className="stat-value">
        {metrics.flagged_claims}/{metrics.total_claims}
      </span>
    </div>
  );
}
