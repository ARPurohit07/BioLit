/** Format a 0-1 ratio as a percentage; null/undefined means "not measured" and renders as an em dash. */
export function formatPercent(value: number | null | undefined, digits = 0): string {
  return value === null || value === undefined ? "—" : `${(value * 100).toFixed(digits)}%`;
}

/** Tooltip for a metric that needs claim verification, explaining an em dash. */
export const NOT_VERIFIED_HINT =
  "Not measured: claims were not checked against the evidence in this mode. Use High-Faithfulness mode to verify them.";

/** Tooltip for the heuristic "check source" flag, with the accuracy it was measured to have. */
export const GROUNDING_FLAG_HINT =
  "Not verified. A fast text-overlap check found little support for this in the cited passages, or it cites nothing. " +
  "Measured against the LLM verifier it is right about half the time (56% of flagged claims were unsupported vs 29% overall), " +
  "so treat it as a prompt to check the source, not a verdict. Use High-Faithfulness mode to verify.";
