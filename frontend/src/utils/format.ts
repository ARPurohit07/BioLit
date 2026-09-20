/** Format a 0-1 ratio as a percentage; null/undefined means "not measured" and renders as an em dash. */
export function formatPercent(value: number | null | undefined, digits = 0): string {
  return value === null || value === undefined ? "—" : `${(value * 100).toFixed(digits)}%`;
}

/** Tooltip for a metric that needs claim verification, explaining an em dash. */
export const NOT_VERIFIED_HINT =
  "Not measured: claims were not checked against the evidence in this mode. Use High-Faithfulness mode to verify them.";
