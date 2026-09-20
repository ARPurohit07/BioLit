import { useEvidenceViewer } from "./EvidenceViewerContext";
import { ClaimStatusBadge } from "./StatusBadge";
import "./EvidenceViewer.css";

const STOPWORDS = new Set([
  "the", "and", "that", "this", "with", "from", "were", "was", "for", "are",
  "have", "has", "had", "not", "but", "which", "their", "these", "those",
  "into", "than", "then", "also", "such", "been", "being", "can", "could",
  "may", "might", "will", "would", "should", "there", "when", "while",
]);

function splitSentences(text: string): string[] {
  const parts = text.match(/[^.!?]+[.!?]*/g);
  return parts ? parts.map((p) => p.trim()).filter(Boolean) : [text];
}

function significantWords(text: string): Set<string> {
  return new Set(
    text
      .toLowerCase()
      .replace(/[^a-z0-9\s]/g, " ")
      .split(/\s+/)
      .filter((w) => w.length > 3 && !STOPWORDS.has(w)),
  );
}

/** Best-effort: highlight the evidence sentence(s) most textually similar to the claim. */
function renderHighlightedText(evidenceText: string, claimText?: string) {
  if (!claimText) return <p className="evidence-text">{evidenceText}</p>;

  const sentences = splitSentences(evidenceText);
  const claimWords = significantWords(claimText);
  if (claimWords.size === 0) return <p className="evidence-text">{evidenceText}</p>;

  let bestIdx = -1;
  let bestScore = 0;
  sentences.forEach((sentence, idx) => {
    const words = significantWords(sentence);
    let overlap = 0;
    words.forEach((w) => {
      if (claimWords.has(w)) overlap += 1;
    });
    if (overlap > bestScore) {
      bestScore = overlap;
      bestIdx = idx;
    }
  });

  if (bestIdx === -1 || bestScore === 0) {
    return <p className="evidence-text">{evidenceText}</p>;
  }

  return (
    <p className="evidence-text">
      {sentences.map((sentence, idx) => (
        <span key={idx} className={idx === bestIdx ? "evidence-highlight" : undefined}>
          {sentence}{" "}
        </span>
      ))}
    </p>
  );
}

export default function EvidenceViewer() {
  const { state, close } = useEvidenceViewer();

  if (!state) return null;

  const { citationId, evidence, claims } = state;
  const item = evidence.find((e) => e.citation_id === citationId);
  const relatedClaims = claims.filter((c) => c.citation_ids.includes(citationId));

  return (
    <div className="evidence-overlay" onClick={close}>
      <div className="evidence-panel" onClick={(e) => e.stopPropagation()}>
        <div className="evidence-panel-header">
          <span className="evidence-citation-tag mono">[{citationId}]</span>
          <button className="btn evidence-close" onClick={close} aria-label="Close">
            Close
          </button>
        </div>

        {!item ? (
          <p className="evidence-empty">
            No evidence found for citation [{citationId}].
          </p>
        ) : (
          <>
            <h3>{item.document_title}</h3>
            <div className="evidence-meta">
              <span>Page {item.page_number}</span>
              <span className="evidence-meta-sep">/</span>
              <span>{item.section || "Unlabeled section"}</span>
              {item.score !== undefined && (
                <>
                  <span className="evidence-meta-sep">/</span>
                  <span className="mono">score {item.score.toFixed(3)}</span>
                </>
              )}
            </div>

            <div className="evidence-block">
              {renderHighlightedText(item.text, relatedClaims[0]?.text)}
            </div>

            {relatedClaims.length > 0 && (
              <div className="evidence-claims">
                <h4>Claims citing this source</h4>
                {relatedClaims.map((claim) => (
                  <div key={claim.claim_id} className="evidence-claim">
                    <div className="evidence-claim-header">
                      <ClaimStatusBadge status={claim.status} grounding={claim.grounding} />
                    </div>
                    <p className="evidence-claim-text">{claim.text}</p>
                    {claim.verifier_rationale && (
                      <p className="evidence-claim-rationale">
                        {claim.verifier_rationale}
                      </p>
                    )}
                  </div>
                ))}
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}
