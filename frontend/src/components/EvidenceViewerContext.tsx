import { createContext, useCallback, useContext, useMemo, useState } from "react";
import type { ReactNode } from "react";
import type { Claim, EvidenceItem } from "../types/api";

export interface EvidenceViewerState {
  citationId: number;
  evidence: EvidenceItem[];
  claims: Claim[];
}

interface EvidenceViewerContextValue {
  state: EvidenceViewerState | null;
  openCitation: (
    citationId: number,
    evidence: EvidenceItem[],
    claims: Claim[],
  ) => void;
  close: () => void;
}

const EvidenceViewerContext = createContext<EvidenceViewerContextValue | null>(
  null,
);

export function EvidenceViewerProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<EvidenceViewerState | null>(null);

  const openCitation = useCallback(
    (citationId: number, evidence: EvidenceItem[], claims: Claim[]) => {
      setState({ citationId, evidence, claims });
    },
    [],
  );

  const close = useCallback(() => setState(null), []);

  const value = useMemo(
    () => ({ state, openCitation, close }),
    [state, openCitation, close],
  );

  return (
    <EvidenceViewerContext.Provider value={value}>
      {children}
    </EvidenceViewerContext.Provider>
  );
}

export function useEvidenceViewer(): EvidenceViewerContextValue {
  const ctx = useContext(EvidenceViewerContext);
  if (!ctx) {
    throw new Error(
      "useEvidenceViewer must be used within an EvidenceViewerProvider",
    );
  }
  return ctx;
}
