import type { RAGMode } from "../types/api";
import "./ModeSelector.css";

const MODES: { value: RAGMode; label: string; description: string }[] = [
  {
    value: "fast",
    label: "Fast",
    description: "Dense retrieval only, no reranking or verification. Lowest latency.",
  },
  {
    value: "balanced",
    label: "Balanced",
    description: "Hybrid BM25 + dense retrieval with reranking, no verification.",
  },
  {
    value: "high_faithfulness",
    label: "High Faithfulness",
    description: "Adds query decomposition and claim verification. Slowest, most reliable.",
  },
];

interface ModeSelectorProps {
  value: RAGMode;
  onChange: (mode: RAGMode) => void;
}

export default function ModeSelector({ value, onChange }: ModeSelectorProps) {
  return (
    <div className="mode-selector" role="radiogroup" aria-label="RAG mode">
      {MODES.map((mode) => (
        <label
          key={mode.value}
          className={"mode-option" + (value === mode.value ? " mode-option-active" : "")}
        >
          <input
            type="radio"
            name="rag-mode"
            value={mode.value}
            checked={value === mode.value}
            onChange={() => onChange(mode.value)}
          />
          <span className="mode-option-body">
            <span className="mode-option-label">{mode.label}</span>
            <span className="mode-option-description">{mode.description}</span>
          </span>
        </label>
      ))}
    </div>
  );
}
