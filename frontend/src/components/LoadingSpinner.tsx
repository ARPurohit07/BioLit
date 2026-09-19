import "./LoadingSpinner.css";

interface LoadingSpinnerProps {
  label?: string;
  elapsedSeconds?: number;
}

export default function LoadingSpinner({ label, elapsedSeconds }: LoadingSpinnerProps) {
  return (
    <div className="loading-spinner-wrap">
      <span className="loading-spinner" aria-hidden="true" />
      <span className="loading-spinner-label">
        {label ?? "Working"}
        {elapsedSeconds !== undefined && (
          <span className="mono loading-spinner-elapsed"> {elapsedSeconds.toFixed(0)}s</span>
        )}
      </span>
    </div>
  );
}
