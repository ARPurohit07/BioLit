import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { API_BASE_URL, getHealth } from "../api/client";
import type { HealthResponse } from "../types/api";
import LoadingSpinner from "../components/LoadingSpinner";
import "./Dashboard.css";

const QUICK_LINKS = [
  { to: "/documents", label: "Document Library", description: "Upload and index PDFs" },
  { to: "/query", label: "Research Query", description: "Ask a question over your corpus" },
  { to: "/literature-review", label: "Literature Review", description: "Generate a topic review" },
  { to: "/compare", label: "Paper Comparison", description: "Compare methodology or results" },
  { to: "/evaluation", label: "Evaluation", description: "Retrieval and generation metrics" },
];

export default function Dashboard() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const loadHealth = () => {
    setLoading(true);
    setError(null);
    getHealth()
      .then(setHealth)
      .catch((err) => setError(err instanceof Error ? err.message : String(err)))
      .finally(() => setLoading(false));
  };

  useEffect(loadHealth, []);

  return (
    <div className="page">
      <div className="page-header">
        <h1>Dashboard</h1>
        <p>System status and quick access to BioLit's research tools.</p>
      </div>

      {error && (
        <div className="dashboard-banner">
          <strong>Backend not reachable at {API_BASE_URL}.</strong>
          <p>{error}</p>
          <button className="btn" onClick={loadHealth}>
            Retry
          </button>
        </div>
      )}

      <div className="card dashboard-health-card">
        <h3>System health</h3>
        {loading ? (
          <LoadingSpinner label="Checking backend" />
        ) : health ? (
          <div className="dashboard-health-grid">
            <div className="stat-chip">
              <span className="stat-label">Backend</span>
              <span className="stat-value">{health.status}</span>
            </div>
            <div className="stat-chip">
              <span className="stat-label">Ollama</span>
              <span
                className={
                  "stat-value " +
                  (health.ollama_available ? "dashboard-ok" : "dashboard-down")
                }
              >
                {health.ollama_available ? "Up" : "Down"}
              </span>
            </div>
            <div className="stat-chip">
              <span className="stat-label">Model</span>
              <span className="stat-value dashboard-model">
                {health.ollama_model ?? "—"}
              </span>
            </div>
            <div className="stat-chip">
              <span className="stat-label">Indexed docs</span>
              <span className="stat-value">{health.num_indexed_documents}</span>
            </div>
            <div className="stat-chip">
              <span className="stat-label">Indexed chunks</span>
              <span className="stat-value">{health.num_indexed_chunks}</span>
            </div>
          </div>
        ) : (
          <p>No health data available.</p>
        )}
      </div>

      <h3 className="dashboard-links-heading">Quick links</h3>
      <div className="dashboard-links-grid">
        {QUICK_LINKS.map((link) => (
          <Link key={link.to} to={link.to} className="card dashboard-link-card">
            <span className="dashboard-link-label">{link.label}</span>
            <span className="dashboard-link-description">{link.description}</span>
          </Link>
        ))}
      </div>
    </div>
  );
}
