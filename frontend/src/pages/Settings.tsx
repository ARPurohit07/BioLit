import { API_BASE_URL } from "../api/client";
import "./Settings.css";

export default function Settings() {
  return (
    <div className="page">
      <div className="page-header">
        <h1>Settings</h1>
        <p>Read-only view of active configuration. Edit config files by hand and restart the backend.</p>
      </div>

      <div className="card settings-section">
        <h3>API connection</h3>
        <dl className="settings-fields">
          <dt>API base URL</dt>
          <dd className="mono">{API_BASE_URL}</dd>
        </dl>
        <p className="settings-hint">
          Set via <span className="mono">VITE_API_BASE_URL</span> in{" "}
          <span className="mono">frontend/.env</span> (see{" "}
          <span className="mono">frontend/.env.example</span>). Rebuild or restart the dev
          server after changing it.
        </p>
      </div>

      <div className="card settings-section">
        <h3>Configuration files</h3>
        <p className="settings-hint">
          BioLit's backend behavior is controlled by hand-edited YAML files, not by this UI.
          Changes require a backend restart to take effect.
        </p>
        <table className="settings-table">
          <thead>
            <tr>
              <th>File</th>
              <th>Controls</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td className="mono">configs/models.yaml</td>
              <td>
                Base model, Ollama host/model name, embedding model, reranker, and verifier
                settings.
              </td>
            </tr>
            <tr>
              <td className="mono">configs/retrieval.yaml</td>
              <td>Chunking, hybrid retrieval (dense/BM25/RRF), reranking, and per-mode behavior.</td>
            </tr>
            <tr>
              <td className="mono">configs/training.yaml</td>
              <td>QLoRA fine-tuning hyperparameters for the local base model.</td>
            </tr>
          </tbody>
        </table>
      </div>

      <div className="card settings-section">
        <h3>Switching models</h3>
        <ol className="settings-steps">
          <li>
            Pull or prepare the desired model with Ollama, e.g.{" "}
            <span className="mono">ollama pull qwen2.5:1.5b-instruct</span>.
          </li>
          <li>
            Update <span className="mono">ollama.model_name</span> /{" "}
            <span className="mono">ollama.base_model_name</span> in{" "}
            <span className="mono">configs/models.yaml</span>.
          </li>
          <li>Restart the backend so it picks up the new configuration.</li>
        </ol>
        <p className="settings-hint">
          This page does not read or write those files — it only documents where they live.
          There is no backend endpoint for live configuration in the current API contract.
        </p>
      </div>
    </div>
  );
}
