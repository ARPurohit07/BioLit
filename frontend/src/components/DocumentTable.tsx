import { useState } from "react";
import type { DocumentMetadata } from "../types/api";
import { DocumentStatusBadge } from "./StatusBadge";
import "./DocumentTable.css";

interface DocumentTableProps {
  documents: DocumentMetadata[];
  onDelete: (documentId: string) => void;
}

export default function DocumentTable({ documents, onDelete }: DocumentTableProps) {
  const [viewing, setViewing] = useState<DocumentMetadata | null>(null);
  const [confirmingDelete, setConfirmingDelete] = useState<string | null>(null);

  if (documents.length === 0) {
    return <p className="doc-table-empty">No documents uploaded yet.</p>;
  }

  return (
    <>
      <table className="doc-table">
        <thead>
          <tr>
            <th>Title</th>
            <th>Authors</th>
            <th>Year</th>
            <th>Pages</th>
            <th>Chunks</th>
            <th>Status</th>
            <th>Indexed?</th>
            <th></th>
          </tr>
        </thead>
        <tbody>
          {documents.map((doc) => (
            <tr key={doc.document_id}>
              <td className="doc-table-title" title={doc.filename}>
                {doc.title || doc.filename}
              </td>
              <td>{doc.authors.length > 0 ? doc.authors.join(", ") : "—"}</td>
              <td>{doc.year ?? "—"}</td>
              <td>{doc.num_pages}</td>
              <td>{doc.num_chunks}</td>
              <td>
                <DocumentStatusBadge status={doc.status} />
              </td>
              <td>{doc.status === "indexed" ? "Yes" : "No"}</td>
              <td className="doc-table-actions">
                <button className="btn doc-table-btn" onClick={() => setViewing(doc)}>
                  View
                </button>
                {confirmingDelete === doc.document_id ? (
                  <>
                    <button
                      className="btn btn-danger doc-table-btn"
                      onClick={() => {
                        onDelete(doc.document_id);
                        setConfirmingDelete(null);
                      }}
                    >
                      Confirm
                    </button>
                    <button
                      className="btn doc-table-btn"
                      onClick={() => setConfirmingDelete(null)}
                    >
                      Cancel
                    </button>
                  </>
                ) : (
                  <button
                    className="btn doc-table-btn"
                    onClick={() => setConfirmingDelete(doc.document_id)}
                  >
                    Delete
                  </button>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>

      {viewing && (
        <div className="doc-modal-overlay" onClick={() => setViewing(null)}>
          <div className="doc-modal card" onClick={(e) => e.stopPropagation()}>
            <div className="doc-modal-header">
              <h3>{viewing.title || viewing.filename}</h3>
              <button className="btn" onClick={() => setViewing(null)}>
                Close
              </button>
            </div>
            <dl className="doc-modal-fields">
              <dt>Document ID</dt>
              <dd className="mono">{viewing.document_id}</dd>
              <dt>Filename</dt>
              <dd>{viewing.filename}</dd>
              <dt>Authors</dt>
              <dd>{viewing.authors.length > 0 ? viewing.authors.join(", ") : "—"}</dd>
              <dt>Year</dt>
              <dd>{viewing.year ?? "—"}</dd>
              <dt>Source</dt>
              <dd>{viewing.source}</dd>
              <dt>Pages</dt>
              <dd>{viewing.num_pages}</dd>
              <dt>Chunks</dt>
              <dd>{viewing.num_chunks}</dd>
              <dt>Status</dt>
              <dd>
                <DocumentStatusBadge status={viewing.status} />
              </dd>
              <dt>Uploaded at</dt>
              <dd className="mono">{viewing.uploaded_at}</dd>
              {viewing.error_message && (
                <>
                  <dt>Error</dt>
                  <dd className="doc-modal-error">{viewing.error_message}</dd>
                </>
              )}
            </dl>
          </div>
        </div>
      )}
    </>
  );
}
