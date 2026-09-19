import { useCallback, useEffect, useRef, useState } from "react";
import {
  deleteDocument,
  indexDocuments,
  listDocuments,
  uploadDocument,
} from "../api/client";
import type { DocumentMetadata } from "../types/api";
import DocumentUploader from "../components/DocumentUploader";
import DocumentTable from "../components/DocumentTable";
import ErrorBanner from "../components/ErrorBanner";
import LoadingSpinner from "../components/LoadingSpinner";
import "./Documents.css";

const POLL_INTERVAL_MS = 3000;
const IN_PROGRESS_STATUSES = new Set(["parsing", "indexing"]);

export default function Documents() {
  const [documents, setDocuments] = useState<DocumentMetadata[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const [indexing, setIndexing] = useState(false);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const refresh = useCallback(() => {
    return listDocuments()
      .then((res) => {
        setDocuments(res.documents);
        setError(null);
      })
      .catch((err) => setError(err instanceof Error ? err.message : String(err)));
  }, []);

  useEffect(() => {
    refresh().finally(() => setLoading(false));
  }, [refresh]);

  useEffect(() => {
    const anyInProgress = documents.some((d) => IN_PROGRESS_STATUSES.has(d.status));
    if (anyInProgress && !pollRef.current) {
      pollRef.current = setInterval(refresh, POLL_INTERVAL_MS);
    } else if (!anyInProgress && pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
    return () => {
      if (pollRef.current) {
        clearInterval(pollRef.current);
        pollRef.current = null;
      }
    };
  }, [documents, refresh]);

  const handleUpload = async (files: File[]) => {
    setUploading(true);
    setError(null);
    try {
      for (const file of files) {
        await uploadDocument(file);
      }
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setUploading(false);
    }
  };

  const handleDelete = async (documentId: string) => {
    try {
      await deleteDocument(documentId);
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  };

  const handleBuildIndex = async () => {
    const pending = documents
      .filter((d) => d.status === "uploaded")
      .map((d) => d.document_id);
    if (pending.length === 0) return;
    setIndexing(true);
    setError(null);
    try {
      await indexDocuments(pending);
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setIndexing(false);
    }
  };

  const pendingCount = documents.filter((d) => d.status === "uploaded").length;

  return (
    <div className="page">
      <div className="page-header">
        <h1>Document Library</h1>
        <p>Upload PDFs, build the retrieval index, and track document status.</p>
      </div>

      {error && <ErrorBanner message={error} onRetry={refresh} />}

      <div className="card documents-upload-card">
        <DocumentUploader onUpload={handleUpload} uploading={uploading} />
      </div>

      <div className="documents-toolbar">
        <button
          className="btn btn-primary"
          onClick={handleBuildIndex}
          disabled={pendingCount === 0 || indexing}
        >
          {indexing ? "Building index…" : `Build Index (${pendingCount} pending)`}
        </button>
        {indexing && <LoadingSpinner label="Indexing" />}
      </div>

      <div className="card">
        {loading ? (
          <LoadingSpinner label="Loading documents" />
        ) : (
          <DocumentTable documents={documents} onDelete={handleDelete} />
        )}
      </div>
    </div>
  );
}
