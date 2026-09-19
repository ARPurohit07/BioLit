import type {
  CompareRequest,
  DocumentIndexResponse,
  DocumentListResponse,
  DocumentUploadResponse,
  EvaluationSummary,
  HealthResponse,
  LiteratureReviewRequest,
  QueryRequest,
  QueryResponse,
  VerifyRequest,
  VerifyResponse,
} from "../types/api";

export const API_BASE_URL: string =
  import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

export class ApiError extends Error {
  status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function extractErrorMessage(res: Response): Promise<string> {
  try {
    const body = await res.json();
    if (typeof body?.detail === "string") return body.detail;
    if (Array.isArray(body?.detail)) {
      return body.detail
        .map((d: { msg?: string }) => d.msg ?? JSON.stringify(d))
        .join("; ");
    }
    if (typeof body?.message === "string") return body.message;
    return JSON.stringify(body);
  } catch {
    return res.statusText || `Request failed with status ${res.status}`;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE_URL}${path}`, {
      ...init,
      headers: {
        ...(init?.body && !(init.body instanceof FormData)
          ? { "Content-Type": "application/json" }
          : {}),
        ...init?.headers,
      },
    });
  } catch (err) {
    throw new ApiError(
      `Cannot reach backend at ${API_BASE_URL}. Is it running? (${
        err instanceof Error ? err.message : String(err)
      })`,
      0,
    );
  }

  if (!res.ok) {
    const message = await extractErrorMessage(res);
    throw new ApiError(message, res.status);
  }

  if (res.status === 204) return undefined as T;

  return (await res.json()) as T;
}

function postJson<T>(path: string, body: unknown): Promise<T> {
  return request<T>(path, { method: "POST", body: JSON.stringify(body) });
}

// ---------------------------------------------------------------------------
// Health
// ---------------------------------------------------------------------------

export function getHealth(): Promise<HealthResponse> {
  return request<HealthResponse>("/api/health");
}

// ---------------------------------------------------------------------------
// Documents
// ---------------------------------------------------------------------------

export function listDocuments(): Promise<DocumentListResponse> {
  return request<DocumentListResponse>("/api/documents");
}

export function uploadDocument(file: File): Promise<DocumentUploadResponse> {
  const form = new FormData();
  form.append("file", file);
  return request<DocumentUploadResponse>("/api/documents/upload", {
    method: "POST",
    body: form,
  });
}

export function deleteDocument(documentId: string): Promise<{ message: string }> {
  return request<{ message: string }>(
    `/api/documents/${encodeURIComponent(documentId)}`,
    { method: "DELETE" },
  );
}

export function indexDocuments(
  documentIds?: string[],
): Promise<DocumentIndexResponse> {
  return postJson<DocumentIndexResponse>("/api/documents/index", {
    document_ids: documentIds,
  });
}

// ---------------------------------------------------------------------------
// Query / compare / literature review / verify
// ---------------------------------------------------------------------------

export function runQuery(req: QueryRequest): Promise<QueryResponse> {
  return postJson<QueryResponse>("/api/query", req);
}

export function runCompare(req: CompareRequest): Promise<QueryResponse> {
  return postJson<QueryResponse>("/api/compare", req);
}

export function runLiteratureReview(
  req: LiteratureReviewRequest,
): Promise<QueryResponse> {
  return postJson<QueryResponse>("/api/literature-review", req);
}

export function verifyClaim(req: VerifyRequest): Promise<VerifyResponse> {
  return postJson<VerifyResponse>("/api/verify", req);
}

// ---------------------------------------------------------------------------
// Evaluation
// ---------------------------------------------------------------------------

export function getEvaluation(): Promise<EvaluationSummary> {
  return request<EvaluationSummary>("/api/evaluation");
}
