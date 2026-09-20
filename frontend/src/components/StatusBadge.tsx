import type { ClaimStatus, DocumentStatus } from "../types/api";
import { GROUNDING_FLAG_HINT } from "../utils/format";
import "./StatusBadge.css";

const DOCUMENT_STATUS_LABEL: Record<DocumentStatus, string> = {
  uploaded: "Uploaded",
  parsing: "Parsing",
  indexing: "Indexing",
  indexed: "Indexed",
  failed: "Failed",
  scanned_needs_ocr: "OCR required",
};

const DOCUMENT_STATUS_CLASS: Record<DocumentStatus, string> = {
  uploaded: "badge-gray",
  parsing: "badge-amber badge-pulse",
  indexing: "badge-amber badge-pulse",
  indexed: "badge-green",
  failed: "badge-red",
  scanned_needs_ocr: "badge-orange",
};

const CLAIM_STATUS_LABEL: Record<ClaimStatus, string> = {
  NOT_VERIFIED: "Not verified",
  SUPPORTED: "Supported",
  PARTIALLY_SUPPORTED: "Partially supported",
  UNSUPPORTED: "Unsupported",
  CONTRADICTED: "Contradicted",
};

const CLAIM_STATUS_CLASS: Record<ClaimStatus, string> = {
  NOT_VERIFIED: "badge-gray",
  SUPPORTED: "badge-green",
  PARTIALLY_SUPPORTED: "badge-yellow",
  UNSUPPORTED: "badge-orange",
  CONTRADICTED: "badge-red",
};

export function DocumentStatusBadge({ status }: { status: DocumentStatus }) {
  const title =
    status === "scanned_needs_ocr"
      ? "This document appears to be a scanned image and needs OCR before it can be indexed."
      : undefined;
  return (
    <span className={`badge ${DOCUMENT_STATUS_CLASS[status]}`} title={title}>
      {DOCUMENT_STATUS_LABEL[status]}
    </span>
  );
}

export function ClaimStatusBadge({
  status,
  grounding,
}: {
  status: ClaimStatus;
  grounding?: "strong" | "weak" | "none" | null;
}) {
  if (status === "NOT_VERIFIED" && grounding === "none") {
    return (
      <span className="badge badge-orange badge-estimate" title={GROUNDING_FLAG_HINT}>
        Check source
      </span>
    );
  }
  const title =
    status === "NOT_VERIFIED"
      ? "This claim was not checked against the evidence. Fast and Balanced modes do not verify; use High-Faithfulness mode."
      : undefined;
  return (
    <span className={`badge ${CLAIM_STATUS_CLASS[status]}`} title={title}>
      {CLAIM_STATUS_LABEL[status]}
    </span>
  );
}
