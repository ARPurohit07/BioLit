import { useRef, useState } from "react";
import type { DragEvent } from "react";
import "./DocumentUploader.css";

interface DocumentUploaderProps {
  onUpload: (files: File[]) => void;
  uploading: boolean;
}

export default function DocumentUploader({ onUpload, uploading }: DocumentUploaderProps) {
  const [dragOver, setDragOver] = useState(false);
  const [rejected, setRejected] = useState<string[]>([]);
  const inputRef = useRef<HTMLInputElement>(null);

  const filterPdfs = (files: FileList | File[]): { pdfs: File[]; rejected: string[] } => {
    const pdfs: File[] = [];
    const rejectedNames: string[] = [];
    Array.from(files).forEach((file) => {
      if (file.name.toLowerCase().endsWith(".pdf")) {
        pdfs.push(file);
      } else {
        rejectedNames.push(file.name);
      }
    });
    return { pdfs, rejected: rejectedNames };
  };

  const handleFiles = (files: FileList | File[]) => {
    const { pdfs, rejected: rejectedNames } = filterPdfs(files);
    setRejected(rejectedNames);
    if (pdfs.length > 0) onUpload(pdfs);
  };

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setDragOver(false);
    if (e.dataTransfer.files.length > 0) handleFiles(e.dataTransfer.files);
  };

  return (
    <div className="uploader">
      <div
        className={"uploader-dropzone" + (dragOver ? " uploader-dropzone-active" : "")}
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleDrop}
        onClick={() => inputRef.current?.click()}
      >
        <input
          ref={inputRef}
          type="file"
          accept=".pdf,application/pdf"
          multiple
          hidden
          onChange={(e) => {
            if (e.target.files) handleFiles(e.target.files);
            e.target.value = "";
          }}
        />
        <p className="uploader-title">
          {uploading ? "Uploading…" : "Drag and drop PDF files, or click to browse"}
        </p>
        <p className="uploader-subtitle">PDF only</p>
      </div>
      {rejected.length > 0 && (
        <p className="uploader-rejected">
          Skipped non-PDF file{rejected.length > 1 ? "s" : ""}: {rejected.join(", ")}
        </p>
      )}
    </div>
  );
}
