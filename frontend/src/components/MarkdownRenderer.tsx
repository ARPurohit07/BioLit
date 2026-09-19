import type { ReactNode } from "react";
import { Fragment } from "react";
import type { Claim, EvidenceItem } from "../types/api";
import { useEvidenceViewer } from "./EvidenceViewerContext";
import "./MarkdownRenderer.css";

interface MarkdownRendererProps {
  markdown: string;
  evidence?: EvidenceItem[];
  claims?: Claim[];
}

/** Tokenizes a line of text for **bold** spans and [n] citation markers. */
function renderInline(
  text: string,
  onCitationClick: (id: number) => void,
): ReactNode[] {
  const nodes: ReactNode[] = [];
  const pattern = /(\*\*[^*]+\*\*)|(\[(\d+)\])/g;
  let lastIndex = 0;
  let match: RegExpExecArray | null;
  let key = 0;

  while ((match = pattern.exec(text)) !== null) {
    if (match.index > lastIndex) {
      nodes.push(
        <Fragment key={key++}>{text.slice(lastIndex, match.index)}</Fragment>,
      );
    }
    if (match[1]) {
      nodes.push(<strong key={key++}>{match[1].slice(2, -2)}</strong>);
    } else if (match[3]) {
      const citationId = Number(match[3]);
      nodes.push(
        <button
          key={key++}
          type="button"
          className="citation-marker mono"
          onClick={() => onCitationClick(citationId)}
        >
          [{citationId}]
        </button>,
      );
    }
    lastIndex = pattern.lastIndex;
  }

  if (lastIndex < text.length) {
    nodes.push(<Fragment key={key++}>{text.slice(lastIndex)}</Fragment>);
  }

  return nodes;
}

export default function MarkdownRenderer({
  markdown,
  evidence = [],
  claims = [],
}: MarkdownRendererProps) {
  const { openCitation } = useEvidenceViewer();
  const handleCitationClick = (id: number) => openCitation(id, evidence, claims);

  const lines = markdown.replace(/\r\n/g, "\n").split("\n");
  const blocks: ReactNode[] = [];
  let listBuffer: { ordered: boolean; items: string[] } | null = null;
  let paragraphBuffer: string[] = [];
  let blockKey = 0;

  const flushList = () => {
    if (!listBuffer) return;
    const Tag = listBuffer.ordered ? "ol" : "ul";
    blocks.push(
      <Tag key={blockKey++}>
        {listBuffer.items.map((item, idx) => (
          <li key={idx}>{renderInline(item, handleCitationClick)}</li>
        ))}
      </Tag>,
    );
    listBuffer = null;
  };

  const flushParagraph = () => {
    if (paragraphBuffer.length === 0) return;
    const text = paragraphBuffer.join(" ");
    blocks.push(<p key={blockKey++}>{renderInline(text, handleCitationClick)}</p>);
    paragraphBuffer = [];
  };

  for (const rawLine of lines) {
    const line = rawLine.trimEnd();

    if (line.trim() === "") {
      flushParagraph();
      flushList();
      continue;
    }

    const headingMatch = /^(#{1,4})\s+(.*)$/.exec(line);
    if (headingMatch) {
      flushParagraph();
      flushList();
      const level = headingMatch[1].length;
      const HeadingTag = (`h${Math.min(level + 1, 6)}`) as keyof JSX.IntrinsicElements;
      blocks.push(
        <HeadingTag key={blockKey++}>
          {renderInline(headingMatch[2], handleCitationClick)}
        </HeadingTag>,
      );
      continue;
    }

    const unorderedMatch = /^[-*]\s+(.*)$/.exec(line);
    const orderedMatch = /^\d+\.\s+(.*)$/.exec(line);
    if (unorderedMatch || orderedMatch) {
      flushParagraph();
      const ordered = !!orderedMatch;
      const content = (unorderedMatch ?? orderedMatch)![1];
      if (!listBuffer || listBuffer.ordered !== ordered) {
        flushList();
        listBuffer = { ordered, items: [] };
      }
      listBuffer.items.push(content);
      continue;
    }

    flushList();
    paragraphBuffer.push(line.trim());
  }

  flushParagraph();
  flushList();

  return <div className="markdown-body">{blocks}</div>;
}
