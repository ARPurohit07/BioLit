"""Structure-aware chunking: tables and figures become their own chunks and leave the body text."""
from __future__ import annotations

import pytest

fitz = pytest.importorskip("fitz")

from backend.app.ingestion.chunker import PageAwareChunker  # noqa: E402
from backend.app.ingestion.pdf_loader import PDFLoader  # noqa: E402
from backend.app.ingestion.section_detector import SectionDetector  # noqa: E402
from backend.app.ingestion.structure import table_to_markdown  # noqa: E402

PROSE = (
    "We evaluate the proposed drug response model on three public cell line collections and compare it against "
    "several published baselines. The model reaches consistently higher correlation on unseen drugs, and the "
    "improvement holds across all three collections that we tried in the inductive setting."
)


def _make_pdf(path):
    doc = fitz.open()
    page = doc.new_page()  # 595 x 842
    page.insert_textbox(fitz.Rect(72, 60, 520, 140), PROSE, fontsize=10)
    # a ruled 3x3 table with a caption above it
    page.insert_text((72, 175), "Table 1: Pearson correlation on the benchmark datasets.", fontsize=9)
    xs, ys = [72, 172, 272, 372], [185, 205, 225, 245]
    for x in xs:
        page.draw_line((x, ys[0]), (x, ys[-1]))
    for y in ys:
        page.draw_line((xs[0], y), (xs[-1], y))
    cells = [["Method", "GDSC", "CCLE"], ["Baseline", "0.412", "0.398"], ["Proposed", "0.567", "0.541"]]
    for r, row in enumerate(cells):
        for c, text in enumerate(row):
            page.insert_text((xs[c] + 5, ys[r] + 14), text, fontsize=9)
    page.insert_textbox(fitz.Rect(72, 270, 520, 350), PROSE, fontsize=10)
    # a vector figure with a caption below it
    page.draw_rect(fitz.Rect(100, 380, 480, 560), fill=(0.8, 0.85, 1), color=(0, 0, 0))
    page.draw_circle((290, 470), 40, fill=(1, 0.6, 0.6))
    page.insert_text((72, 585), "Figure 1: Overview of the proposed architecture.", fontsize=9)
    page.insert_textbox(fitz.Rect(72, 610, 520, 700), PROSE, fontsize=10)
    doc.save(str(path))
    doc.close()


@pytest.fixture()
def parsed(tmp_path):
    pdf = tmp_path / "synthetic.pdf"
    _make_pdf(pdf)
    pages = PDFLoader().load_structured(str(pdf), "doc1", tmp_path / "figs", tmp_path)
    chunks = PageAwareChunker(400, 60, 10).chunk_document("doc1", pages, SectionDetector())
    return pages, chunks, tmp_path


def test_a_ruled_table_becomes_a_captioned_markdown_chunk(parsed):
    _pages, chunks, _ = parsed
    tables = [c for c in chunks if c.chunk_type == "table"]
    assert len(tables) == 1
    t = tables[0]
    assert t.label == "Table 1"
    assert t.text.startswith("Table 1: Pearson correlation")
    assert "| Method | GDSC | CCLE |" in t.text and "| Proposed | 0.567 | 0.541 |" in t.text


def test_table_cells_and_caption_are_removed_from_the_body_text(parsed):
    _pages, chunks, _ = parsed
    body = " ".join(c.text for c in chunks if c.chunk_type == "text")
    assert "0.567" not in body and "Pearson correlation on the benchmark" not in body
    assert "drug response model" in body            # ordinary prose is untouched


def test_a_vector_figure_becomes_a_caption_chunk_with_an_image_crop(parsed):
    pages, chunks, tmp = parsed
    figures = [c for c in chunks if c.chunk_type == "figure"]
    assert len(figures) == 1 and figures[0].label == "Figure 1"
    assert "Overview of the proposed architecture" in figures[0].text
    assert figures[0].image_path and (tmp / figures[0].image_path).exists()
    body = " ".join(c.text for c in chunks if c.chunk_type == "text")
    assert "Overview of the proposed architecture" not in body


def test_text_chunks_keep_their_default_type(parsed):
    _pages, chunks, _ = parsed
    assert {c.chunk_type for c in chunks} == {"text", "table", "figure"}
    assert all(c.label is None and c.image_path is None for c in chunks if c.chunk_type == "text")


def test_a_long_table_is_split_by_rows_with_caption_and_header_repeated():
    from backend.app.ingestion.pdf_loader import PageText
    from backend.app.ingestion.structure import TableBlock

    rows = [["Drug", "AUC"]] + [[f"compound number {i} with a long descriptive name", f"0.{i:03d}"] for i in range(80)]
    page = PageText(1, "", tables=[TableBlock(1, "Table 9", "Table 9: Many rows.", rows, (0, 0, 1, 1), "lines")])
    chunks = PageAwareChunker(120, 20, 10).chunk_document("d", [page], SectionDetector())
    parts = [c for c in chunks if c.chunk_type == "table"]
    assert len(parts) > 1
    assert all(p.text.startswith("Table 9: Many rows.\n| Drug | AUC |") for p in parts)
    assert sum(p.text.count("compound number") for p in parts) == 80        # no row lost or duplicated


def test_markdown_escapes_pipes_in_cells():
    assert "a\\|b" in table_to_markdown([["h"], ["a|b"]])


def test_a_plain_page_is_unchanged(tmp_path):
    doc = fitz.open()
    doc.new_page().insert_textbox(fitz.Rect(72, 72, 520, 200), PROSE, fontsize=10)
    doc.save(str(tmp_path / "plain.pdf"))
    pages = PDFLoader().load_structured(str(tmp_path / "plain.pdf"), "d", tmp_path / "f", tmp_path)
    assert not pages[0].tables and not pages[0].figures and "drug response model" in pages[0].text
