"""Structure-aware page parsing: pull tables and figures out of the running text of a PDF page.

The plain-text extractor flattens a results table into a run of numbers and hands it to the sentence chunker, which
produces chunks that are neither readable nor retrievable, and it ignores figures altogether. This module works on
the page layout instead (this is structural, not semantic, chunking):

  * tables  -> a caption plus a Markdown grid (ruled tables via PyMuPDF's line detection; "booktabs"-style tables
               with no vertical rules via a text-alignment pass anchored on the caption);
  * figures -> the caption plus a PNG crop of the graphics above it (embedded images and vector drawings alike);
               the caption text is what gets embedded, since no model here reads pixels;
  * body    -> everything else, with table cells, figure text and captions removed so they are not indexed twice.

Everything degrades to the old behaviour: if a table or figure cannot be located reliably, its text simply stays
in the body.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

try:
    import fitz  # PyMuPDF
except ImportError:  # pragma: no cover
    fitz = None

_CAPTION_RE = re.compile(
    r"^\s*(?P<kind>Table|TABLE|Tab\.|Figure|FIGURE|Fig\.)\s*(?P<num>\d+|[IVX]+)\s*(?:[.:|]\s*\S|\s+[A-Z])"
)
_MIN_FIGURE_AREA = 6000.0      # pt^2; smaller graphics are logos, bullets or rules
_FIGURE_DPI = 110
_MAX_FIGURE_TEXT = 300         # characters of in-figure text (axis labels, legends) kept for retrieval
_PARAGRAPH_MIN_CHARS = 200     # a text block at least this long is body prose, not a caption or table cell


@dataclass
class TableBlock:
    page_number: int
    label: str                 # "Table 2"
    caption: str
    rows: list[list[str]]      # header row first
    bbox: tuple[float, float, float, float]
    detected_by: str           # "lines" | "text"


@dataclass
class FigureBlock:
    page_number: int
    label: str                 # "Figure 3"
    caption: str
    image_path: str | None     # repo-relative PNG crop, None when the caption was found but no graphic
    figure_text: str = ""      # text that sits inside the figure region (axis labels, legends)
    bbox: tuple[float, float, float, float] | None = None


@dataclass
class PageStructure:
    body_text: str
    tables: list[TableBlock] = field(default_factory=list)
    figures: list[FigureBlock] = field(default_factory=list)


_SURROGATES = re.compile("[\ud800-\udfff]")


def sanitize(text: str) -> str:
    """Drop NULs and lone UTF-16 surrogates (PDF math alphabets sometimes arrive split): they cannot be written as UTF-8."""
    return _SURROGATES.sub("", text.replace("\x00", ""))


def _clean_cell(cell) -> str:
    return re.sub(r"\s+", " ", sanitize(cell or "")).strip()


def _clean_rows(raw: list[list]) -> list[list[str]]:
    rows = [[_clean_cell(c) for c in row] for row in raw if row]
    rows = [r for r in rows if any(r)]
    if not rows:
        return []
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    keep = [i for i in range(width) if any(r[i] for r in rows)]   # drop columns that are empty everywhere
    return [[r[i] for i in keep] for r in rows]


def _digit_share(rows: list[list[str]]) -> float:
    cells = [c for r in rows for c in r if c]
    return sum(any(ch.isdigit() for ch in c) for c in cells) / len(cells) if cells else 0.0


def _cells_look_tabular(rows: list[list[str]]) -> bool:
    """Reject grids that are really body prose caught by the detector: table cells are short and mostly filled."""
    cells = [c for r in rows for c in r]
    filled = [c for c in cells if c]
    if not filled or len(rows[0]) > 14:
        return False
    prose = sum(len(c.split()) >= 6 for c in filled) / len(filled)
    empty = 1 - len(filled) / len(cells)
    return prose < 0.2 and empty < 0.6


def _looks_like_table(rows: list[list[str]], captioned: bool) -> bool:
    if len(rows) < 2 or max(len(r) for r in rows) < 2:
        return False
    if not _cells_look_tabular(rows):
        return False
    if captioned:
        return True
    # Ruled-line detection also fires on two-column body text and boxed prose; uncaptioned "tables" must look numeric.
    return len(rows) >= 3 and _digit_share(rows) >= 0.4


def _area(r) -> float:
    return max(0.0, r[2] - r[0]) * max(0.0, r[3] - r[1])


def _inside_fraction(inner, outer) -> float:
    """Share of `inner`'s area that lies within `outer`."""
    x0, y0, x1, y1 = max(inner[0], outer[0]), max(inner[1], outer[1]), min(inner[2], outer[2]), min(inner[3], outer[3])
    return _area((x0, y0, x1, y1)) / _area(inner) if _area(inner) > 0 else 0.0


def _caption_of(block_text: str):
    m = _CAPTION_RE.match(block_text)
    if not m:
        return None
    kind = "Table" if m.group("kind").lower().startswith("tab") else "Figure"
    return kind, f"{kind} {m.group('num')}"


def _find_tables(page, **kw):
    try:
        return page.find_tables(**kw).tables
    except Exception:  # PyMuPDF raises on a few malformed pages; treat as "no tables found"
        return []


class StructureParser:
    def __init__(self, figures_dir: Path, repo_root: Path):
        self.figures_dir = Path(figures_dir)
        self.repo_root = Path(repo_root)

    # ------------------------------------------------------------------ public
    def parse_page(self, page, doc_key: str, page_number: int) -> PageStructure:
        blocks = [(*b[:4], sanitize(b[4]), *b[5:]) for b in page.get_text("blocks") if b[6] == 0 and b[4].strip()]
        captions = []  # (kind, label, block_index)
        for i, b in enumerate(blocks):
            cap = _caption_of(b[4])
            if cap:
                captions.append((cap[0], cap[1], i))

        removed: set[int] = set()
        tables = self._tables(page, page_number, blocks, captions, removed)
        figures = self._figures(page, doc_key, page_number, blocks, captions, removed)

        body = "\n".join(b[4].rstrip() for i, b in enumerate(blocks) if i not in removed)
        return PageStructure(body_text=body.strip(), tables=tables, figures=figures)

    # ------------------------------------------------------------------ tables
    def _tables(self, page, page_number, blocks, captions, removed) -> list[TableBlock]:
        table_caps = [(label, i) for kind, label, i in captions if kind == "Table"]
        found: list[TableBlock] = []
        claimed_caps: set[int] = set()

        for t in _find_tables(page):
            rows = _clean_rows(t.extract())
            cap = self._nearest_caption(t.bbox, blocks, table_caps, claimed_caps)
            if not _looks_like_table(rows, captioned=cap is not None):
                continue
            found.append(self._make_table(page_number, cap, blocks, rows, t.bbox, "lines"))
            if cap:
                claimed_caps.add(cap[1])

        # Captions with no ruled table (booktabs style): read the aligned text below the caption instead.
        for label, i in table_caps:
            if i in claimed_caps:
                continue
            clip = self._region_below(page, blocks, i)
            if clip is None:
                continue
            candidates = [(t, _clean_rows(t.extract())) for t in _find_tables(page, clip=clip, strategy="text")]
            candidates = [(t, r) for t, r in candidates if _looks_like_table(r, captioned=True)]
            if not candidates:
                continue
            t, rows = max(candidates, key=lambda tr: len(tr[1]) * len(tr[1][0]))
            found.append(self._make_table(page_number, (label, i), blocks, rows, t.bbox, "text"))
            claimed_caps.add(i)

        for tb in found:
            for j, b in enumerate(blocks):
                if _inside_fraction(b[:4], tb.bbox) >= 0.5:
                    removed.add(j)
        for _label, i in table_caps:
            if i in claimed_caps:
                removed.add(i)
        return found

    def _make_table(self, page_number, cap, blocks, rows, bbox, how) -> TableBlock:
        label, caption = ("Table", "")
        if cap:
            label, idx = cap
            caption = re.sub(r"\s+", " ", blocks[idx][4]).strip()
        return TableBlock(page_number, label if cap else "Table", caption, rows, tuple(bbox), how)

    @staticmethod
    def _column_span(page, blocks, cap) -> tuple[float, float]:
        """x-range a caption's table/figure can occupy: its own column on a two-column page, else the full width.
        (A caption's own width is a poor guide: a short caption sits above a table that is wider than it.)"""
        width = page.rect.width
        prose = [b for b in blocks if len(b[4]) >= _PARAGRAPH_MIN_CHARS and _caption_of(b[4]) is None]
        narrow = sum((b[2] - b[0]) < 0.6 * width for b in prose)
        two_column = len(prose) >= 3 and narrow >= 0.6 * len(prose)
        centre = ((cap[0] + cap[2]) / 2 - page.rect.x0) / width
        if two_column and centre < 0.42:
            return page.rect.x0 + 20, page.rect.x0 + width / 2 + 6
        if two_column and centre > 0.58:
            return page.rect.x0 + width / 2 - 6, page.rect.x1 - 20
        return page.rect.x0 + 20, page.rect.x1 - 20

    @staticmethod
    def _nearest_caption(bbox, blocks, table_caps, claimed):
        best, best_gap = None, 60.0   # a caption more than ~60pt from the grid is not its caption
        for label, i in table_caps:
            if i in claimed:
                continue
            b = blocks[i]
            gap = bbox[1] - b[3] if b[3] <= bbox[1] + 5 else b[1] - bbox[3] if b[1] >= bbox[3] - 5 else None
            if gap is not None and -5 <= gap < best_gap:
                best, best_gap = (label, i), gap
        return best

    def _region_below(self, page, blocks, cap_idx):
        cap = blocks[cap_idx]
        y_end = page.rect.y1 - 36
        for b in blocks:
            if b[1] > cap[3] + 12 and len(b[4]) >= _PARAGRAPH_MIN_CHARS and _caption_of(b[4]) is None:
                y_end = min(y_end, b[1])
        if y_end - cap[3] < 25:
            return None
        x0, x1 = self._column_span(page, blocks, cap)
        return fitz.Rect(x0, cap[3], x1, y_end)

    # ------------------------------------------------------------------ figures
    def _figures(self, page, doc_key, page_number, blocks, captions, removed) -> list[FigureBlock]:
        figure_caps = [(label, i) for kind, label, i in captions if kind == "Figure"]
        if not figure_caps:
            return []
        graphics = [tuple(im["bbox"]) for im in page.get_image_info()]
        try:
            graphics += [tuple(d["rect"]) for d in page.get_drawings() if _area(tuple(d["rect"])) > 4.0]
        except Exception:
            pass

        figures: list[FigureBlock] = []
        for n, (label, i) in enumerate(figure_caps):
            cap = blocks[i]
            x0, x1 = self._column_span(page, blocks, cap)
            y_top = page.rect.y0 + 20
            for j, b in enumerate(blocks):
                overlaps_x = b[0] < x1 and b[2] > x0
                if j != i and b[3] <= cap[1] and overlaps_x and len(b[4]) >= _PARAGRAPH_MIN_CHARS and _caption_of(b[4]) is None:
                    y_top = max(y_top, b[3])
            region = (x0, y_top, x1, cap[1] + 2)
            inside = [g for g in graphics if _inside_fraction(g, region) >= 0.6 and _area(g) > 1.0]
            if not inside:
                continue
            bbox = (min(g[0] for g in inside), min(g[1] for g in inside), max(g[2] for g in inside), max(g[3] for g in inside))
            if _area(bbox) < _MIN_FIGURE_AREA:
                continue
            in_fig = [j for j, b in enumerate(blocks) if j != i and _inside_fraction(b[:4], bbox) >= 0.6]
            fig_text = re.sub(r"\s+", " ", " ".join(blocks[j][4] for j in in_fig)).strip()[:_MAX_FIGURE_TEXT]
            path = self._render(page, bbox, doc_key, page_number, n)
            removed.update(in_fig)
            removed.add(i)
            figures.append(FigureBlock(
                page_number, label, re.sub(r"\s+", " ", cap[4]).strip(), path, fig_text, tuple(bbox),
            ))
        return figures

    def _render(self, page, bbox, doc_key, page_number, n) -> str | None:
        try:
            self.figures_dir.mkdir(parents=True, exist_ok=True)
            out = self.figures_dir / f"{doc_key}_p{page_number}_f{n}.png"
            clip = fitz.Rect(*bbox) + (-4, -4, 4, 4)
            page.get_pixmap(clip=clip & page.rect, dpi=_FIGURE_DPI).save(str(out))
            try:
                return out.relative_to(self.repo_root).as_posix()
            except ValueError:
                return out.as_posix()
        except Exception:
            return None


def table_to_markdown(rows: list[list[str]]) -> str:
    esc = lambda c: c.replace("|", "\\|")
    head, body = rows[0], rows[1:]
    lines = ["| " + " | ".join(esc(c) for c in head) + " |", "|" + " --- |" * len(head)]
    lines += ["| " + " | ".join(esc(c) for c in r) + " |" for r in body]
    return "\n".join(lines)


def rows_from_markdown(md: str) -> list[list[str]]:
    rows = []
    for line in md.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip().replace(r"\|", "|") for c in line.strip("|").split("|")]
        if all(re.fullmatch(r":?-{2,}:?", c) for c in cells if c):
            continue
        rows.append(cells)
    return _clean_rows(rows)


def apply_transcriptions(tables: list[TableBlock], page_number: int, cache: dict) -> None:
    """Swap in a cached transcription for a table's rows, but only one that passed the number check when it was made."""
    for idx, tb in enumerate(tables):
        entry = cache.get(f"p{page_number}_t{idx}")
        if entry and entry.get("accepted"):
            rows = rows_from_markdown(entry["markdown"])
            if len(rows) >= 2 and max(len(r) for r in rows) >= 2:
                tb.rows = rows
