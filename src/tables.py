"""Turn an HTML table into a Markdown table for a model prompt.

Unstructured extracts tables only as HTML (``metadata.text_as_html``), and
that HTML is what the store keeps: it is lossless and the dashboard renders
it. Prompts use Markdown instead, which takes about 30 percent fewer tokens
on our tables. Markdown cannot merge cells, so a value that spans several
rows or columns (``rowspan``, ``colspan``) is repeated in every cell it
covers; each row then reads on its own. As in a browser, a ``rowspan`` stops
at the end of its section (``thead``, ``tbody`` or ``tfoot``), so a malformed
header span never spills into the body. A table with no cells becomes an
empty string.

Used by the summary prompt (``src/ingestion/enrich.py``) and the answer
prompt (``src/retrieval/generate.py``).
"""

from __future__ import annotations

from html.parser import HTMLParser


class _TableCells(HTMLParser):
    """Collect each row's cells as ``(text, rowspan, colspan)``, and each row's section."""

    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[tuple[str, int, int]]] = []
        self.sections: list[int] = []
        self._section = 0
        self._cell: list[str] | None = None
        self._span = (1, 1)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in ("thead", "tbody", "tfoot"):
            self._section += 1
        elif tag == "tr":
            self.rows.append([])
            self.sections.append(self._section)
        elif tag in ("td", "th"):
            values = dict(attrs)
            self._cell = []
            self._span = (_span(values.get("rowspan")), _span(values.get("colspan")))

    def handle_endtag(self, tag: str) -> None:
        if tag in ("td", "th") and self._cell is not None:
            if not self.rows:
                self.rows.append([])
                self.sections.append(self._section)
            self.rows[-1].append((" ".join("".join(self._cell).split()), *self._span))
            self._cell = None

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)


def _span(value: str | None) -> int:
    """A rowspan or colspan attribute as a count of at least 1."""
    try:
        return max(1, int(value or 1))
    except ValueError:
        return 1


def table_to_markdown(html: str) -> str:
    """Convert one HTML table to a Markdown table.

    The first row becomes the header. Merged cells are repeated in every
    cell they cover, and ``|`` inside a cell is escaped.

    Args:
        html: A ``<table>`` element as Unstructured writes it.

    Returns:
        The Markdown table; an empty string when the table has no cells; or
        ``html`` unchanged when it cannot be parsed, so a prompt never loses
        a table.
    """
    try:
        parser = _TableCells()
        parser.feed(html)
        parser.close()
    except Exception:
        return html

    grid: dict[tuple[int, int], str] = {}
    for row_index, row in enumerate(parser.rows):
        column = 0
        for text, rowspan, colspan in row:
            while (row_index, column) in grid:
                column += 1
            for dr in range(rowspan):
                covered = row_index + dr
                if covered >= len(parser.rows) or parser.sections[covered] != parser.sections[row_index]:
                    break
                for dc in range(colspan):
                    grid[(covered, column + dc)] = text
            column += colspan
    if not grid:
        return ""

    rows = max(r for r, _ in grid) + 1
    columns = max(c for _, c in grid) + 1
    lines = [
        "| " + " | ".join(grid.get((r, c), "").replace("|", "\\|") for c in range(columns)) + " |"
        for r in range(rows)
    ]
    return "\n".join([lines[0], "|" + "---|" * columns, *lines[1:]])
