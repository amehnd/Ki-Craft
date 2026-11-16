"""Tests for src/tables.py: HTML tables as Markdown for prompts."""

from src.tables import table_to_markdown


def test_simple_table_becomes_markdown_with_a_header_row():
    html = "<table><tr><th>Pin</th><th>Name</th></tr><tr><td>1</td><td>ADJ</td></tr></table>"
    assert table_to_markdown(html) == "| Pin | Name |\n|---|---|\n| 1 | ADJ |"


def test_merged_cells_are_repeated_in_every_cell_they_cover():
    """A rowspan or colspan value fills each covered cell, so every row reads alone."""
    html = (
        "<table>"
        "<tr><th>Parameter</th><th colspan='2'>VREF (V)</th></tr>"
        "<tr><td rowspan='2'>Reference</td><td>MIN</td><td>MAX</td></tr>"
        "<tr><td>1.2</td><td>1.3</td></tr>"
        "</table>"
    )
    assert table_to_markdown(html).splitlines() == [
        "| Parameter | VREF (V) | VREF (V) |",
        "|---|---|---|",
        "| Reference | MIN | MAX |",
        "| Reference | 1.2 | 1.3 |",
    ]


def test_cell_text_is_flattened_and_pipes_escaped():
    html = "<table><tr><td>TO-252 (KVU) | 3</td><td>a\n  &amp; b</td></tr></table>"
    assert table_to_markdown(html) == "| TO-252 (KVU) \\| 3 | a & b |\n|---|---|"


def test_a_table_without_cells_is_empty():
    """Unstructured sometimes writes an empty table; the prompt then skips it."""
    assert table_to_markdown("<table><tbody></tbody></table>") == ""
    assert table_to_markdown("") == ""


def test_a_header_rowspan_stops_at_the_end_of_the_header():
    """As in a browser, a rowspan never crosses from thead into tbody."""
    html = (
        "<table><thead><tr><th rowspan='5'>PARAMETER</th><th>MIN</th></tr></thead>"
        "<tbody><tr><td>VREF</td><td>1.2</td></tr></tbody></table>"
    )
    assert table_to_markdown(html).splitlines() == ["| PARAMETER | MIN |", "|---|---|", "| VREF | 1.2 |"]


def test_short_rows_are_padded():
    html = "<table><tr><td>a</td><td>b</td></tr><tr><td>c</td></tr></table>"
    assert table_to_markdown(html).splitlines()[-1] == "| c |  |"
