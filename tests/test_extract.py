"""Tests for src/ingestion/extract.py, with partition_pdf mocked."""

import src.ingestion.extract as extract


def test_figures_are_returned_on_the_elements(monkeypatch):
    """Figures must come back as image_base64 on each element, not as files."""
    captured = {}
    monkeypatch.setattr(extract, "partition_pdf", lambda **kwargs: captured.update(kwargs) or [])
    extract.partition_document("data/x.pdf")
    assert captured["extract_image_block_types"] == ["Image"]
    assert captured["extract_image_block_to_payload"] is True


from unstructured.documents.elements import ElementMetadata, Footer, Header, Image, NarrativeText, Title


def _on(element, page: int):
    element.metadata = ElementMetadata(page_number=page)
    return element


def _page(n: int, logo_text: str = "I} TEXAS INSTRUMENTS www.ti.com") -> list:
    """One datasheet page: running header, logo, content, a real figure, footer."""
    return [
        _on(Header("NE555 SLFS022K"), n),
        _on(Image(logo_text), n),
        _on(NarrativeText(f"content of page {n}"), n),
        _on(Image("Figure supply current versus supply voltage VCC at TA 25C typical"), n),
        _on(Image(""), n),
        _on(Footer("Submit Document Feedback"), n),
    ]


def test_headers_footers_and_repeated_logos_are_dropped():
    elements = _page(1) + _page(2, "TEXAS INSTRUMENTS www.ti.com") + _page(3)
    kept = extract.remove_page_decoration(elements, min_pages=3)
    assert [e.text for e in kept if isinstance(e, NarrativeText)] == [f"content of page {n}" for n in (1, 2, 3)]
    assert not [e for e in kept if isinstance(e, (Header, Footer))]
    images = [e.text for e in kept if isinstance(e, Image)]
    assert not any("TEXAS" in t for t in images), "the logo repeats on 3 pages despite OCR noise"
    assert images.count("") == 3, "images without OCR words are never treated as logos"
    assert images.count("Figure supply current versus supply voltage VCC at TA 25C typical") == 3, (
        "a graph repeated on every page has too many words to be a logo"
    )


def test_an_image_on_fewer_pages_is_kept():
    kept = extract.remove_page_decoration(_page(1) + _page(2), min_pages=3)
    assert sum("TEXAS" in e.text for e in kept if isinstance(e, Image)) == 2


def test_document_context_uses_the_running_header():
    elements = [_on(Title("1 Features"), 1)] + _page(1) + _page(2)
    assert extract.document_context(elements, "ne555") == "ne555: NE555 SLFS022K"
    assert extract.document_context([_on(Title("x"), 1)], "ne555") == "ne555"
