"""Tests for src/ingestion/extract.py, with partition_pdf mocked."""

import src.ingestion.extract as extract


def test_figures_are_returned_on_the_elements(monkeypatch):
    """Figures must come back as image_base64 on each element, not as files."""
    captured = {}
    monkeypatch.setattr(extract, "partition_pdf", lambda **kwargs: captured.update(kwargs) or [])
    extract.partition_document("data/x.pdf")
    assert captured["extract_image_block_types"] == ["Image"]
    assert captured["extract_image_block_to_payload"] is True

