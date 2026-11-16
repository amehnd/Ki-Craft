"""Tests for src/documents.py: reading a stored chunk's metadata."""

import base64
import json

import pytest

langchain_core = pytest.importorskip("langchain_core")

from langchain_core.documents import Document

import src.documents as documents
from src.documents import chunk_text_and_tables, extract_original_data, stored_image_count


def _doc(metadata):
    return Document(page_content="summary", metadata=metadata)


def test_text_and_tables_are_read_from_metadata():
    doc = _doc({"raw_text": "flat", "tables_html": '["<t/>"]', "image_paths": "[]"})
    assert chunk_text_and_tables(doc) == {"raw_text": "flat", "tables_html": ["<t/>"]}


def test_metadata_without_optional_keys():
    out = extract_original_data(_doc({}))
    assert out == {"raw_text": "", "tables_html": [], "images_base64": []}


def test_figure_count_reads_no_file():
    doc = _doc({"image_paths": json.dumps(["dbv2/images/missing_a.jpg", "dbv2/images/missing_b.jpg"])})
    assert stored_image_count(doc) == 2


def test_image_paths_read_from_disk(tmp_path, monkeypatch):
    (tmp_path / "fig.jpg").write_bytes(b"jpeg-bytes")
    monkeypatch.setattr(documents, "_PROJECT_ROOT", tmp_path)
    out = extract_original_data(_doc({"raw_text": "", "image_paths": ["fig.jpg"]}))
    assert out["images_base64"] == [base64.b64encode(b"jpeg-bytes").decode()]


def test_missing_image_path_is_tolerated():
    doc = _doc({"raw_text": "", "image_paths": ["does_not_exist.jpg"]})
    assert extract_original_data(doc)["images_base64"] == [""]
