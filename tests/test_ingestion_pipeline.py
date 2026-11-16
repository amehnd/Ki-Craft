"""Tests for src/ingestion/pipeline.py: chunk exports become stored records."""

import base64
import json

import langsmith
import pytest

pytest.importorskip("langchain_chroma")

from src.ingestion import pipeline
from src.ingestion.pipeline import documents_from_export, run_complete_ingestion_pipeline

JPEG = b"\xff\xd8\xff\xe0fig"


@pytest.fixture(autouse=True)
def _no_tracing():
    """The pipeline functions are traceable; keep them off the network."""
    langsmith.configure(enabled=False)
    yield
    langsmith.configure(enabled=None)


def _export() -> list[dict]:
    """A two-chunk export as export_chunks_to_json writes it."""
    return [
        {
            "chunk_id": 1,
            "enhanced_content": "summary one",
            "metadata": {"original_content": {"raw_text": "text one", "tables_html": [], "images_base64": []}},
        },
        {
            "chunk_id": 2,
            "enhanced_content": "summary two",
            "metadata": {
                "original_content": {
                    "raw_text": "text two",
                    "tables_html": ["<table/>"],
                    "images_base64": [base64.b64encode(JPEG).decode(), base64.b64encode(b"second").decode()],
                }
            },
        },
    ]


def test_records_have_flat_metadata_and_figure_files(tmp_path):
    images_dir = tmp_path / "images"
    docs = documents_from_export(_export(), "ne555", images_dir)

    assert [d.page_content for d in docs] == ["summary one", "summary two"]
    assert set(docs[1].metadata) == {
        "source", "chunk_id", "raw_text", "tables_html", "image_paths", "has_table", "has_image",
    }
    assert docs[0].metadata["image_paths"] == "[]"
    assert not docs[0].metadata["has_image"] and not docs[0].metadata["has_table"]

    paths = json.loads(docs[1].metadata["image_paths"])
    assert paths == [str(images_dir / "ne555_c2_img0.jpg"), str(images_dir / "ne555_c2_img1.jpg")]
    assert (images_dir / "ne555_c2_img0.jpg").read_bytes() == JPEG
    assert json.loads(docs[1].metadata["tables_html"]) == ["<table/>"]
    assert docs[1].metadata["source"] == "ne555" and docs[1].metadata["chunk_id"] == 2


def test_stored_paths_keep_the_images_dir_as_given(tmp_path, monkeypatch):
    """A relative images folder gives repo-relative paths, like the shipped store."""
    monkeypatch.chdir(tmp_path)
    docs = documents_from_export(_export(), "ne555", "dbv2/images")
    assert json.loads(docs[1].metadata["image_paths"])[0] == "dbv2/images/ne555_c2_img0.jpg"


def test_existing_export_stops_before_any_work(tmp_path, monkeypatch):
    (tmp_path / "ne555.json").write_text("[]")

    def forbidden(*args, **kwargs):
        raise AssertionError("no step should run")

    monkeypatch.setattr(pipeline, "partition_document", forbidden)
    with pytest.raises(FileExistsError):
        run_complete_ingestion_pipeline("data/ne555.pdf", export_dir=tmp_path)


def test_pipeline_writes_export_figures_and_records(tmp_path, monkeypatch):
    stored = {}
    monkeypatch.setattr(pipeline, "partition_document", lambda path: ["element"])
    monkeypatch.setattr(pipeline, "create_chunks_by_title", lambda elements: ["chunk"])
    monkeypatch.setattr(pipeline, "summarise_chunks", lambda chunks, backend: ["summarised"])
    monkeypatch.setattr(
        pipeline, "export_chunks_to_json",
        lambda docs, filename: (open(filename, "w").write("[]"), _export())[1],
    )
    monkeypatch.setattr(
        pipeline, "create_vector_store",
        lambda docs, persist_directory: stored.update(docs=docs, dir=persist_directory) or "db",
    )

    db = run_complete_ingestion_pipeline(
        "data/ne555.pdf", persist_directory="store", images_dir=tmp_path / "img", export_dir=tmp_path / "json"
    )
    assert db == "db"
    assert (tmp_path / "json" / "ne555.json").exists()
    assert stored["dir"] == "store"
    assert [d.metadata["source"] for d in stored["docs"]] == ["ne555", "ne555"]
    assert (tmp_path / "img" / "ne555_c2_img0.jpg").read_bytes() == JPEG

