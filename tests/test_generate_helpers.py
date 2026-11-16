"""Tests for pure helper functions in src/retrieval/generate.py.

Cloud/model dependencies are optional; the module is skipped entirely when
they are not installed so the suite stays lightweight.
"""

from pathlib import Path

import pytest

pytest.importorskip("langsmith")
pytest.importorskip("langchain_core")

from langchain_core.documents import Document

from src.retrieval.generate import (
    PromptFigure,
    _build_message_content,
    _build_text_prompt,
    _image_blocks,
    _capture_usage,
    _estimate_cost,
    _extract_text,
    _prompt_stats,
    _trace_inputs,
    select_figures,
)


class _FakeMsg:
    def __init__(self, usage_metadata=None, response_metadata=None):
        self.usage_metadata = usage_metadata
        self.response_metadata = response_metadata
        self.content = ""


def test_capture_usage_from_usage_metadata():
    usage = {}
    _capture_usage(_FakeMsg({"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}), usage)
    assert usage == {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}


def test_capture_usage_from_response_metadata():
    usage = {}
    _capture_usage(
        _FakeMsg(None, {"usage_metadata": {"input_tokens": 2, "output_tokens": 3, "total_tokens": 5}}),
        usage,
    )
    assert usage["input_tokens"] == 2


def test_capture_usage_ignores_non_int():
    usage = {}
    _capture_usage(_FakeMsg({"input_tokens": "nope", "output_tokens": 1, "total_tokens": 10}), usage)
    assert usage == {"output_tokens": 1, "total_tokens": 10}


def test_capture_usage_no_metadata():
    usage = {}
    _capture_usage(_FakeMsg(), usage)
    assert usage == {}


def test_estimate_cost_basic():
    usage = {"input_tokens": 1_000_000, "output_tokens": 0}
    # price config is imported from settings; just assert the shape is a float.
    cost = _estimate_cost(usage)
    assert cost is None or isinstance(cost, float)


def test_extract_text_string():
    assert _extract_text("plain") == "plain"


def test_extract_text_list_of_parts():
    parts = [{"type": "text", "text": "a"}, {"type": "content", "text": "b"}, "c"]
    assert _extract_text(parts) == "abc"


def test_extract_text_other():
    assert _extract_text(None) == "None"


def test_prompt_stats_counts_text_and_images():
    content = [
        {"type": "text", "text": "hello"},
        {"type": "image_url", "image_url": {"url": "x"}},
        {"type": "image_url", "image_url": {"url": "y"}},
    ]
    chars, images = _prompt_stats(content)
    assert chars == 5
    assert images == 2


def test_trace_inputs_keep_the_question_and_drop_objects():
    """Only plain values reach LangSmith; the chat model (and its key) never does."""
    recorded = _trace_inputs(
        {
            "retriever": object(),
            "query": "LM555 pins?",
            "llm": object(),
            "model_id": "gemini-3.5-flash-lite",
            "chunks": ["a", "b"],
            "max_images": 0,
            "send_images": True,
        }
    )
    assert recorded == {"question": "LM555 pins?", "max_images": 0, "send_images": True, "num_chunks": 2}


def test_generation_trace_records_figure_labels_not_bytes():
    figures = [PromptFigure("F1", 1, 0, "id-1", "lm317", Path("dbv2/images/a.jpg"))]
    recorded = _trace_inputs({"query": "q", "chunks": ["a"], "figures": figures})
    assert recorded == {"question": "q", "num_chunks": 1, "figures": ["F1"]}


@pytest.fixture(autouse=True)
def _figures_root(monkeypatch, tmp_path):
    """Figure paths resolve under ``tmp_path``, so no test reads ``dbv2/images``."""
    import src.documents

    (tmp_path / "dbv2" / "images").mkdir(parents=True)
    monkeypatch.setattr(src.documents, "_PROJECT_ROOT", tmp_path)
    return tmp_path


def _chunk(summary: str, images: list[str], source: str | None = None, record_id: str | None = None) -> Document:
    """A stored chunk whose figure files hold the given strings as bytes."""
    import json

    import src.documents

    paths = []
    for name in images:
        path = f"dbv2/images/{name}.jpg"
        (src.documents._PROJECT_ROOT / path).write_bytes(name.encode())
        paths.append(path)
    return Document(
        id=record_id,
        page_content=summary,
        metadata={"raw_text": "", "tables_html": "[]", "image_paths": json.dumps(paths)}
        | ({"source": source} if source else {}),
    )


def _b64(text: str) -> str:
    import base64

    return base64.b64encode(text.encode()).decode()


def test_every_retrieved_chunk_reaches_the_prompt():
    """No hidden limit: the prompt holds as many chunks as were retrieved."""
    chunks = [_chunk(f"summary {i}", []) for i in range(1, 8)]
    prompt = _build_text_prompt(chunks, "q")
    assert "--- Chunk 7 ---" in prompt and "summary 7" in prompt
    assert "--- Chunk 8 ---" not in prompt


def _images(blocks: list[dict]) -> list[str]:
    return [block["image_url"]["url"] for block in blocks if block["type"] == "image_url"]


def _labels(blocks: list[dict]) -> list[str]:
    return [block["text"] for block in blocks if block["type"] == "text"]


def test_image_cap_takes_images_in_chunk_order():
    """Chunk 1's images come first, then chunk 2's, until the cap."""
    chunks = [_chunk("a", ["A1", "A2"]), _chunk("b", ["B1", "B2"]), _chunk("c", ["C1"])]
    assert _images(_image_blocks(select_figures(chunks, 3))) == [f"data:image/jpeg;base64,{_b64(x)}" for x in ("A1", "A2", "B1")]


def test_image_cap_zero_sends_every_image():
    chunks = [_chunk("a", ["A1", "A2"]), _chunk("b", ["B1"])]
    assert len(_images(_image_blocks(select_figures(chunks, 0)))) == 3


def test_each_figure_is_labelled_just_before_its_image():
    """Labels number figures in prompt order and name the document and chunk."""
    chunks = [_chunk("a", ["A1"], source="lm317"), _chunk("b", ["B1", "B2"], source="ne555")]
    blocks = _image_blocks(select_figures(chunks, 0))
    assert [block["type"] for block in blocks] == ["text", "image_url"] * 3
    assert _labels(blocks) == [
        "Figure F1, from lm317, chunk 1",
        "Figure F2, from ne555, chunk 2",
        "Figure F3, from ne555, chunk 2",
    ]


def test_unreadable_figure_takes_no_label_or_slot(_figures_root):
    """A missing file is skipped: the next figure becomes F1 and fills the cap."""
    chunk = _chunk("a", ["A1", "A2"])
    (_figures_root / "dbv2" / "images" / "A1.jpg").unlink()
    blocks = _image_blocks(select_figures([chunk], 1))
    assert _labels(blocks) == ["Figure F1, from unknown document, chunk 1"]
    assert _images(blocks) == [f"data:image/jpeg;base64,{_b64('A2')}"]


def test_text_only_model_gets_no_images_and_no_mention_of_them():
    chunks = [_chunk("a", ["A1", "A2"]), _chunk("b", ["B1"])]
    content = _build_message_content(chunks, "q", figures=[])
    assert [part["type"] for part in content] == ["text"]
    assert "image" not in content[0]["text"].lower()


def test_vision_model_gets_images_and_is_told_about_them():
    chunks = [_chunk("a", ["A1"])]
    content = _build_message_content(chunks, "q", select_figures(chunks, 0))
    assert [part["type"] for part in content] == ["text", "text", "image_url"]
    assert "attached images" in content[0]["text"]
    assert "[F1]" in content[0]["text"], "the model is told how to cite a figure"
    assert content[1]["text"] == "Figure F1, from unknown document, chunk 1"


def test_selected_figures_know_their_chunk_position_and_record():
    """Each chosen figure can be turned back into its stable image URL."""
    chunks = [_chunk("a", [], record_id="r1"), _chunk("b", ["B1", "B2"], source="ne555", record_id="r2")]
    figures = select_figures(chunks, 0)
    assert [(f.label, f.chunk_number, f.position, f.record_id, f.source) for f in figures] == [
        ("F1", 2, 0, "r2", "ne555"),
        ("F2", 2, 1, "r2", "ne555"),
    ]


def test_skipped_figure_keeps_the_next_one_at_its_stored_position(_figures_root):
    """A missing first file shifts the label, never the position in the URL."""
    chunk = _chunk("a", ["A1", "A2"], record_id="r1")
    (_figures_root / "dbv2" / "images" / "A1.jpg").unlink()
    (figure,) = select_figures([chunk], 0)
    assert (figure.label, figure.position) == ("F1", 1)


def test_trace_attaches_only_the_sent_figures_by_label():
    from types import SimpleNamespace

    from src.retrieval.generate import _attach_figures_to_trace

    chunks = [_chunk("a", ["A1", "A2"])]
    run_tree = SimpleNamespace(attachments=None)
    _attach_figures_to_trace(run_tree, select_figures(chunks, 1))
    assert run_tree.attachments == {"F1": ("image/jpeg", b"A1")}


def test_choosing_figures_reads_no_file(monkeypatch):
    """Bytes are read for the model request and the trace, not when choosing."""
    chunks = [_chunk("a", ["A1"])]

    def opened(path):
        raise AssertionError(f"select_figures read {path}")

    monkeypatch.setattr(Path, "read_bytes", opened)
    assert [figure.label for figure in select_figures(chunks, 0)] == ["F1"]


def test_figure_outside_the_images_folder_is_not_sent(_figures_root):
    """A stored path is used only if it lies inside IMAGES_DIR, as for the image endpoint."""
    import json

    (_figures_root / "secret.txt").write_bytes(b"secret")
    chunk = Document(page_content="a", metadata={"image_paths": json.dumps(["secret.txt"])})
    assert select_figures([chunk], 0) == []
