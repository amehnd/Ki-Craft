"""Tests for pure helper functions in src/retrieval/generate.py.

Cloud/model dependencies are optional; the module is skipped entirely when
they are not installed so the suite stays lightweight.
"""

import pytest

pytest.importorskip("langsmith")
pytest.importorskip("langchain_core")

from langchain_core.documents import Document

from src.retrieval.generate import (
    _build_message_content,
    _build_text_prompt,
    _collect_images,
    _capture_usage,
    _estimate_cost,
    _extract_text,
    _prompt_stats,
    _trace_inputs,
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


def _chunk(summary: str, images: list[str]) -> Document:
    """A chunk in the nested schema with the given base64 images."""
    import json

    return Document(
        page_content=summary,
        metadata={"original_content": json.dumps({"raw_text": "", "tables_html": [], "images_base64": images})},
    )


def test_every_retrieved_chunk_reaches_the_prompt():
    """No hidden limit: the prompt holds as many chunks as were retrieved."""
    chunks = [_chunk(f"summary {i}", []) for i in range(1, 8)]
    prompt = _build_text_prompt(chunks, "q")
    assert "--- Chunk 7 ---" in prompt and "summary 7" in prompt
    assert "--- Chunk 8 ---" not in prompt


def test_image_cap_takes_images_in_chunk_order():
    """Chunk 1's images come first, then chunk 2's, until the cap."""
    chunks = [_chunk("a", ["A1", "A2"]), _chunk("b", ["B1", "B2"]), _chunk("c", ["C1"])]
    urls = [block["image_url"]["url"] for block in _collect_images(chunks, 3)]
    assert urls == [f"data:image/jpeg;base64,{x}" for x in ("A1", "A2", "B1")]


def test_image_cap_zero_sends_every_image():
    chunks = [_chunk("a", ["A1", "A2"]), _chunk("b", ["B1"])]
    assert len(_collect_images(chunks, 0)) == 3


def test_text_only_model_gets_no_images_and_no_mention_of_them():
    chunks = [_chunk("a", ["A1", "A2"]), _chunk("b", ["B1"])]
    content = _build_message_content(chunks, "q", max_images=0, send_images=False)
    assert [part["type"] for part in content] == ["text"]
    assert "image" not in content[0]["text"].lower()


def test_vision_model_gets_images_and_is_told_about_them():
    chunks = [_chunk("a", ["A1"])]
    content = _build_message_content(chunks, "q", max_images=0, send_images=True)
    assert [part["type"] for part in content] == ["text", "image_url"]
    assert "attached images" in content[0]["text"]
