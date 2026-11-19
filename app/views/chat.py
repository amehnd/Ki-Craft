"""Query & Retrieve page (``/chat``): ask a question and watch the answer and its evidence arrive.

One question, all over HTTP through ``api_client.stream``:

    click Ask -> POST /v1/query/stream
      retrieval event (about 1 s)  -> the retrieved chunks are drawn at once
      token events                 -> the answer grows in place
      done event                   -> cited figures above the answer, the other
                                      figures below it, chunks marked as cited,
                                      token usage and cost
      error event                  -> the error is shown; the chunks stay

The page reserves one empty slot per section before the request starts
(``_Slots``), so each event fills its own section without moving the others.
Streamlit reruns the whole script on every click, so the finished result is
kept in session state and drawn again into the same slots on later runs.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import streamlit as st

from api_client import API_BASE_URL
from api_client import health as api_health
from api_client import stream as api_stream
from components.answer_view import (
    cited_first,
    render_chunks,
    render_figure_row,
    render_unknown_citations,
    render_usage,
)
from components.example_questions import ask_example, render_example_buttons
from components.model_settings import render_model_settings


@dataclass
class _Slots:
    """One placeholder per page section, top to bottom, created before an answer starts."""

    progress: st.delta_generator.DeltaGenerator
    chunks: st.delta_generator.DeltaGenerator
    cited: st.delta_generator.DeltaGenerator
    answer: st.delta_generator.DeltaGenerator
    others: st.delta_generator.DeltaGenerator
    usage: st.delta_generator.DeltaGenerator


def _make_slots() -> _Slots:
    """Reserve the page sections in display order; each starts empty."""
    return _Slots(*(st.empty() for _ in range(6)))


def _backend_problem() -> str | None:
    """Why the API cannot answer, or None when ``/health`` reports it ready."""
    try:
        status = api_health()
    except Exception as exc:
        return f"is unreachable ({exc})"
    if status.get("status") != "ok":
        return f"is running but unhealthy: {status.get('detail', status)}"
    return None


def _requested_llm() -> dict | None:
    """The ``llm`` object for the request: None for the server's default model.

    Raises:
        ValueError: If "Use my own model" is on but a provider, model or key is missing.
    """
    state = st.session_state
    if not state.get("use_custom_provider"):
        return None
    model = (state.get("llm_custom_model") or "").strip() or state.get("llm_model")
    if not (state.get("llm_provider") and model and state.get("llm_api_key")):
        raise ValueError("Choose a provider and model and enter your API key, or untick \"Use my own model\".")
    return {
        "provider": state.llm_provider,
        "model": model,
        "api_key": state.llm_api_key,
        "temperature": state.llm_temperature,
        "max_tokens": state.llm_max_tokens or None,
    }


def _draw_answer(slots: _Slots, answer: str, done: dict | None, streaming: bool = False) -> None:
    """Fill the answer and, once it is done, its figures and usage.

    Args:
        slots: The page sections.
        answer: The answer text so far.
        done: The ``done`` event's data, or None while the answer streams.
        streaming: True while tokens are still arriving; shows a cursor.
    """
    slots.answer.markdown(f"### Answer\n\n{answer}{' ▌' if streaming else ''}")
    if done is None:
        return
    ordered = cited_first(done.get("figures") or [], done.get("citations") or {})
    with slots.cited.container():
        render_figure_row("Cited figures", [figure for figure, cited in ordered if cited])
    with slots.others.container():
        render_unknown_citations(done.get("citations") or {})
        render_figure_row("Other figures the model saw", [figure for figure, cited in ordered if not cited])
    with slots.usage.container():
        render_usage(done.get("usage"), st.session_state.chat_session_usage)


def _add_to_session_usage(usage: dict | None) -> None:
    """Add one answer's tokens and cost to this tab's running totals."""
    if not usage:
        return
    totals = st.session_state.chat_session_usage
    totals["answers"] += 1
    totals["input_tokens"] += usage.get("input_tokens") or 0
    totals["output_tokens"] += usage.get("output_tokens") or 0
    totals["cost"] += usage.get("estimated_cost_usd") or 0.0


def _ask(question: str, slots: _Slots) -> None:
    """Send the question and fill each section as its stream event arrives.

    The result is saved to session state as it arrives, so a later rerun
    redraws whatever was received, including after an error.
    """
    state = st.session_state
    state.chat_retrieval, state.chat_answer, state.chat_done, state.chat_error = None, None, None, None
    try:
        llm = _requested_llm()
    except ValueError as exc:
        slots.progress.error(str(exc))
        return

    started = time.perf_counter()
    slots.progress.info("Retrieving chunks ...")
    tokens: list[str] = []
    try:
        for event, data in api_stream(question, top_k=state.llm_top_k, max_images=state.llm_max_images, llm=llm):
            elapsed = time.perf_counter() - started
            if event == "retrieval":
                state.chat_retrieval = data
                with slots.chunks.container():
                    render_chunks(data.get("chunks", []), [])
                slots.progress.info(
                    f"Found {data.get('num_chunks', 0)} chunks in {elapsed:.1f} s. Waiting for the model ..."
                )
            elif event == "token":
                if not tokens:
                    slots.progress.info(f"The model started answering after {elapsed:.1f} s ...")
                tokens.append(data.get("delta", ""))
                _draw_answer(slots, "".join(tokens), None, streaming=True)
            elif event == "done":
                state.chat_answer, state.chat_done = data.get("answer", ""), data
                _add_to_session_usage(data.get("usage"))
                slots.progress.success(f"Answered in {elapsed:.1f} s.")
                _draw_answer(slots, state.chat_answer, data)
                with slots.chunks.container():
                    render_chunks(state.chat_retrieval.get("chunks", []), data.get("citations", {}).get("chunks", []))
            elif event == "error":
                state.chat_error = data.get("message", "Unknown backend error")
                slots.progress.error(f"The model failed: {state.chat_error}")
    except Exception as exc:
        state.chat_error = str(exc)
        slots.progress.error(f"Query failed: {exc}")
        return
    if tokens and state.chat_done is None:
        # The stream ended mid-answer: keep what arrived, without a cursor.
        state.chat_answer = "".join(tokens)
        _draw_answer(slots, state.chat_answer, None)


def _draw_saved(slots: _Slots) -> None:
    """Draw the last question's result from session state, as it was left."""
    state = st.session_state
    if state.chat_error:
        slots.progress.error(state.chat_error)
    done = state.chat_done or {}
    if state.chat_retrieval:
        with slots.chunks.container():
            render_chunks(state.chat_retrieval.get("chunks", []), done.get("citations", {}).get("chunks", []))
    if state.chat_answer:
        _draw_answer(slots, state.chat_answer, state.chat_done)


def _clear() -> None:
    """Forget the last question's result (the question text stays)."""
    for key in ("chat_retrieval", "chat_answer", "chat_done", "chat_error"):
        st.session_state[key] = None


def render() -> None:
    """Draw the page: settings in the sidebar, the question and its answer in the body."""
    render_model_settings()
    st.header("Query & Retrieve")
    st.caption(
        "Ask about the indexed datasheets, or pick an example. The retrieved chunks appear "
        "first, then the answer as the model writes it, with the figures it cites right above it."
    )
    problem = _backend_problem()
    if problem:
        st.error(
            f"Backend API at `{API_BASE_URL}` {problem}. Set API_BASE_URL to the backend you "
            "want, e.g. `API_BASE_URL=http://127.0.0.1:8001 streamlit run app/dashboard.py`."
        )

    has_result = bool(st.session_state.chat_retrieval or st.session_state.chat_error)
    # Drawn above the question box, so a clicked example can still fill it this run.
    with st.expander("Try asking", expanded=not has_result):
        clicked = render_example_buttons("chat")
    if clicked:
        ask_example(clicked)

    question = st.text_area("Question", key="chat_query", placeholder="Ask a question about your documents ...")
    col1, col2, _ = st.columns([1, 1, 4])
    submitted = col1.button("Ask", type="primary", width="stretch")
    col2.button("Clear", width="stretch", on_click=_clear)
    # Set by an example question, here or on the home page: ask it without a second click.
    if st.session_state.chat_ask_now:
        st.session_state.chat_ask_now = False
        submitted = True

    slots = _make_slots()
    if submitted and question.strip():
        _ask(question.strip(), slots)
    elif has_result:
        _draw_saved(slots)
    else:
        slots.progress.info("Enter a question and press Ask.")
