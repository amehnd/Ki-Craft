"""Query & Retrieve page (``/chat``): ask a question, stream the answer and its evidence."""

from __future__ import annotations

import streamlit as st

from api_client import API_BASE_URL
from api_client import health as api_health
from api_client import query as api_query
from api_client import stream as api_stream
from components.answer_view import remember_answer, render_figures, render_retrieval, render_usage
from components.model_settings import render_model_settings


def render() -> None:
    """Draw the page: model settings in the sidebar, question and answer in the body."""
    render_model_settings()
    render_chat_page()

def render_chat_page() -> None:
    st.header("Query & Retrieve")

    st.markdown(
        "Enter a query to search the vector store and generate "
        "a multimodal answer with referenced chunks. All retrieval and "
        "generation runs on the FastAPI backend."
    )

    # Check the backend is reachable before doing anything else.
    backend_problem = None
    try:
        backend_status = api_health()
        if backend_status.get("status") != "ok":
            backend_problem = f"is running but unhealthy: {backend_status.get('detail', backend_status)}"
    except Exception as exc:
        backend_problem = f"is unreachable ({exc})"

    if backend_problem:
        st.error(
            f"Backend API at `{API_BASE_URL}` {backend_problem}. "
            "Set API_BASE_URL to the backend you want, e.g. "
            "`API_BASE_URL=http://127.0.0.1:8001 streamlit run app/dashboard.py`."
        )
    else:
        st.success(f"Backend connected — vector store: {backend_status.get('vector_store')}")

    query = st.text_area(
        "Query",
        value=st.session_state.chat_query,
        placeholder="Ask a question about your documents ...",
        key="chat_query_input",
    )

    col1, col2 = st.columns([1, 5])
    with col1:
        submitted = st.button("Search", type="primary", width="stretch")
    with col2:
        clear = st.button("Clear", width="stretch")
        if clear:
            st.session_state.chat_answer = None
            st.session_state.chat_retrieval = None
            st.session_state.chat_usage = None
            st.session_state.chat_figures = []
            st.session_state.chat_citations = {}
            st.rerun()

    if submitted and query.strip():
        st.session_state.chat_query = query
        try:
            use_custom = st.session_state.get("use_custom_provider", False)

            if use_custom:
                # Custom provider mode: POST /v1/query with an llm object
                provider = st.session_state.get("llm_provider")
                model = st.session_state.get("llm_model")
                custom_model = (st.session_state.get("llm_custom_model") or "").strip()
                if custom_model:
                    model = custom_model
                api_key = st.session_state.get("llm_api_key")
                temperature = st.session_state.get("llm_temperature", 0.0)
                max_tokens = st.session_state.get("llm_max_tokens", 0) or None
                max_images = st.session_state.get("llm_max_images", 3)

                if not (provider and model and api_key):
                    st.error("Please select provider, model, and enter API key for custom provider mode.")
                    return

                status = st.status(f"Querying {provider}/{model} ...", expanded=True)
                answer_placeholder = st.empty()

                with st.spinner("Retrieving chunks and generating answer..."):
                    result = api_query(
                        query,
                        top_k=st.session_state.get("llm_top_k", 5),
                        max_images=max_images,
                        llm={
                            "provider": provider,
                            "model": model,
                            "api_key": api_key,
                            "temperature": temperature,
                            "max_tokens": max_tokens,
                        },
                    )

                retrieval = result
                chunks = retrieval.get("chunks", [])
                st.session_state.chat_retrieval = retrieval
                status.update(
                    label=f"Retrieved {len(chunks)} chunks",
                    state="complete",
                    expanded=False,
                )
                render_retrieval(retrieval)

                answer_placeholder.markdown(f"### Answer\n\n{result.get('answer', '')}")
                remember_answer(result)
                render_figures(st.session_state.chat_figures, st.session_state.chat_citations)
                render_usage(result.get("usage"))

            else:
                # Default backend mode - use streaming
                events = api_stream(
                    query,
                    top_k=st.session_state.get("llm_top_k", 5),
                    max_images=st.session_state.get("llm_max_images"),
                )
                status = st.status("Retrieving relevant chunks ...", expanded=True)
                answer_placeholder = st.empty()
                collected: list[str] = []

                for event, data in events:
                    if event == "retrieval":
                        retrieval = data
                        chunks = retrieval.get("chunks", [])
                        st.session_state.chat_retrieval = retrieval
                        status.update(
                            label=f"Retrieved {len(chunks)} chunks",
                            state="complete",
                            expanded=False,
                        )
                        render_retrieval(retrieval)
                    elif event == "token":
                        collected.append(data.get("delta", ""))
                        answer_placeholder.markdown(f"### Answer\n\n{''.join(collected)}▌")
                    elif event == "done":
                        answer_placeholder.markdown(f"### Answer\n\n{data.get('answer', '')}")
                        remember_answer(data)
                        render_figures(st.session_state.chat_figures, st.session_state.chat_citations)
                        render_usage(data.get("usage"))
                    elif event == "error":
                        status.update(label="Answer generation failed", state="error")
                        st.error(data.get("message", "Unknown backend error"))

                if not collected:
                    answer_placeholder.markdown("### Answer\n\n*No answer returned.*")
        except Exception as e:
            st.error(f"Query failed: {e}")

    # Display persisted results on subsequent script runs.
    answer = st.session_state.get("chat_answer")
    retrieval = st.session_state.get("chat_retrieval")

    if answer is None and retrieval is None:
        st.info("Enter a query and press Search to get started.")
        return

    if submitted and query.strip():
        return

    if retrieval:
        render_retrieval(retrieval)

    if answer:
        st.markdown("### Answer")
        st.markdown(answer)
        render_figures(st.session_state.get("chat_figures") or [], st.session_state.get("chat_citations") or {})

    render_usage(st.session_state.get("chat_usage"))


