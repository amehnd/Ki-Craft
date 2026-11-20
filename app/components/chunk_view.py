"""How one stored or retrieved chunk is drawn, on the Explorer, Compare and Query pages.

A stored chunk (``GET /v1/documents/{id}/chunks``) and a retrieved one (the
query stream's ``chunks``) carry the same content fields: ``enhanced_content``,
``raw_text``, ``tables_html`` and ``image_urls``. ``render_content_tabs`` draws
those fields the same way on every page; each page adds only its own frame
around it: the full detail on Explorer, a compact card on Compare, a collapsed
panel marked "cited" on Query. Nothing here fetches anything.
"""

from __future__ import annotations

import json

import streamlit as st

from chunk_data import (
    chunk_fields,
    get_enhanced_content,
    get_image_urls,
    get_raw_text,
    get_tables_html,
    health_score,
    parse_table_html,
    word_count,
)


def _health_color(score: int) -> str:
    """The Streamlit colour name for a health score: green, orange or red."""
    if score >= 75:
        return "green"
    if score >= 50:
        return "orange"
    return "red"


def _status_badge(present: bool, label: str) -> str:
    """A coloured PASS or FAIL label in Streamlit Markdown, for one kind of content."""
    if present:
        return f":green[PASS] {label}"
    return f":red[FAIL] {label}"


def _plural(count: int, noun: str) -> str:
    """``1 figure``, ``2 figures``."""
    return f"{count} {noun}{'' if count == 1 else 's'}"


def _render_text(text: str) -> None:
    """The original text in a scrolling box, with its word count."""
    if not text:
        st.info("No original text in this chunk.")
        return
    st.caption(f"{word_count(text)} words, {len(text)} characters")
    with st.container(height=300):
        st.text(text)


def _render_figures(image_urls: list[str]) -> None:
    """Each figure, loaded by the browser from the API's image URL."""
    if not image_urls:
        st.info("No figures in this chunk.")
        return
    for position, url in enumerate(image_urls, 1):
        st.image(url, width="stretch", caption=f"Figure {position} of this chunk")
        st.markdown(f"[Open full size]({url})")


def _render_tables(tables: list[str]) -> None:
    """Each table as a DataFrame, or as rendered HTML when pandas cannot read it."""
    if not tables:
        st.info("No tables in this chunk.")
        return
    for position, html in enumerate(tables, 1):
        st.markdown(f"**Table {position}**")
        df = parse_table_html(html)
        if df is not None:
            st.dataframe(df, width="stretch")
        else:
            st.caption("This table could not be read as rows and columns; shown as HTML.")
            st.html(html)


def render_content_tabs(chunk: dict, with_metadata: bool = False) -> None:
    """Draw a chunk's summary, original text, figures and tables, one tab each.

    Nothing here is a widget (no text area, no keys), so the same chunk can be
    drawn twice in one run, as the Query page does while an answer streams.

    Args:
        chunk: A stored or retrieved chunk dict from the API.
        with_metadata: Add a tab with the chunk's identifying fields.
    """
    images = get_image_urls(chunk)
    tables = get_tables_html(chunk)
    names = ["Summary", "Original text", f"Figures ({len(images)})", f"Tables ({len(tables)})"]
    if with_metadata:
        names.append("Metadata")
    tabs = st.tabs(names)
    with tabs[0]:
        st.markdown(get_enhanced_content(chunk) or "*No summary.*")
    with tabs[1]:
        _render_text(get_raw_text(chunk))
    with tabs[2]:
        _render_figures(images)
    with tabs[3]:
        _render_tables(tables)
    if with_metadata:
        with tabs[4]:
            st.json(chunk_fields(chunk))


def render_detail(chunk: dict) -> None:
    """Draw one chunk in full for the Chunk Explorer: counts, health, export buttons, content.

    Args:
        chunk: One chunk dict from ``GET /v1/documents/{id}/chunks``.
    """
    cid = chunk.get("chunk_id", "?")
    images = get_image_urls(chunk)
    tables = get_tables_html(chunk)
    raw = get_raw_text(chunk)
    enhanced = get_enhanced_content(chunk)
    score, label = health_score(chunk)

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Chunk ID", cid)
    col2.metric("Images", len(images))
    col3.metric("Tables", len(tables))
    col4.metric("Raw Chars", len(raw))
    col5.metric("Enhanced Chars", len(enhanced))

    st.markdown(f"### Health Score: {score}/100  --  :{_health_color(score)}[{label}]")
    st.progress(score / 100)

    qc1, qc2, qc3, qc4 = st.columns(4)
    qc1.markdown(_status_badge(bool(enhanced), "Enhanced Content"))
    qc2.markdown(_status_badge(bool(raw), "Raw Text"))
    qc3.markdown(_status_badge(bool(images), "Images"))
    qc4.markdown(_status_badge(bool(tables), "Tables"))

    with st.container(horizontal=True, gap="small"):
        st.download_button(
            "Export JSON",
            data=json.dumps(chunk, indent=2),
            file_name=f"chunk_{cid}.json",
            mime="application/json",
        )
        st.download_button(
            "Export TXT",
            data=f"Chunk {cid}\n\n--- Enhanced ---\n{enhanced}\n\n--- Raw ---\n{raw}",
            file_name=f"chunk_{cid}.txt",
            mime="text/plain",
        )

    render_content_tabs(chunk, with_metadata=True)


def render_chunk_card(chunk: dict) -> None:
    """Draw a compact view of one chunk for the Compare page: health, counts, content."""
    score, _ = health_score(chunk)
    st.metric("Health Score", f"{score}/100")
    st.progress(score / 100)
    st.write(
        f"**Raw:** {len(get_raw_text(chunk))} chars  |  "
        f"**Enhanced:** {len(get_enhanced_content(chunk))} chars"
    )
    render_content_tabs(chunk, with_metadata=True)


def render_retrieved_chunk(payload: dict, cited: bool) -> None:
    """Draw one retrieved chunk as a collapsed panel, marked when the answer cites it.

    Args:
        payload: One entry of the API's ``chunks`` list.
        cited: Whether the answer cites this chunk.
    """
    parts = [
        f"C{payload.get('chunk_id')}",
        payload.get("source") or "unknown document",
        _plural(len(get_image_urls(payload)), "figure"),
        _plural(len(get_tables_html(payload)), "table"),
    ]
    label = " · ".join(parts) + ("  :green[cited]" if cited else "")
    with st.expander(label):
        render_content_tabs(payload)
