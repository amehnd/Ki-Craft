"""How one stored or retrieved chunk is drawn: its detail tabs and compact card.

Used by the Chunk Explorer, Compare and Query pages. Every function takes a
chunk dict from the API and draws it; none fetches anything.
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
    if score >= 75:
        return "green"
    elif score >= 50:
        return "orange"
    return "red"


def _status_badge(present: bool, label: str) -> str:
    if present:
        return f":green[PASS] {label}"
    return f":red[FAIL] {label}"


def render_detail(chunk: dict) -> None:
    cid = chunk.get("chunk_id", "?")
    images = get_image_urls(chunk)
    tables = get_tables_html(chunk)
    raw = get_raw_text(chunk)
    enhanced = get_enhanced_content(chunk)

    score, label = health_score(chunk)
    color = _health_color(score)

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Chunk ID", cid)
    col2.metric("Images", len(images))
    col3.metric("Tables", len(tables))
    col4.metric("Raw Chars", len(raw))
    col5.metric("Enhanced Chars", len(enhanced))

    st.markdown(f"### Health Score: {score}/100  --  :{color}[{label}]")
    st.progress(score / 100)

    qc1, qc2, qc3, qc4 = st.columns(4)
    qc1.markdown(_status_badge(bool(enhanced), "Enhanced Content"))
    qc2.markdown(_status_badge(bool(raw), "Raw Text"))
    qc3.markdown(_status_badge(bool(images), "Images"))
    qc4.markdown(_status_badge(bool(tables), "Tables"))

    col_e1, col_e2, _ = st.columns([1, 1, 6])
    with col_e1:
        st.download_button(
            "Export JSON",
            data=json.dumps(chunk, indent=2),
            file_name=f"chunk_{cid}.json",
            mime="application/json",
        )
    with col_e2:
        export_txt = (
            f"Chunk {cid}\n\n"
            f"--- Enhanced ---\n{enhanced}\n\n"
            f"--- Raw ---\n{raw}"
        )
        st.download_button(
            "Export TXT",
            data=export_txt,
            file_name=f"chunk_{cid}.txt",
            mime="text/plain",
        )

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Enhanced Content", "Raw Text", "Images", "Tables", "Metadata"]
    )

    with tab1:
        _render_enhanced_tab(enhanced)
    with tab2:
        _render_raw_tab(raw)
    with tab3:
        _render_images_tab(images)
    with tab4:
        _render_tables_tab(tables)
    with tab5:
        _render_metadata_tab(chunk)


def _render_enhanced_tab(text: str) -> None:
    st.markdown(text)

    wc = word_count(text)
    cc = len(text)

    col1, col2 = st.columns(2)
    col1.metric("Words", wc)
    col2.metric("Characters", cc)

    with st.expander("Show / Hide"):
        st.markdown(text)


def _render_raw_tab(text: str) -> None:
    wc = word_count(text)
    cc = len(text)

    col1, col2 = st.columns(2)
    col1.metric("Words", wc)
    col2.metric("Characters", cc)

    st.text_area(
        "Raw Text",
        value=text,
        height=400,
        disabled=True,
        label_visibility="collapsed",
    )

    st.download_button(
        "Download", data=text, file_name="raw_text.txt", mime="text/plain"
    )


def _render_images_tab(image_urls: list[str]) -> None:
    """Show each figure; the browser loads it from the API's image URL."""
    if not image_urls:
        st.info("No images found in this chunk.")
        return

    for i, url in enumerate(image_urls, 1):
        st.subheader(f"Image {i}")
        st.image(url, width="stretch")
        st.markdown(f"[Open full size]({url})")


def _render_tables_tab(tables: list[str]) -> None:
    if not tables:
        st.info("No tables found in this chunk.")
        return

    for i, html in enumerate(tables, 1):
        st.subheader(f"Table {i}")
        df = parse_table_html(html)
        if df is not None:
            st.dataframe(df, width="stretch")
        else:
            st.warning(
                f"Could not parse Table {i} as DataFrame. Showing raw HTML."
            )
            st.html(html)

        with st.expander(f"Raw HTML -- Table {i}"):
            st.code(html, language="html")


def _render_metadata_tab(chunk: dict) -> None:
    """Counts and the chunk's identifying fields."""
    col1, col2 = st.columns(2)
    col1.metric("Image Count", len(get_image_urls(chunk)))
    col2.metric("Table Count", len(get_tables_html(chunk)))

    st.divider()
    st.json(chunk_fields(chunk))


def render_chunk_card(chunk: dict) -> None:
    images = get_image_urls(chunk)
    tables = get_tables_html(chunk)
    raw = get_raw_text(chunk)
    enhanced = get_enhanced_content(chunk)
    score, label = health_score(chunk)
    color = _health_color(score)

    st.metric("Health Score", f"{score}/100")
    st.progress(score / 100)
    st.write(f"**Images:** {len(images)}  |  **Tables:** {len(tables)}")
    st.write(
        f"**Raw:** {len(raw)} chars  |  **Enhanced:** {len(enhanced)} chars"
    )

    with st.expander("Enhanced Content"):
        st.markdown(enhanced or "*empty*")
    with st.expander("Raw Text"):
        st.text(raw or "*empty*")
    with st.expander(f"Images ({len(images)})"):
        for i, url in enumerate(images, 1):
            st.image(url, width=300, caption=f"Image {i}")
    with st.expander(f"Tables ({len(tables)})"):
        for i, html in enumerate(tables, 1):
            df = parse_table_html(html)
            if df is not None:
                st.dataframe(df)
            else:
                st.html(html)
    with st.expander("Metadata"):
        st.json(chunk_fields(chunk))


def render_retrieved_chunk(payload: dict, cited: bool) -> None:
    """Draw one retrieved chunk as a collapsed panel with its content in tabs.

    Nothing here is a widget (no text area, no keys), so the same chunk can
    be drawn twice in one run: once when retrieval arrives, again with its
    citation mark when the answer is done.

    Args:
        payload: One entry of the API's ``chunks`` list.
        cited: Whether the answer cites this chunk.
    """
    raw_text = payload.get("raw_text", "")
    tables = payload.get("tables_html", [])
    image_urls = payload.get("image_urls", [])
    enhanced = payload.get("enhanced_content", "")

    parts = [f"C{payload.get('chunk_id')}", payload.get("source") or "unknown document"]
    parts.append(f"{len(image_urls)} figure{'s' if len(image_urls) != 1 else ''}")
    parts.append(f"{len(tables)} table{'s' if len(tables) != 1 else ''}")
    label = " · ".join(parts) + ("  :green[cited]" if cited else "")

    with st.expander(label):
        tab_e, tab_r, tab_i, tab_t = st.tabs(["Summary", "Original text", "Figures", "Tables"])
        with tab_e:
            st.markdown(enhanced)
        with tab_r:
            st.caption(f"{len(raw_text.split())} words")
            with st.container(height=250):
                st.text(raw_text)
        with tab_i:
            if not image_urls:
                st.info("No figures in this chunk.")
            for j, url in enumerate(image_urls):
                st.image(url, width=400, caption=f"Figure {j + 1} of this chunk")
        with tab_t:
            if not tables:
                st.info("No tables in this chunk.")
            for j, html in enumerate(tables):
                st.markdown(f"**Table {j + 1}**")
                df = parse_table_html(html)
                if df is not None:
                    st.dataframe(df, width="stretch")
                else:
                    st.warning("Could not parse table.")
                    st.code(html, language="html")
