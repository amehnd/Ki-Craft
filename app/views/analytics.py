"""Dataset Analytics page (``/analytics``): distributions over one document's chunks."""

from __future__ import annotations

import plotly.express as px
import streamlit as st

from chunk_data import build_chunk_df
from components.dataset import render_totals, require_dataset

_PURPOSE = (
    "Corpus-level view: distributions of images, tables, raw length and enhanced "
    "length per chunk, plus top-20 lists for the largest and most multimodal chunks."
)


def render() -> None:
    """Draw the page: document picker in the sidebar, charts in the body."""
    data = require_dataset("Dataset Analytics", _PURPOSE)
    if data is not None:
        render_analytics(data)

def render_analytics(data: list[dict]) -> None:
    """Draw totals, four histograms and three top-20 tables for one document's chunks."""
    st.header("Dataset Analytics")

    df = build_chunk_df(data)
    if df.empty:
        st.info("No data to analyse.")
        return

    render_totals(df)

    col1, col2 = st.columns(2)
    with col1:
        fig1 = px.histogram(
            df,
            x="Images",
            title="Distribution of Images per Chunk",
            labels={"Images": "Image Count", "count": "Chunks"},
            nbins=int(df["Images"].max()) + 1,
        )
        st.plotly_chart(fig1, width="stretch")
    with col2:
        fig2 = px.histogram(
            df,
            x="Tables",
            title="Distribution of Tables per Chunk",
            labels={"Tables": "Table Count", "count": "Chunks"},
            nbins=int(df["Tables"].max()) + 1,
        )
        st.plotly_chart(fig2, width="stretch")

    col3, col4 = st.columns(2)
    with col3:
        fig3 = px.histogram(
            df,
            x="Raw Len",
            title="Distribution of Raw Text Length",
            labels={"Raw Len": "Character Count", "count": "Chunks"},
        )
        st.plotly_chart(fig3, width="stretch")
    with col4:
        fig4 = px.histogram(
            df,
            x="Enh Len",
            title="Distribution of Enhanced Content Length",
            labels={"Enh Len": "Character Count", "count": "Chunks"},
        )
        st.plotly_chart(fig4, width="stretch")

    st.subheader("Top 20 Lists")
    tcol1, tcol2, tcol3 = st.columns(3)

    with tcol1:
        st.markdown("**Largest Chunks (raw text)**")
        top_raw = df.nlargest(20, "Raw Len")[["Chunk ID", "Raw Len"]]
        st.dataframe(top_raw, width="stretch")

    with tcol2:
        st.markdown("**Most Images**")
        top_img = df.nlargest(20, "Images")[["Chunk ID", "Images"]]
        st.dataframe(top_img, width="stretch")

    with tcol3:
        st.markdown("**Most Tables**")
        top_tbl = df.nlargest(20, "Tables")[["Chunk ID", "Tables"]]
        st.dataframe(top_tbl, width="stretch")


