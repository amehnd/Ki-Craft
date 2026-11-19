"""Dataset Analytics page (``/analytics``): distributions over one document's chunks."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from chunk_data import get_enhanced_content, get_image_urls, get_raw_text, get_tables_html
from components.dataset import require_dataset

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

    df = pd.DataFrame(
        [
            {
                "chunk_id": c.get("chunk_id"),
                "images": len(get_image_urls(c)),
                "tables": len(get_tables_html(c)),
                "raw_len": len(get_raw_text(c)),
                "enhanced_len": len(get_enhanced_content(c)),
            }
            for c in data
        ]
    )

    if df.empty:
        st.info("No data to analyse.")
        return

    m1, m2, m3 = st.columns(3)
    m1.metric("Total Images", df["images"].sum())
    m2.metric("Total Tables", df["tables"].sum())
    m3.metric("Avg Chunk Size (chars)", f"{df['raw_len'].mean():.0f}")

    col1, col2 = st.columns(2)
    with col1:
        fig1 = px.histogram(
            df,
            x="images",
            title="Distribution of Images per Chunk",
            labels={"images": "Image Count", "count": "Chunks"},
            nbins=int(df["images"].max()) + 1,
        )
        st.plotly_chart(fig1, width="stretch")
    with col2:
        fig2 = px.histogram(
            df,
            x="tables",
            title="Distribution of Tables per Chunk",
            labels={"tables": "Table Count", "count": "Chunks"},
            nbins=int(df["tables"].max()) + 1,
        )
        st.plotly_chart(fig2, width="stretch")

    col3, col4 = st.columns(2)
    with col3:
        fig3 = px.histogram(
            df,
            x="raw_len",
            title="Distribution of Raw Text Length",
            labels={"raw_len": "Character Count", "count": "Chunks"},
        )
        st.plotly_chart(fig3, width="stretch")
    with col4:
        fig4 = px.histogram(
            df,
            x="enhanced_len",
            title="Distribution of Enhanced Content Length",
            labels={"enhanced_len": "Character Count", "count": "Chunks"},
        )
        st.plotly_chart(fig4, width="stretch")

    st.subheader("Top 20 Lists")
    tcol1, tcol2, tcol3 = st.columns(3)

    with tcol1:
        st.markdown("**Largest Chunks (raw text)**")
        top_raw = df.nlargest(20, "raw_len")[["chunk_id", "raw_len"]]
        st.dataframe(top_raw, width="stretch")

    with tcol2:
        st.markdown("**Most Images**")
        top_img = df.nlargest(20, "images")[["chunk_id", "images"]]
        st.dataframe(top_img, width="stretch")

    with tcol3:
        st.markdown("**Most Tables**")
        top_tbl = df.nlargest(20, "tables")[["chunk_id", "tables"]]
        st.dataframe(top_tbl, width="stretch")


