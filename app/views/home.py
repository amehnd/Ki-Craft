"""Home page: what the project is, how it works, and where to go next."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from chunk_data import display_name, document_ids_or_error
from components.diagram import MERMAID_DIAGRAM, mermaid_iframe

GITHUB_URL = "https://github.com/moksh-m9u/Multimodal-Rag"


LIVE_APP_URL = "https://multimodal-rag-complex.streamlit.app/"


EVAL_DASHBOARD_URL = (
    "https://multimodal-rag-iquyr6mkijuujmcr8cmikc.streamlit.app/"
)


PROJECT_LINKS: list[tuple[str, str]] = [
    ("GitHub", GITHUB_URL),
    ("Live App", LIVE_APP_URL),
    ("Evaluation Dashboard", EVAL_DASHBOARD_URL),
    ("API Reference", f"{GITHUB_URL}/blob/main/docs/API.md"),
    ("LangSmith", "https://smith.langchain.com/"),
]


CORE_STACK: list[str] = [
    "Python",
    "FastAPI",
    "Streamlit",
    "LangChain",
    "LangSmith",
    "LLMOps",
    "ChromaDB",
    "Tesseract OCR",
    "GLM-4.5V",
    "Gemini",
    "Hugging Face",
]


PIPELINE_STEPS: list[tuple[str, str, str, list[str]]] = [
    (
        "Step 01",
        "Multimodal ingestion of electronic datasheets",
        "Datasheets hide their answers inside pictures: pinout diagrams, application "
        "circuits, characteristic curves, timing waveforms, package drawings. "
        "<code>partition_pdf(strategy=\"hi_res\")</code> splits every datasheet into typed "
        "elements — narrative text, structured HTML tables and base64 image blocks — while "
        "preserving page numbers and reading order so each figure stays associated with the "
        "section that explains it.",
        ["PDF in", "unstructured", "hi_res strategy", "OCR"],
    ),
    (
        "Step 02",
        "OCR + vision-LLM enhanced representations",
        "Tesseract OCR recovers text that physically lives inside images — pin labels, axis "
        "values, curve annotations. Then <b>GLM-4.5V</b>, a vision language model, writes a "
        "multimodal summary of every image- and table-rich chunk. That summary becomes "
        "<code>enhanced_content</code>: a text-only retrieval handle for knowledge that was "
        "originally visual, so a figure can now be found by asking about it.",
        ["Tesseract OCR", "GLM-4.5V", "enhanced_content", "+1024 tok"],
    ),
    (
        "Step 03",
        "Vector storage and diversity-aware retrieval",
        "Each chunk — enhanced content, raw text, tables and images — is embedded with "
        "Granite embeddings and persisted to <b>ChromaDB</b> using cosine similarity over HNSW. "
        "Retrieval uses <b>MMR</b> (the top 5 from a 20-chunk candidate pool) so results "
        "are relevant <i>and</i> diverse instead of near-duplicate paragraphs. The same "
        "chunks are exported to JSON, which is what powers the inspector pages.",
        ["Granite embeddings", "ChromaDB", "cosine HNSW", "MMR k=5 / fetch_k=20"],
    ),
    (
        "Step 04",
        "Chunk Inspector &amp; Dataset Analytics",
        "A granular inspection dashboard where every chunk is opened independently: raw OCR "
        "text, AI-generated summary, decoded images, parsed tables, full metadata and "
        "cross-modal associations. Each chunk gets a 0–100 <b>health score</b> from content "
        "completeness, so weak chunks can be found before they hurt a query. Dataset Analytics "
        "adds distributions and top-20 lists over the whole corpus.",
        ["0–100 health score", "per-chunk drill-down", "histograms", "top-20 lists"],
    ),
    (
        "Step 05",
        "Grounded multimodal query &amp; retrieve",
        "The <b>FastAPI</b> backend runs embed → search → context assembly → generation. The "
        "top-5 relevant chunks are passed to the generation model together with their actual "
        "images and tables, not as flattened text, so answers can describe a pinout or read a "
        "curve. Answers stream back over SSE, with token usage, latency and estimated cost. "
        "Any provider and model can be selected in the sidebar with <b>Bring Your Own API Key</b>.",
        ["Top-5 chunks", "BYO API key", "Gemini / Groq / OpenAI / HF"],
    ),
    (
        "Step 06",
        "LangSmith observability &amp; LLMOps",
        "End-to-end tracing covers embedding, retrieval, multimodal context assembly and "
        "generation. Every run is inspectable in LangSmith: the chunks that were retrieved, the "
        "model calls made, inputs and outputs, latency, token usage and the full execution flow — "
        "so a bad answer can be traced back to the exact stage that caused it.",
        ["end-to-end traces", "latency", "token usage", "inputs / outputs"],
    ),
    (
        "Step 07",
        "LLM-as-a-Judge evaluation framework",
        "A judge model scores both halves of the system: retrieval (relevance, coverage, "
        "ranking, redundancy, noise) and generation (correctness, faithfulness, completeness, "
        "conciseness, image utilisation). Across <b>13 curated datasheet questions</b> the "
        "system reaches an <b>89.8/100</b> average overall score with a <b>92% pass rate</b>, "
        "backed by a live evaluation dashboard with score distribution and failure analysis.",
        ["89.8 / 100 avg", "92% pass rate", "13 cases", "failure analysis"],
    ),
]


EVAL_METRICS: list[tuple[str, str, str]] = [
    ("89.8", "Avg overall judge score", "of 100 across 13 evaluation cases"),
    ("92%", "Pass rate", "12 of 13 cases graded Pass"),
    ("87.0", "Avg retrieval score", "relevance, coverage, ranking, redundancy, noise"),
    ("93.3", "Avg generation score", "correctness, faithfulness, completeness, conciseness"),
]


def _asset(name: str) -> str:
    return str(Path(__file__).resolve().parent.parent / "assets" / name)


def _render_hero() -> None:
    buttons = "".join(
        f'<a class="mmr-btn" href="{url}" target="_blank" rel="noopener">'
        f"{label} &#8599;</a>"
        for label, url in PROJECT_LINKS
    )
    badges = "".join(
        f'<span class="mmr-badge">{tech}</span>' for tech in CORE_STACK
    )

    st.markdown(
        f"""
        <div class="mmr-hero">
          <h2>Multimodal RAG AI Pipeline for Electronic Datasheets</h2>
          <p>A production pipeline that reads datasheets the way an engineer does:
          narrative text, tables, figures, schematics, pinout diagrams and the
          relationships between them &mdash; then answers questions grounded in
          the exact chunks and images that support them.</p>
          <div class="mmr-actions">{buttons}</div>
          <div class="mmr-badges">{badges}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_problem() -> None:
    st.markdown("## The problem with text-only RAG on datasheets")
    st.markdown(
        """
A datasheet rarely states its most important facts in sentences. The answer
lives in the visuals: pin assignments drawn on a package outline, absolute
maximum ratings inside a table, gain and dropout curves, switching waveforms,
typical application circuits. A text-only pipeline retrieves the paragraph
*near* the figure and calls it a day, so the model confidently answers the wrong
part &mdash; or nothing at all.

This project treats every modality as first-class: extraction, enrichment,
storage, retrieval and generation all carry text, tables **and** images together.
"""
    )


def _html_grid(fragments: list[str], css_class: str) -> str:
    """Join HTML fragments into one raw-HTML block safe for markdown.

    Fragments are stripped before joining: markdown ends a raw HTML block on the
    first blank line, and a whitespace-only line counts as blank.  Leaving the
    indentation that trails each triple-quoted fragment in place would escape
    every card after the first one into visible markup.
    """
    body = "".join(f.strip() for f in fragments)
    return f'<div class="{css_class}">{body}</div>'


def _render_pipeline_steps() -> None:
    st.markdown("## What we built &mdash; step by step")

    cards = []
    for step, title, body, tags in PIPELINE_STEPS:
        tag_html = "".join(f'<span class="mmr-tag">{t}</span>' for t in tags)
        cards.append(
            f"""<div class="mmr-card">
  <div class="mmr-step">{step}</div>
  <h4>{title}</h4>
  <p>{body}</p>
  <div class="mmr-tags">{tag_html}</div>
</div>"""
        )

    st.markdown(_html_grid(cards, "mmr-grid"), unsafe_allow_html=True)


def _render_results() -> None:
    st.markdown("## Evaluation results")
    st.markdown(
        f"""
The [LLM-as-a-Judge framework]({EVAL_DASHBOARD_URL}) grades retrieval and
generation separately across **13 curated datasheet questions**. Headline
numbers from the live run:
"""
    )

    cells = []
    for value, label, hint in EVAL_METRICS:
        cells.append(
            f"""<div class="mmr-metric">
  <div class="v">{value}</div>
  <div class="l">{label}</div>
  <div class="h">{hint}</div>
</div>"""
        )

    st.markdown(_html_grid(cells, "mmr-metrics"), unsafe_allow_html=True)
    st.caption(
        "Open the Evaluation Dashboard for the per-question score distribution, "
        "the retrieval-vs-generation comparison and the failure analysis that "
        "explains every non-pass case."
    )


def _render_architecture() -> None:
    st.markdown("## Architecture")
    st.markdown(
        "End-to-end multimodal data flow &mdash; PDF in, grounded answer out, "
        "with every stage traced."
    )

    diagram_html = mermaid_iframe(MERMAID_DIAGRAM)
    if hasattr(st, "iframe"):
        st.iframe(diagram_html, height=710)
    else:  # Streamlit < 1.45, where st.iframe does not exist yet
        import streamlit.components.v1 as components

        components.html(diagram_html, height=710)

    with st.expander("Mermaid source"):
        st.code(MERMAID_DIAGRAM.strip(), language="mermaid")


def _render_how_to_explore() -> None:
    st.markdown("## How to explore this project")
    st.markdown(
        "Pick a datasheet in the **sidebar** and every page below becomes live."
    )

    steps = [
        (
            "Select a datasheet",
            "Sidebar &rarr; *Select a datasheet*. The API currently holds: "
            + ", ".join(f"**{display_name(d)}**" for d in document_ids_or_error()[0])
            + ". The sidebar then reports total chunks and unlocks the filters.",
        ),
        (
            "Chunk Explorer",
            "Dataset-wide metrics, then filter by search text, images, tables, raw "
            "text, enhanced content or chunk ID. Click any row to open that chunk's "
            "health score, AI summary, raw OCR text, decoded images, parsed tables "
            "and full metadata.",
        ),
        (
            "Compare Chunks",
            "Put any two chunks side by side &mdash; useful to see how one visual "
            "region got summarised and how it differs from a text-only chunk.",
        ),
        (
            "Dataset Analytics",
            "Distributions of images, tables, raw length and enhanced length, plus "
            "top-20 lists for largest chunks and most multimodal chunks.",
        ),
        (
            "Query &amp; Retrieve",
            "Ask a question and watch the backend retrieve the top-5 chunks and "
            "stream a grounded answer. Expand any retrieved chunk to see the exact "
            "text, tables and images the model was given, then check token usage, "
            "latency and estimated cost. Choose any provider or model "
            "in the sidebar &mdash; or enable <i>Use Custom Provider</i> and bring "
            "your own API key.",
        ),
        (
            "Evaluation Dashboard",
            f"The judge results live in a [separate deployment]({EVAL_DASHBOARD_URL}): "
            "per-question scores, retrieval vs generation breakdown, metric averages "
            "and root-cause diagnosis for failures.",
        ),
    ]

    st.markdown(
        "\n\n".join(
            f"{i}. **{title}** &mdash; {body}"
            for i, (title, body) in enumerate(steps, 1)
        )
    )


def _render_gallery() -> None:
    shots = [
        ("Chunk Explorer", "dashboard-chunk-explorer.png"),
        ("Query & Retrieve", "dashboard-query.png"),
        ("Dataset Analytics", "dashboard-analytics.png"),
        ("Compare Chunks", "dashboard-compare.png"),
    ]
    available = [(label, _asset(f)) for label, f in shots if Path(_asset(f)).exists()]
    if not available:
        return

    with st.expander("Screenshots &mdash; the app in action"):
        for label, path in available:
            st.image(path, caption=label, width="stretch")


def render() -> None:
    """Project walkthrough shown until a datasheet is selected."""
    _render_hero()
    st.divider()
    _render_problem()
    st.divider()
    _render_pipeline_steps()
    st.divider()
    _render_architecture()
    st.divider()
    _render_results()
    st.divider()
    _render_how_to_explore()
    st.divider()
    _render_gallery()

    st.markdown(
        f"""
        <div class="mmr-hero">
          <h2>Ready to look under the hood?</h2>
          <p>Select a datasheet in the sidebar to inspect real chunks, or jump
          straight to Query &amp; Retrieve to ask the pipeline a question.</p>
          <div class="mmr-actions">
            <a class="mmr-btn" href="{GITHUB_URL}" target="_blank" rel="noopener">Source code &#8599;</a>
            <a class="mmr-btn" href="{EVAL_DASHBOARD_URL}" target="_blank" rel="noopener">Evaluation Dashboard &#8599;</a>
            <a class="mmr-btn" href="{LIVE_APP_URL}" target="_blank" rel="noopener">Live App &#8599;</a>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


