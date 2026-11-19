"""Home page (``/``): what KI-CRAFT does for you, and where to go next.

Written as a product page for someone who reads datasheets, not as a tour of
the code. The copy follows the README; the document list is read live from
``GET /v1/documents``. ``app/dashboard.py`` passes in the pages this one links
to, because a page link needs the registered page object.
"""

from __future__ import annotations

import streamlit as st

from chunk_data import display_name, list_document_summaries
from components.diagram import MERMAID_DIAGRAM, mermaid_iframe

GITHUB_URL = "https://github.com/moksh-m9u/Multimodal-Rag"

# (title, body) of each card in "What it does".
_FEATURES: list[tuple[str, str]] = [
    (
        "Answers from text, tables and figures",
        "Pinouts, curves and application circuits live in pictures. KI-CRAFT reads the "
        "figures and tables of the relevant sections, not just the paragraphs around them.",
    ),
    (
        "Shows its evidence",
        "Every answer cites the sections it used, as [C1], and the figures, as [IMG1]. The "
        "cited figures appear right above the answer, so you can check them at a glance.",
    ),
    (
        "Refuses to guess",
        "It says when the documents do not state something, keeps units and test "
        "conditions, and never mixes values from different parts.",
    ),
    (
        "Brings your own model",
        "Gemini by default. Any listed model from Gemini, OpenAI, Groq or Hugging Face "
        "works with your own API key.",
    ),
]

_AUDIENCE: list[tuple[str, str]] = [
    ("KiCad users", "tired of cross-checking every pin, capacitor value and footnote against a 60-page PDF."),
    ("Hardware engineers", "checking a design: dropout at a given load, the output voltage formula, the recommended layout."),
    ("Students and makers", "learning to read datasheets: what a pin does, typical versus maximum, what a graph shows."),
    ("Teams", "who want an open alternative to closed AI EDA tools, one they can run, inspect and extend."),
]

_EXAMPLE_QUESTIONS: list[str] = [
    "What is the LM317 reference voltage, and under which conditions?",
    "Show me the LM2596 typical application circuit and its component values.",
    "How does the TLV1117 dropout voltage change with output current?",
    "What does each pin of the NE555 do?",
]

# (phase, goal, status)
_ROADMAP: list[tuple[str, str, str]] = [
    ("1. Datasheet assistant", "Trustworthy answers from datasheets: text, tables, figures, cited", "In progress"),
    ("2. Schematic review", "Check a KiCad schematic against the datasheets of its parts", "Planned"),
    ("3. Design agent", "Draft schematics in KiCad from requirements, assist with layout", "Planned"),
]


def _cards(items: list[tuple[str, str]]) -> str:
    """Raw HTML for a grid of title-and-text cards, styled by ``styles.css``.

    Fragments are joined with no blank lines: Markdown ends a raw HTML block
    at the first blank line and would print the rest as text.
    """
    body = "".join(f'<div class="mmr-card"><h4>{title}</h4><p>{text}</p></div>' for title, text in items)
    return f'<div class="mmr-grid">{body}</div>'


def _render_hero(chat_page: st.Page, explorer_page: st.Page) -> None:
    """The headline, the one-paragraph pitch and the two main actions."""
    st.markdown(
        """
        <div class="mmr-hero">
          <h2>KI-CRAFT: ask your datasheets, see the figure behind every answer</h2>
          <p>KI-CRAFT reads datasheets and application notes the way an engineer does,
          text, tables and figures together, and answers your questions with the exact
          sections and figures it relied on. It is the first step of an open-source,
          AI-assisted PCB design toolkit built to work alongside KiCad.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    col1, col2, col3, _ = st.columns([1, 1, 1, 3])
    col1.page_link(chat_page, label="Ask a datasheet")
    col2.page_link(explorer_page, label="Browse the datasheets")
    col3.page_link(GITHUB_URL, label="Source on GitHub")


def _render_documents() -> None:
    """The documents ready to query, read live from the API."""
    st.markdown("## Ready to query")
    try:
        documents = list_document_summaries()
    except Exception as exc:
        st.warning(f"The document list is unavailable: {exc}")
        return
    if not documents:
        st.info("No documents are indexed yet.")
        return
    columns = st.columns(len(documents))
    for column, document in zip(columns, documents):
        column.metric(
            display_name(document["id"]),
            f"{document['image_count']} figures",
            f"{document['chunk_count']} sections, {document['table_count']} tables",
            delta_color="off",
            delta_arrow="off",
            border=True,
        )
    st.markdown("**Try asking:**\n" + "\n".join(f"- {question}" for question in _EXAMPLE_QUESTIONS))


def _render_how_it_works() -> None:
    """A short explanation of the pipeline, with the interactive diagram."""
    st.markdown("## How it works")
    st.markdown(
        "When a datasheet is added, a vision model writes a plain-text summary of each "
        "section, covering its prose, tables and figures. The summaries are what gets "
        "searched; the original text, tables and figures are stored alongside. When you "
        "ask, the most relevant sections are found by meaning, and the model answers from "
        "their **original** text, tables and figures, citing each one by label. The server "
        "then checks every citation against what the model was actually given."
    )
    # Drawn in the open: inside a collapsed expander the iframe has no width
    # when the diagram is laid out, and it opens zoomed onto an empty corner.
    st.iframe(mermaid_iframe(MERMAID_DIAGRAM), height=560)


def _render_roadmap() -> None:
    """Where the project is going, as in the README's status table."""
    st.markdown("## Roadmap")
    st.table(
        [{"Phase": phase, "Goal": goal, "Status": status} for phase, goal, status in _ROADMAP]
    )


def render(chat_page: st.Page, explorer_page: st.Page) -> None:
    """Draw the home page.

    Args:
        chat_page: The Query page, for the "Ask a datasheet" link.
        explorer_page: The Chunk Explorer page, for the "Browse" link.
    """
    _render_hero(chat_page, explorer_page)
    st.markdown("## What it does")
    st.markdown(_cards(_FEATURES), unsafe_allow_html=True)
    st.markdown("## Who it is for")
    st.markdown(_cards(_AUDIENCE), unsafe_allow_html=True)
    _render_documents()
    _render_how_it_works()
    _render_roadmap()
    st.caption(
        f"KI-CRAFT is open source under the Apache 2.0 license. "
        f"[Source]({GITHUB_URL}) · [API reference]({GITHUB_URL}/blob/main/docs/API.md) · "
        f"[Contributing]({GITHUB_URL}/blob/main/CONTRIBUTING.md)"
    )
