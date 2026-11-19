"""Example questions shown on the home page and the Query page.

Each one is answerable from the indexed datasheets and shows a different kind
of evidence: a value with its conditions, a figure, a graph, a table, a pinout.
Clicking one puts it in the Query page's question box and asks it at once:
``ask_example`` sets ``chat_query`` and ``chat_ask_now``, and the Query page
sends any question that arrives with ``chat_ask_now`` set.
"""

from __future__ import annotations

import streamlit as st

EXAMPLE_QUESTIONS: list[str] = [
    "How does the LM555 output voltage drop change with output current?",
    "Show me the LM2596 typical application circuit and its component values.",
    "How does the TLV1117 dropout voltage change with output current?",
    "What does each pin of the NE555 do?",
    "Why does AN699 recommend an analog anti-aliasing filter before the ADC?",
]


def ask_example(question: str) -> None:
    """Put an example in the question box and mark it to be sent on the next run.

    Called before the Query page draws its question box, either as a button
    callback or before ``st.switch_page``; Streamlit forbids changing a
    widget's value after the widget is drawn in the same run.
    """
    st.session_state.chat_query = question
    st.session_state.chat_ask_now = True


def render_example_buttons(key_prefix: str) -> str | None:
    """Draw the examples as a row of small buttons that wrap on narrow screens.

    Args:
        key_prefix: Makes the button keys unique to the page drawing them.

    Returns:
        The clicked question, or None when none was clicked this run.
    """
    clicked = None
    with st.container(horizontal=True, gap="small"):
        for index, question in enumerate(EXAMPLE_QUESTIONS):
            if st.button(question, key=f"{key_prefix}_example_{index}", type="secondary", width="content"):
                clicked = question
    return clicked
