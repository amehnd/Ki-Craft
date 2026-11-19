"""Citation check: which chunks and images an answer cites, and which labels it invented.

Runs after generation, on the finished answer text. The prompt
(``src/retrieval/prompt.py``) labels chunks ``C1``, ``C2``, ... and images
``IMG1``, ``IMG2``, ..., and asks the model to cite them in square brackets.
Asking does not guarantee it: a model can cite ``[IMG4]`` when only three
images were sent. This module reads the answer and sorts every label it
cites into chunks that exist, images that were sent, and labels that match
nothing, so a client can trust the first two and warn about the third.
Nothing here calls a model.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

# One bracketed group, such as "[C1]" or "[C1, IMG2]". The length cap keeps
# long bracketed prose from being scanned as a list of labels.
_BRACKETS = re.compile(r"\[([^\[\]]{1,80})\]")
# A label as the prompt writes it. Anything else in brackets ("[1]", "[see
# below]", Markdown link text) is ordinary text, not a citation.
_LABEL = re.compile(r"(IMG|C)(\d+)", re.IGNORECASE)


@dataclass(frozen=True)
class Citations:
    """The labels an answer cites, in the order it first cites them.

    Attributes:
        chunks: 1-based numbers of the cited chunks that were in the prompt.
        figures: Cited image labels that were sent, such as ``IMG1``.
        unknown: Cited labels that match no chunk or sent image, such as
            ``IMG4`` when three images were sent.
    """

    chunks: list[int]
    figures: list[str]
    unknown: list[str]


def find_citations(answer: str, chunk_count: int, figure_labels: Sequence[str]) -> Citations:
    """Sort the labels an answer cites into valid chunks, valid images and unknown labels.

    A bracketed group may hold several labels separated by commas or
    semicolons (``[C1, C3]``). Labels are matched without regard to case and
    written back in the prompt's form (``img1`` becomes ``IMG1``). Each label
    is listed once, at its first citation.

    Args:
        answer: The generated answer text.
        chunk_count: How many chunks were in the prompt; ``C1`` to
            ``C<chunk_count>`` are valid.
        figure_labels: The labels of the images sent, such as ``["IMG1"]``.

    Returns:
        The cited chunks, images and unknown labels.
    """
    sent = set(figure_labels)
    seen: set[str] = set()
    chunks: list[int] = []
    figures: list[str] = []
    unknown: list[str] = []
    for group in _BRACKETS.findall(answer):
        for token in re.split(r"[,;]", group):
            match = _LABEL.fullmatch(token.strip())
            if match is None:
                continue
            kind, number = match.group(1).upper(), int(match.group(2))
            label = f"{kind}{number}"
            if label in seen:
                continue
            seen.add(label)
            if kind == "C" and 1 <= number <= chunk_count:
                chunks.append(number)
            elif kind == "IMG" and label in sent:
                figures.append(label)
            else:
                unknown.append(label)
    return Citations(chunks=chunks, figures=figures, unknown=unknown)
