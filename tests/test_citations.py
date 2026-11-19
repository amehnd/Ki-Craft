"""Tests for the citation check in ``src/retrieval/citations.py`` (no network)."""

from src.retrieval.citations import Citations, find_citations


def test_valid_chunks_and_figures_in_first_cited_order():
    answer = "Vref is 1.25 V [C3]. The pinout [IMG2] shows ADJ [C1, IMG1]."
    assert find_citations(answer, 3, ["IMG1", "IMG2"]) == Citations(
        chunks=[3, 1], figures=["IMG2", "IMG1"], unknown=[]
    )


def test_labels_that_point_at_nothing_are_unknown():
    """An image that was not sent, and a chunk past the prompt's last one."""
    answer = "See [IMG3] and [C6]; also [C0]."
    assert find_citations(answer, 5, ["IMG1", "IMG2"]) == Citations(
        chunks=[], figures=[], unknown=["IMG3", "C6", "C0"]
    )


def test_each_label_is_listed_once():
    assert find_citations("[C1] [C1] [c1] [IMG1][img1]", 2, ["IMG1"]) == Citations(
        chunks=[1], figures=["IMG1"], unknown=[]
    )


def test_ordinary_bracketed_text_is_not_a_citation():
    """Reference numbers, link text and the datasheet's own figure names are ignored."""
    answer = "As in [1], see [the table](https://x) and [Figure 3] or [F1]; [C1 and C2]."
    assert find_citations(answer, 5, ["IMG1"]) == Citations(chunks=[], figures=[], unknown=[])


def test_semicolons_and_spaces_separate_labels_in_one_group():
    assert find_citations("[ C2 ; IMG1 ]", 2, ["IMG1"]) == Citations(
        chunks=[2], figures=["IMG1"], unknown=[]
    )


def test_no_figures_sent_makes_every_image_label_unknown():
    """A text-only model was sent no images, so any image citation is invented."""
    assert find_citations("[IMG1] [C1]", 1, []).unknown == ["IMG1"]
