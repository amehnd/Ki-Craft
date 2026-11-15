"""Tests for retriever configuration in ``src/retrieval/search.py`` and the
search type setting in ``config/settings.py`` (no vector store is opened)."""

from __future__ import annotations

import pytest

import config.settings as settings
import src.retrieval.search as search


class _RecordingStore:
    """Stands in for Chroma and records what ``as_retriever`` was given."""

    def as_retriever(self, search_type: str, search_kwargs: dict):
        """Mimic ``Chroma.as_retriever`` by returning its arguments."""
        return {"search_type": search_type, "search_kwargs": search_kwargs}


def test_similarity_sends_only_k():
    assert search.search_kwargs("similarity", 5) == {"k": 5}


def test_mmr_sends_pool_and_diversity():
    assert search.search_kwargs("mmr", 5) == {
        "k": 5,
        "fetch_k": settings.RETRIEVAL_FETCH_K,
        "lambda_mult": settings.RETRIEVAL_LAMBDA_MULT,
    }


def test_mmr_pool_is_never_smaller_than_k():
    k = settings.RETRIEVAL_FETCH_K + 5
    assert search.search_kwargs("mmr", k)["fetch_k"] == k


def test_score_threshold_sends_the_threshold():
    assert search.search_kwargs("similarity_score_threshold", 5) == {
        "k": 5,
        "score_threshold": settings.RETRIEVAL_SCORE_THRESHOLD,
    }


def test_unknown_search_type_is_rejected():
    with pytest.raises(ValueError, match="Unknown search type 'hybrid'"):
        search.search_kwargs("hybrid", 5)


def test_every_allowed_search_type_has_kwargs():
    for search_type in settings.RETRIEVAL_SEARCH_TYPES:
        assert search.search_kwargs(search_type, 3)["k"] == 3


@pytest.mark.parametrize("search_type", ["similarity", "mmr", "similarity_score_threshold"])
def test_build_retriever_uses_the_configured_search_type(monkeypatch, search_type):
    """A per-request k keeps the configured type, so no MMR-only argument leaks."""
    monkeypatch.setattr(search, "RETRIEVAL_SEARCH_TYPE", search_type)
    built = search.build_retriever(_RecordingStore(), 3)
    assert built == {
        "search_type": search_type,
        "search_kwargs": search.search_kwargs(search_type, 3),
    }


def test_env_choice_reads_an_allowed_value(monkeypatch):
    monkeypatch.setenv("RETRIEVAL_SEARCH_TYPE", "similarity")
    assert settings._env_choice("RETRIEVAL_SEARCH_TYPE", "mmr", ("similarity", "mmr")) == "similarity"


def test_env_choice_falls_back_to_the_default(monkeypatch):
    monkeypatch.delenv("RETRIEVAL_SEARCH_TYPE", raising=False)
    assert settings._env_choice("RETRIEVAL_SEARCH_TYPE", "mmr", ("similarity", "mmr")) == "mmr"


def test_env_choice_names_the_allowed_values(monkeypatch):
    monkeypatch.setenv("RETRIEVAL_SEARCH_TYPE", "MMR")
    with pytest.raises(ValueError, match="'MMR' is not one of: similarity, mmr"):
        settings._env_choice("RETRIEVAL_SEARCH_TYPE", "mmr", ("similarity", "mmr"))
