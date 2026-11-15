"""Retrieval pipeline entry point.

Loads the vector store, builds the retriever, and streams the answer to one
question from the default multimodal LLM.

Usage:
    python -m scripts.query
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import GENERATION_MODEL
from src.embed import load_embedding_model
from src.logger import get_logger
from src.retrieval.generate import answer_query_stream, default_llm
from src.retrieval.search import build_retriever, load_vector_store

logger = get_logger(__name__)


def main() -> None:
    """Run the interactive query loop: load store -> retrieve -> generate."""
    logger.info("Loading embedding model...")
    embedding_model = load_embedding_model()

    logger.info("Loading vector store...")
    db = load_vector_store(embedding_model)

    retriever = build_retriever(db)
    logger.info("Ready.")

    query = input("Enter your query: ")
    if not query.strip():
        logger.warning("No query provided.")
        return

    print("\n" + "-" * 5)
    for token in answer_query_stream(retriever, query, default_llm(), GENERATION_MODEL):
        print(token, end="", flush=True)
    print()


if __name__ == "__main__":
    main()
