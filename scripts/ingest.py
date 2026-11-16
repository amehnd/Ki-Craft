"""Ingest one PDF into the Chroma store.

Runs :func:`src.ingestion.pipeline.run_complete_ingestion_pipeline`: the
chunk export goes to ``json/<pdf name>.json``, figures to ``dbv2/images``,
and the records to ``dbv2/chroma_db``.

Usage:
    python -m scripts.ingest path/to/document.pdf
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ingestion.pipeline import run_complete_ingestion_pipeline

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python -m scripts.ingest path/to/document.pdf")
    run_complete_ingestion_pipeline(sys.argv[1])
