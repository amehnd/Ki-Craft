"""Ingest PDFs into the Chroma store, one document at a time.

Each PDF goes through :func:`src.ingestion.pipeline.run_complete_ingestion_pipeline`:
the chunk export goes to ``<exports>/<pdf name>.json``, figures to
``<images>``, and the records to ``<store>``. A PDF whose export already exists
is skipped, so an interrupted batch can be re-run with the same command. A PDF
that fails is logged and the batch moves on; nothing of it is written.

Usage:
    python -m scripts.ingest data/ne555.pdf
    python -m scripts.ingest data/ --backend local
    python -m scripts.ingest data/ne555.pdf --backend local --store /tmp/store --images /tmp/images --exports /tmp/json

The summary backend's tunables (timeout, retries, max tokens, image limits)
are environment variables; see the enhancement settings in config/settings.py.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import (
    CHROMA_PERSIST_DIR,
    ENHANCEMENT_BACKEND,
    ENHANCEMENT_BACKENDS,
    IMAGES_DIR,
    JSON_EXPORT_DIR,
)
from src.ingestion.enrich import check_enhancement_backend
from src.ingestion.pipeline import run_complete_ingestion_pipeline
from src.logger import get_logger

logger = get_logger(__name__)


def _pdfs(paths: list[str]) -> list[Path]:
    """Expand folders to the PDFs inside them, sorted by name."""
    found: list[Path] = []
    for raw in paths:
        path = Path(raw)
        found.extend(sorted(path.glob("*.pdf")) if path.is_dir() else [path])
    return found


def main() -> int:
    """Parse arguments, ingest each PDF, and return the process exit code."""
    parser = argparse.ArgumentParser(description="Ingest PDFs into the Chroma store.")
    parser.add_argument("paths", nargs="+", help="PDF files, or folders of PDFs")
    parser.add_argument("--backend", choices=sorted(ENHANCEMENT_BACKENDS), default=ENHANCEMENT_BACKEND,
                        help="summary model backend (default: ENHANCEMENT_BACKEND, %(default)s)")
    parser.add_argument("--store", default=CHROMA_PERSIST_DIR, help="Chroma folder (default: %(default)s)")
    parser.add_argument("--images", default=IMAGES_DIR, help="figure files folder (default: %(default)s)")
    parser.add_argument("--exports", default=JSON_EXPORT_DIR, help="chunk exports folder (default: %(default)s)")
    args = parser.parse_args()

    check_enhancement_backend(args.backend)
    pdfs = _pdfs(args.paths)
    logger.info("Ingesting %d PDFs with the %s backend into %s", len(pdfs), args.backend, args.store)

    done, skipped, failed = [], [], []
    for n, pdf in enumerate(pdfs, start=1):
        logger.info("[%d/%d] %s", n, len(pdfs), pdf)
        t0 = time.monotonic()
        try:
            run_complete_ingestion_pipeline(pdf, args.store, args.images, args.exports, args.backend)
        except FileExistsError as exc:
            logger.info("  Skipped: %s", exc)
            skipped.append(pdf)
            continue
        except Exception:
            logger.exception("  Failed: %s", pdf)
            failed.append(pdf)
            continue
        logger.info("  Done in %.0fs", time.monotonic() - t0)
        done.append(pdf)

    logger.info("Ingested %d, skipped %d, failed %d", len(done), len(skipped), len(failed))
    for pdf in failed:
        logger.info("  failed: %s", pdf)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
