"""PDF extraction step.

:func:`partition_document` uses Unstructured's ``partition_pdf`` to turn a PDF
into a list of ``Element`` objects (titles, text, tables, images), with each
figure attached as base64. :func:`remove_page_decoration` then drops what is
printed on every page rather than written about the content: running headers,
footers, page numbers and repeated logos. :func:`document_context` keeps the
running header once, as a label for the whole document.
"""

from __future__ import annotations

import logging
import re
from collections import Counter, defaultdict
from pathlib import Path

from unstructured.partition.pdf import partition_pdf

from config.settings import (
    PDF_DECORATION_MIN_PAGES,
    PDF_LOGO_WORDS,
    PDF_EXTRACT_IMAGE_BLOCK_TYPES,
    PDF_INFER_TABLE_STRUCTURE,
    PDF_STRATEGY,
)
from src.logger import get_logger

logger = get_logger(__name__)


def partition_document(file_path: str | Path) -> list:
    """Extract structured content elements from a PDF.

    Args:
        file_path: Path to the PDF to process.

    Returns:
        A list of Unstructured ``Element`` objects (headings, narrative text,
        tables and extracted images).
    """
    pdf_path = Path(file_path)
    logger.info(
        "Extracting PDF elements: %s (strategy=%s, extract_images=%s, infer_tables=%s)",
        pdf_path.name,
        PDF_STRATEGY,
        PDF_EXTRACT_IMAGE_BLOCK_TYPES,
        PDF_INFER_TABLE_STRUCTURE,
    )

    # ``unstructured`` is verbose; keep its log noise out of our logs.
    logging.getLogger("unstructured").setLevel(logging.WARNING)

    return partition_pdf(
        filename=str(pdf_path),
        strategy=PDF_STRATEGY,
        extract_image_block_types=PDF_EXTRACT_IMAGE_BLOCK_TYPES,
        # Put each figure on its element as metadata.image_base64, which
        # enrich.separate_content_types reads. Without it unstructured writes
        # the figures to a ./figures folder and the elements carry none.
        extract_image_block_to_payload=True,
        infer_table_structure=PDF_INFER_TABLE_STRUCTURE,
        extract_images_in_pdf=False,
        chunking_strategy=None,
    )


# Element types Unstructured gives to text printed on every page, not content.
_PAGE_DECORATION_TYPES = {"Header", "Footer", "PageNumber"}
# A logo's OCR text is a few words ("texas instruments www com"); a graph's
# axis labels and title are more, so a repeated graph is never taken for a logo.
_LOGO_MAX_WORDS = 6


def _image_text_key(element) -> str:
    """The words (3+ letters) an image's OCR text contains, sorted, lowercase.

    OCR of the same logo varies in stray characters ("I} TEXAS INSTRUMENTS"
    vs "TEXAS INSTRUMENTS"), so the key ignores everything but real words.
    """
    return " ".join(sorted(set(re.findall(r"[a-z]{3,}", (element.text or "").lower()))))


def remove_page_decoration(
    elements: list,
    min_pages: int = PDF_DECORATION_MIN_PAGES,
    logo_words: tuple[str, ...] = PDF_LOGO_WORDS,
) -> list:
    """Drop running headers, footers, page numbers and logos.

    Header, footer and page-number elements are dropped by type. An image is
    a logo when its OCR text is at most ``_LOGO_MAX_WORDS`` words and either
    the same words appear on ``min_pages`` or more pages, or one of them is in
    ``logo_words`` (a vendor name). The word limit keeps a real figure whose
    caption names the vendor. Images with no OCR words are kept.

    Args:
        elements: Elements from :func:`partition_document`, in page order.
        min_pages: On how many pages an image must repeat to count as a logo.
        logo_words: Vendor-name words that mark a short image as a logo.

    Returns:
        The elements that describe the document's content, in the same order.
    """
    pages_by_key: dict[str, set] = defaultdict(set)
    for element in elements:
        if type(element).__name__ == "Image" and _image_text_key(element):
            pages_by_key[_image_text_key(element)].add(element.metadata.page_number)
    logos = {
        key for key, pages in pages_by_key.items()
        if len(key.split()) <= _LOGO_MAX_WORDS
        and (len(pages) >= min_pages or set(key.split()) & set(logo_words))
    }

    kept = [
        element
        for element in elements
        if type(element).__name__ not in _PAGE_DECORATION_TYPES
        and not (type(element).__name__ == "Image" and _image_text_key(element) in logos)
    ]
    logger.info("Removed %d page-decoration elements of %d", len(elements) - len(kept), len(elements))
    return kept


def document_context(elements: list, source: str) -> str:
    """Name the document for the summary prompt: its source and running header.

    The running header (the most common ``Header`` text) is the publisher's own
    label for the document, for example the part numbers and document number.

    Args:
        elements: Elements from :func:`partition_document`, before
            :func:`remove_page_decoration`.
        source: The document's name in the store.

    Returns:
        ``"<source>: <header>"``, or just ``source`` when there is no header.
    """
    headers = Counter(e.text.strip() for e in elements if type(e).__name__ == "Header" and e.text.strip())
    return f"{source}: {headers.most_common(1)[0][0]}" if headers else source
