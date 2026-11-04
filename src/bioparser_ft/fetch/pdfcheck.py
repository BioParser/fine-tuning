"""This module makes sure that downloaded bytes are a real PDF file of the correct article."""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass

from .matching import title_recall_in_text

MIN_PDF_BYTES = 4096  # A smaller "PDF" is almost always an error page or an empty file
TITLE_RECALL_OK = 0.6  # The part of the title words that must be in the first pages

logging.getLogger("pypdf").setLevel(logging.ERROR)


def is_pdf(data: bytes) -> bool:
    return b"%PDF-" in data[:1024]


@dataclass
class PdfCheck:
    is_pdf: bool
    verified: bool  # True if the first pages contain the title or the DOI of the article
    pages: int = 0
    title_recall: float | None = None
    doi_found: bool = False
    reason: str = ""

    @property
    def usable(self) -> bool:
        return self.is_pdf and self.pages > 0


def check_pdf(data: bytes, title: str, doi: str | None, *, min_bytes: int = MIN_PDF_BYTES) -> PdfCheck:
    """Examine a downloaded file and return a PdfCheck.

    The file is "verified" when its first 3 pages contain the DOI, or 60% or more of the
    title words.
    """
    if not is_pdf(data):
        return PdfCheck(False, False, reason="not a PDF file: an HTML page or an error")
    if len(data) < min_bytes:
        return PdfCheck(False, False, reason=f"only {len(data)} bytes")
    try:
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            reader.decrypt("")  # Many publisher PDF files use an empty user password
        pages = len(reader.pages)
        text = " ".join((reader.pages[i].extract_text() or "") for i in range(min(3, pages)))
    except Exception as exc:  # A damaged or unusual PDF file: keep it, but not as verified
        return PdfCheck(True, False, reason=f"pypdf cannot read the file: {type(exc).__name__}")
    doi_found = bool(doi and doi.lower() in text.lower().replace(" ", ""))
    if len(text.strip()) < 200:
        return PdfCheck(True, doi_found, pages, None, doi_found, "no text layer: a scanned document?")
    recall = title_recall_in_text(title, text) if title else None
    verified = doi_found or (recall is not None and recall >= TITLE_RECALL_OK)
    reason = (
        ""
        if verified
        else f"title words found: {recall:.0%}"
        if recall is not None
        else "no title to compare"
    )
    return PdfCheck(True, verified, pages, recall, doi_found, reason)
