from pathlib import Path
from typing import List

import pymupdf


def load_pdf(pdf_path: str) -> List[dict]:
    """
    Load a PDF page-by-page while preserving page provenance.
    """

    path = Path(pdf_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file: {pdf_path}")

    document = pymupdf.open(pdf_path)

    pages = []

    try:
        for page_number, page in enumerate(document, start=1):
            pages.append(
                {
                    "page_number": page_number,
                    "text": page.get_text("text"),
                    "page": page,
                }
            )
    finally:
        document.close()

    return pages