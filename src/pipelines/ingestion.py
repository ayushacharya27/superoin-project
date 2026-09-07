import os
from typing import List

import pymupdf4llm
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter


HEADERS_TO_SPLIT_ON = [
    ("#", "Header_1"),
    ("##", "Header_2"),
    ("###", "Header_3"),
]


def ingest_pdf(file_path: str) -> List[Document]:
    """
    Convert a PDF into page-aware Markdown chunks.

    Each chunk contains:
        - extracted Markdown content
        - source filename
        - PDF page number
        - chunk index within the page

    LLM-based metadata extraction is intentionally handled
    separately from this function.
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"PDF not found: {file_path}"
        )

    if not file_path.lower().endswith(".pdf"):
        raise ValueError(
            f"Expected a PDF file, got: {file_path}"
        )

    print(f"[Ingestion] Processing: {file_path}")

    # ---------------------------------------------------------
    # 1. Convert PDF → Markdown
    # ---------------------------------------------------------

    pages = pymupdf4llm.to_markdown(
        file_path,
        page_chunks=True,
    )

    if not pages:
        raise ValueError(
            f"No content could be extracted from: {file_path}"
        )

    file_name = os.path.basename(file_path)

    # ---------------------------------------------------------
    # 2. Configure Markdown-aware chunking
    # ---------------------------------------------------------

    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT_ON,
        strip_headers=False,
    )

    chunks: List[Document] = []

    # ---------------------------------------------------------
    # 3. Process each page independently
    # ---------------------------------------------------------

    for page_index, page in enumerate(pages, start=1):

        page_text = page.get("text", "")

        if not page_text.strip():
            continue

        page_metadata = page.get("metadata", {})

        # pymupdf4llm normally provides page_number.
        # Fall back to our page_index if it isn't available.
        page_number = page_metadata.get(
            "page_number",
            page_index,
        )

        # -----------------------------------------------------
        # 4. Split page by Markdown headers
        # -----------------------------------------------------

        page_chunks = splitter.split_text(page_text)

        for chunk_index, chunk in enumerate(page_chunks):

            chunk.metadata.update(
                {
                    "source_file": file_name,
                    "page_number": page_number,
                    "chunk_index": chunk_index,
                }
            )

            chunks.append(chunk)

    if not chunks:
        raise ValueError(
            f"PDF contained no usable text: {file_path}"
        )

    print(
        f"[Ingestion] Extracted {len(pages)} pages "
        f"into {len(chunks)} chunks."
    )

    return chunks