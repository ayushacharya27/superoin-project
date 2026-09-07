import os
from typing import List

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI


from src.schemas.fact_schema import DocumentMetadataPrior


# Load variables from .env
load_dotenv()


def extract_global_metadata(
    chunks: List[Document],
    max_pages: int = 3,
) -> DocumentMetadataPrior:
    """
    Extract document-level context from the first few pages.

    This is a context prior, not a fact extraction step.
    """

    if not chunks:
        raise ValueError("Cannot extract metadata from empty chunks.")

    # ---------------------------------------------------------
    # 1. Collect chunks belonging to the first N pages
    # ---------------------------------------------------------

    pages = {}

    for chunk in chunks:
        page_number = chunk.metadata.get("page_number")

        if page_number is None:
            continue

        if page_number <= max_pages:
            pages.setdefault(page_number, [])
            pages[page_number].append(chunk.page_content)

    if not pages:
        raise ValueError("No content found in the first pages.")

    # ---------------------------------------------------------
    # 2. Reconstruct the introductory text
    # ---------------------------------------------------------

    page_sections = []

    for page_number in sorted(pages):
        page_sections.append(
            f"\n--- PAGE {page_number} ---\n"
        )

        page_sections.extend(pages[page_number])

    text_sample = "\n".join(page_sections)

    # ---------------------------------------------------------
    # 3. Get API key from .env
    # ---------------------------------------------------------

    google_api_key = os.getenv("GOOGLE_API_KEY")

    if not google_api_key:
        raise ValueError(
            "GOOGLE_API_KEY is not set in the .env file."
        )

    # ---------------------------------------------------------
    # 4. Initialize Gemini
    # ---------------------------------------------------------

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.1,
        google_api_key=google_api_key,
    )

    # Tell Gemini to return our Pydantic schema
    structured_llm = llm.with_structured_output(
        DocumentMetadataPrior
    )

    # ---------------------------------------------------------
    # 5. Prompt
    # ---------------------------------------------------------

    prompt = f"""
You are analyzing the introductory pages of a document.

Your task is to identify GLOBAL DOCUMENT METADATA.

Return:

1. document_title
   The formal title of the document.

2. primary_entities
   The main companies, organizations, people, or other entities
   the document is primarily about.

3. reporting_period
   The broad reporting period covered by the document, if it is
   explicitly stated.

Examples:
- FY 2022
- FY 2023
- 2024
- Q3 2025
- As of October 2024

IMPORTANT RULES:

- Use only information supported by the supplied text.
- Do not invent information.
- Do not treat every mentioned entity as a primary entity.
- If the reporting period is unclear, return null.
- This metadata will later be supplied as context to a
  fact-extraction model.
- Do not extract individual financial or operational facts yet.

DOCUMENT:

{text_sample}
"""

    # ---------------------------------------------------------
    # 6. Call Gemini
    # ---------------------------------------------------------

    try:
        metadata = structured_llm.invoke(prompt)

        if not isinstance(metadata, DocumentMetadataPrior):
            raise TypeError(
                "Unexpected response type from metadata LLM."
            )

        return metadata

    except Exception as exc:
        print(f"[Metadata] Extraction failed: {exc}")

        # Safe fallback so one LLM failure doesn't destroy
        # the entire pipeline.
        return DocumentMetadataPrior(
            document_title="Unknown Document",
            primary_entities=[],
            reporting_period=None,
        )