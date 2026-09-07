import os
from typing import List

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI

from src.schemas.fact_schema import (
    DocumentMetadataPrior,
    ExtractedFact,
    ExtractedFactList,
)


# Load environment variables from .env
load_dotenv()


def extract_facts_from_chunk(
    chunk: Document,
    metadata_prior: DocumentMetadataPrior,
) -> List[ExtractedFact]:
    """
    Extract meaningful semantic and numerical facts from
    a single document chunk.

    The LLM proposes facts.

    Grounding validation and deterministic normalization
    will be performed separately.
    """

    # ---------------------------------------------------------
    # 1. Validate input
    # ---------------------------------------------------------

    if not chunk.page_content.strip():
        return []

    # ---------------------------------------------------------
    # 2. Get Google API key
    # ---------------------------------------------------------

    google_api_key = os.getenv("GOOGLE_API_KEY")

    if not google_api_key:
        raise ValueError(
            "GOOGLE_API_KEY is not set in the .env file."
        )

    # ---------------------------------------------------------
    # 3. Initialize Gemini
    # ---------------------------------------------------------

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.1,
        google_api_key=google_api_key,
    )

    # Gemini expects a concrete Pydantic schema here.
    #
    # Do NOT use:
    #
    #     List[ExtractedFact]
    #
    # because this LangChain Gemini version does not accept
    # typing.List as a structured-output schema.

    structured_llm = llm.with_structured_output(
        ExtractedFactList
    )

    # ---------------------------------------------------------
    # 4. Extract deterministic provenance from chunk
    # ---------------------------------------------------------

    source_file = chunk.metadata.get("source_file")

    page_number = chunk.metadata.get("page_number")

    chunk_index = chunk.metadata.get("chunk_index")

    # ---------------------------------------------------------
    # 5. Prepare global document context
    # ---------------------------------------------------------

    if metadata_prior.primary_entities:
        primary_entities = ", ".join(
            metadata_prior.primary_entities
        )
    else:
        primary_entities = "Unknown"

    reporting_period = (
        metadata_prior.reporting_period
        or "Not explicitly known"
    )

    # ---------------------------------------------------------
    # 6. Build extraction prompt
    # ---------------------------------------------------------

    prompt = f"""
You are a fact extraction system for a
cross-document Fact Knowledge Layer.

Your task is to extract MEANINGFUL, COMPARABLE FACTS
from the supplied document chunk.

The extracted facts will later be compared against
facts from other documents.

=========================================================
GLOBAL DOCUMENT CONTEXT
=========================================================

Document title:
{metadata_prior.document_title}

Primary entities:
{primary_entities}

Broad reporting period:
{reporting_period}


=========================================================
WHAT COUNTS AS A FACT?
=========================================================

A fact should contain meaningful information about:

- an entity
- an attribute
- a value
- a measurement
- an event
- a relationship
- or a meaningful state


Examples of useful facts:

- Revenue was $1.2 billion.
- The company had 25,000 employees.
- The company operates 20 warehouses.
- EBITDA margin was 12%.
- The company acquired XYZ Ltd.
- The CEO is John Smith.
- The company operates in India.
- Revenue increased by 15%.
- The company had 500 vehicles.


=========================================================
EXTRACTION RULES
=========================================================

=========================================================
WHAT COUNTS AS A SUBSTANTIVE FACT?
=========================================================

A substantive fact describes the underlying subject matter
of the document.

Useful facts may describe:

- financial performance
- financial position
- operating metrics
- business activities
- products or services
- customers
- employees or workforce
- assets or liabilities
- geographic presence
- market position
- ownership
- management
- acquisitions or other major events
- production or operational capacity
- scientific or technical measurements
- other meaningful domain-specific characteristics


=========================================================
WHAT SHOULD NOT BE EXTRACTED?
=========================================================

Do NOT extract information whose primary purpose is to
describe the document, filing, formatting, or legal procedure.

Examples:

- document title
- document type
- publication date
- filing date
- prospectus date
- section numbers
- clause numbers
- page numbers
- table numbers
- legal references
- registration numbers
- phone numbers
- addresses unless substantively relevant
- document formatting
- boilerplate instructions
- offer structure
- IPO/book-building terminology
- administrative information
- OCR artifacts


IMPORTANT:

A piece of text should NOT become a fact merely because
it has a number or can be represented as entity + attribute
+ value.

Ask:

"Does this describe something meaningful about the
underlying entity or subject matter?"

If NO, do not extract it.


Examples:

"Section 32 of the Companies Act"
→ NOT a fact.

"Dated May 14, 2022"
→ NOT a substantive fact.

"100% Book Built Offer"
→ NOT a substantive company fact.

"Revenue was ₹7,742 million"
→ SUBSTANTIVE FACT.

"The company had 21,000 employees"
→ SUBSTANTIVE FACT.

"The company operates 20 warehouses"
→ SUBSTANTIVE FACT.

"The company acquired XYZ"
→ SUBSTANTIVE FACT.

"The CEO is John Smith"
→ SUBSTANTIVE FACT.
4. Preserve the original value in `raw_value`.

The raw value must represent how the value appears
in the source.

For example:

Source:
"$1.2 billion"

Use:

raw_value = "$1.2 billion"
unit = "USD"


Source:
"25,000 employees"

Use:

raw_value = "25,000"
unit = "employees"


Source:
"12%"

Use:

raw_value = "12%"
unit = "percentage"


5. Extract explicit time context whenever present.

Examples:

FY2022
FY 2023
Q3 2024
March 2025
As of December 31, 2024

IMPORTANT:

Do NOT automatically assign the global reporting period
to a fact unless the chunk itself supports that period.

The global reporting period is only a context prior.


6. Extract scope whenever explicitly stated.

Examples:

India
Global
North America
Enterprise customers
Retail segment
Subsidiary operations


7. Preserve important qualifiers.

Examples:

approximately
approximately 500
estimated
projected
expected
unaudited
more than
less than
at least


8. `exact_quote` MUST be copied verbatim from the
provided document chunk.

The quote must directly support the extracted fact.

Do NOT invent or paraphrase the quote.


9. Do not invent facts.

Every extracted fact must be supported by the
provided document chunk.


10. If the text is ambiguous or insufficient to establish
a meaningful fact, DO NOT extract the fact.


11. Confidence score:

Use a score from 1 to 10.

10 = explicitly and unambiguously stated
8-9 = very clearly supported
6-7 = reasonably supported
4-5 = somewhat ambiguous
1-3 = very weak support

Prefer NOT extracting a fact over creating a
low-confidence unsupported fact.


12. Do not perform complicated numerical calculations.

Preserve the source value.

Deterministic normalization will be performed
later by Python.


=========================================================
SOURCE INFORMATION
=========================================================

File:
{source_file}

Page:
{page_number}

Chunk:
{chunk_index}


=========================================================
DOCUMENT CHUNK
=========================================================

{chunk.page_content}
"""

    # ---------------------------------------------------------
    # 7. Call Gemini
    # ---------------------------------------------------------

    try:
        result = structured_llm.invoke(prompt)

        if not isinstance(result, ExtractedFactList):
            raise TypeError(
                "Unexpected response type from fact extraction LLM."
            )

        facts = result.facts

        # -----------------------------------------------------
        # 8. Attach provenance
        # -----------------------------------------------------
        #
        # These values come from our application, NOT the LLM.
        # This prevents the model from hallucinating source
        # locations.

        for fact in facts:
            fact.source_file = source_file
            fact.page_number = page_number
            fact.chunk_index = chunk_index

        return facts

    # ---------------------------------------------------------
    # 9. Handle extraction failure
    # ---------------------------------------------------------

    except Exception as exc:

        print(
            f"[Extraction] Failed on "
            f"{source_file}, page {page_number}: {exc}"
        )

        # Returning an empty list means this particular chunk
        # failed without crashing the entire document pipeline.
        return []