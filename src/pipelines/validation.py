from typing import List

from langchain_core.documents import Document

from src.schemas.fact_schema import ExtractedFact


def _normalize_text(text: str) -> str:
    """
    Normalize whitespace and simple Markdown formatting
    to make grounding checks more robust.
    """

    return (
        text.replace("**", "")
        .replace("_", "")
        .replace("<br>", " ")
        .replace("\n", " ")
        .replace("  ", " ")
        .strip()
        .lower()
    )


def validate_fact_grounding(
    facts: List[ExtractedFact],
    chunk: Document,
) -> List[ExtractedFact]:
    """
    Verify that extracted facts are supported by the
    original document chunk.

    The LLM proposes the fact.

    Python independently verifies that the important
    evidence exists in the source.
    """

    source_text = chunk.page_content
    normalized_source = _normalize_text(source_text)

    validated_facts = []

    for fact in facts:

        quote = fact.exact_quote.strip()
        raw_value = fact.raw_value.strip()
        attribute = fact.attribute.strip()

        # -----------------------------------------------------
        # 1. Empty evidence
        # -----------------------------------------------------

        if not quote:
            fact.status = "GROUNDING_FAILURE"
            validated_facts.append(fact)
            continue

        # -----------------------------------------------------
        # 2. Exact quote match
        # -----------------------------------------------------

        normalized_quote = _normalize_text(quote)

        if normalized_quote in normalized_source:
            fact.status = "VALID"
            validated_facts.append(fact)
            continue

        # -----------------------------------------------------
        # 3. Evidence-component validation
        #
        # This is especially useful for Markdown tables where
        # the LLM may return only part of a row.
        # -----------------------------------------------------

        normalized_value = _normalize_text(raw_value)
        normalized_attribute = _normalize_text(attribute)

        attribute_present = (
            normalized_attribute in normalized_source
        )

        value_present = (
            normalized_value in normalized_source
        )

        # -----------------------------------------------------
        # 4. Validate based on attribute + value
        # -----------------------------------------------------

        if attribute_present and value_present:
            fact.status = "VALID"

        else:
            fact.status = "GROUNDING_FAILURE"

        validated_facts.append(fact)

    return validated_facts