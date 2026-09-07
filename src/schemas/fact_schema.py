from typing import List, Literal, Optional, Union

from pydantic import BaseModel, Field


class DocumentMetadataPrior(BaseModel):
    """
    Global context extracted from the introductory pages
    of a document.

    This is a context prior for downstream fact extraction.
    """

    document_title: str

    primary_entities: List[str]

    reporting_period: Optional[str] = None


class ExtractedFact(BaseModel):
    """
    A single semantic or numerical fact extracted from a
    document chunk.
    """

    entity: str = Field(
        description="The primary subject of the fact"
    )

    attribute: str = Field(
        description=(
            "The property, measurement, relationship, "
            "or event being described"
        )
    )

    raw_value: str = Field(
        description=(
            "The value exactly as expressed in the source"
        )
    )

    unit: Optional[str] = Field(
        default=None,
        description=(
            "Unit such as USD, INR, people, percentage, "
            "units, etc."
        )
    )

    time: Optional[str] = Field(
        default=None,
        description=(
            "Explicit time context such as FY2025, "
            "Q3 2024, March 2025, or As of December 2024"
        )
    )

    scope: Optional[str] = Field(
        default=None,
        description=(
            "Scope such as global, India, subsidiary, "
            "product line, customer segment, etc."
        )
    )

    qualifier: Optional[str] = Field(
        default=None,
        description=(
            "Qualifier such as approximately, estimated, "
            "projected, expected, more than, etc."
        )
    )

    exact_quote: str = Field(
        description=(
            "Verbatim source text that directly supports "
            "this fact"
        )
    )

    confidence_score: int = Field(
        ge=1,
        le=10,
        description=(
            "Confidence that the supplied text clearly "
            "supports this extracted fact"
        )
    )

    normalized_value: Optional[Union[str, float]] = Field(
        default=None,
        description=(
            "Deterministically normalized value. This is "
            "normally populated after LLM extraction."
        )
    )

    source_file: Optional[str] = None

    page_number: Optional[int] = None

    chunk_index: Optional[int] = None

    status: Literal[
        "VALID",
        "GROUNDING_FAILURE",
    ] = "VALID"


class ExtractedFactList(BaseModel):
    """
    Wrapper used for structured LLM output.

    Gemini/LangChain expects a concrete Pydantic model
    rather than typing.List[ExtractedFact].
    """

    facts: List[ExtractedFact]


class FactRelationship(BaseModel):
    """
    Relationship between two extracted facts.
    """

    status: Literal[
        "CORROBORATION",
        "CONTRADICTION",
        "CONTEXTUAL_RESOLUTION",
        "UNRELATED",
        "UNCERTAIN",
    ]

    reasoning: str = Field(
        description=(
            "Brief explanation based only on the supplied "
            "facts and their evidence"
        )
    )

    confidence_score: int = Field(
        ge=1,
        le=10,
        description=(
            "Confidence in the relationship classification"
        )
    )