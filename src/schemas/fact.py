from typing import Literal, Optional

from pydantic import BaseModel, Field


class CanonicalFact(BaseModel):
    """
    A semantic fact produced by the LLM from grounded evidence.
    """

    fact_id: str

    entity: str
    attribute: str

    raw_value: str

    unit: Optional[str] = None
    normalized_value: Optional[float] = None

    time: Optional[str] = None
    scope: Optional[str] = None
    qualifier: Optional[str] = None

    exact_quote: str

    source_file: str
    page_number: int
    evidence_id: str

    extraction_confidence: int = Field(
        ge=1,
        le=10,
    )

    status: Literal[
        "VALID",
        "GROUNDING_FAILURE",
        "UNCERTAIN",
    ] = "VALID"