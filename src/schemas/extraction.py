from typing import List, Optional

from pydantic import BaseModel, Field


class ExtractedFact(BaseModel):
    entity: str = ""
    attribute: str = ""
    raw_value: str = ""
    unit: Optional[str] = None
    normalized_value: Optional[float] = None
    time: Optional[str] = None
    scope: Optional[str] = None
    qualifier: Optional[str] = None

    # Prose grounding
    exact_quote: str = ""
    evidence_id: str = ""

    # Table grounding
    table_ref: Optional[str] = None
    row_index: Optional[int] = None
    column_index: Optional[int] = None
    column_name: Optional[str] = None

    extraction_confidence: int = Field(
        default=5,
        ge=1,
        le=10,
    )
    status: str = "VALID"


class FactExtractionBatch(BaseModel):
    facts: List[ExtractedFact] = Field(default_factory=list)