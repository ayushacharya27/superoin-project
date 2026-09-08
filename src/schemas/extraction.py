from typing import List, Optional

from pydantic import BaseModel, Field


class ExtractedFact(BaseModel):
    entity: Optional[str] = None
    attribute: Optional[str] = None
    raw_value: Optional[str] = None
    unit: Optional[str] = None
    normalized_value: Optional[float] = None
    time: Optional[str] = None
    scope: Optional[str] = None
    qualifier: Optional[str] = None

    exact_quote: Optional[str] = None
    evidence_id: Optional[str] = None

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