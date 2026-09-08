from typing import Literal

from pydantic import BaseModel, Field


class FactRelationship(BaseModel):
    relationship_id: str

    fact_a_id: str
    fact_b_id: str

    status: Literal[
        "CORROBORATION",
        "CONTRADICTION",
        "CONTEXTUAL_RESOLUTION",
        "UNRELATED",
        "UNCERTAIN",
    ]

    reasoning: str

    confidence: int = Field(
        ge=1,
        le=10,
    )

    resolution_method: Literal[
        "DETERMINISTIC",
        "LLM",
    ]