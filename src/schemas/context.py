from typing import List, Optional

from pydantic import BaseModel, Field


class ContextEvidence(BaseModel):
    evidence_id: str
    source_file: str
    page_number: int
    text: str

    element_type: Optional[str] = None
    table_ref: Optional[str] = None

    table_headers: List[str] = Field(default_factory=list)
    table_rows: List[List[str]] = Field(default_factory=list)
    table_evidence_ids: List[str] = Field(default_factory=list)


class CompactContext(BaseModel):
    context_id: str
    topic: str
    evidence: List[ContextEvidence]
    estimated_tokens: int