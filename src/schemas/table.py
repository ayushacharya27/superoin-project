from typing import List, Optional

from pydantic import BaseModel, Field


class ReconstructedTableRow(BaseModel):
    evidence_ids: List[str] = Field(default_factory=list)
    values: List[str] = Field(default_factory=list)
    row_label: Optional[str] = None
    source_file: str
    page_number: int


class TableContext(BaseModel):
    table_id: str
    source_file: str
    start_page: int
    end_page: int
    caption: Optional[str] = None
    unit: Optional[str] = None
    headers: List[str] = Field(default_factory=list)
    rows: List[ReconstructedTableRow] = Field(default_factory=list)
    deterministic: bool = False
    confidence: float = 0.0