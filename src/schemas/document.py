from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class TableCell(BaseModel):
    text: str
    row_index: int
    column_index: int


class StructuredTable(BaseModel):
    table_id: str
    page_number: int
    caption: Optional[str] = None
    unit: Optional[str] = None
    headers: List[str] = Field(default_factory=list)
    rows: List[List[str]] = Field(default_factory=list)
    evidence_ids: List[str] = Field(default_factory=list)
    bbox: Optional[List[float]] = None


class DocumentElement(BaseModel):
    element_id: str
    source_file: str
    page_number: int

    element_type: Literal[
        "HEADING",
        "PROSE",
        "TABLE",
    ]

    text: str = ""
    heading: Optional[str] = None

    table: Optional[StructuredTable] = None

    evidence_ids: List[str] = Field(
        default_factory=list
    )

    top: float = 0.0