from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class EvidenceUnit(BaseModel):
    """
    A small, source-grounded piece of information extracted
    deterministically from a PDF.
    """

    evidence_id: str

    source_file: str
    page_number: int

    evidence_type: Literal[
        "TABLE_ROW",
        "TABLE_CELL",
        "PROSE",
        "HEADING",
    ]

    text: str

    # Structural information
    section_heading: Optional[str] = None
    table_caption: Optional[str] = None
    row_label: Optional[str] = None
    column_label: Optional[str] = None

    # Cheap deterministic hints
    number_hints: List[str] = Field(default_factory=list)
    unit_hint: Optional[str] = None
    time_hint: Optional[str] = None
    entity_hint: Optional[str] = None

    # Embedding is attached later.
    embedding: Optional[List[float]] = None