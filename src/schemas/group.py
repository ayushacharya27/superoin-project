from typing import List

from pydantic import BaseModel, Field

from src.schemas.evidence import EvidenceUnit


class EvidenceGroup(BaseModel):
    """
    A semantically coherent collection of evidence units.

    Embedding similarity creates the group.
    It does NOT imply that the evidence units are factually equivalent.
    """

    group_id: str

    topic_label: str

    evidence: List[EvidenceUnit] = Field(default_factory=list)

    representative_evidence_ids: List[str] = Field(
        default_factory=list
    )