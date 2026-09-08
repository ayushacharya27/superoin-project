from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from config import settings
from src.schemas.evidence import EvidenceUnit


class EvidenceEmbedder:
    """
    Generates semantic embeddings for evidence units.

    Embeddings are used for routing/grouping evidence, not for
    deciding factual equivalence.
    """

    def __init__(self):
        self.model = SentenceTransformer(settings.EMBEDDING_MODEL)

    def embed(
        self,
        evidence: List[EvidenceUnit],
    ) -> List[EvidenceUnit]:
        if not evidence:
            return []

        texts = [self._embedding_text(item) for item in evidence]

        vectors = self.model.encode(
            texts,
            batch_size=settings.EMBEDDING_BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        for item, vector in zip(evidence, vectors):
            item.embedding = vector.tolist()

        return evidence

    @staticmethod
    def _embedding_text(item: EvidenceUnit) -> str:
        """
        Include useful structural context so that identical wording
        in different sections is less likely to become semantically
        indistinguishable.
        """
        parts = []

        if item.section_heading:
            parts.append(f"Section: {item.section_heading}")

        if item.table_caption:
            parts.append(f"Table: {item.table_caption}")

        if item.row_label:
            parts.append(f"Row: {item.row_label}")

        if item.column_label:
            parts.append(f"Column: {item.column_label}")

        parts.append(item.text)

        return " | ".join(parts)