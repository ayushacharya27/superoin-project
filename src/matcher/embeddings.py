from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

from config import settings


@dataclass
class CandidatePair:
    fact_a: object
    fact_b: object
    similarity: float


class FactEmbeddingIndex:
    """
    Lightweight in-memory fact embedding index.

    Embeddings represent semantic identity:
        entity + attribute + time + scope + qualifier

    The numerical value is intentionally excluded so that
    different values for the same metric still become candidates.
    """

    def __init__(self) -> None:
        print(f"[Embeddings] Loading {settings.EMBEDDING_MODEL}...")
        self.model = SentenceTransformer(settings.EMBEDDING_MODEL)

    @staticmethod
    def _document_of(fact: object) -> str:
        """
        Extract source document from the available provenance fields.
        """
        if getattr(fact, "source_file", None):
            return str(getattr(fact, "source_file"))

        if getattr(fact, "table_ref", None):
            return str(fact.table_ref).split(":", 1)[0]

        evidence_id = getattr(fact, "evidence_id", "") or ""
        if evidence_id:
            return str(evidence_id).split(":", 1)[0]

        return "unknown"

    @staticmethod
    def _identity_text(fact: object) -> str:
        """
        Build semantic identity without numerical value.
        """
        entity = getattr(fact, "entity", "") or ""
        attribute = getattr(fact, "attribute", "") or ""
        time = getattr(fact, "time", "") or ""
        scope = getattr(fact, "scope", "") or ""
        qualifier = getattr(fact, "qualifier", "") or ""

        return (
            f"Entity: {entity} | "
            f"Attribute: {attribute} | "
            f"Time: {time} | "
            f"Scope: {scope} | "
            f"Qualifier: {qualifier}"
        )

    def encode(self, facts: List[object]) -> np.ndarray:
        """
        Encode facts in batches and return a normalized matrix.
        """
        if not facts:
            return np.empty((0, 0))

        texts = [self._identity_text(fact) for fact in facts]
        print(f"[Embeddings] Encoding {len(texts)} facts...")

        embeddings = self.model.encode(
            texts,
            batch_size=settings.EMBEDDING_BATCH_SIZE,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        return embeddings

    def find_candidates(
        self,
        facts: List[object],
        similarity_threshold: float = 0.75,
    ) -> List[CandidatePair]:
        """
        Find semantically similar facts across different documents.
        """
        valid_facts = [
            fact
            for fact in facts
            if getattr(fact, "status", "VALID") == "VALID"
        ]

        if len(valid_facts) < 2:
            return []

        embeddings = self.encode(valid_facts)
        similarity_matrix = np.dot(embeddings, embeddings.T)
        num_facts = len(valid_facts)
        candidates: List[CandidatePair] = []

        for i in range(num_facts):
            doc_a = self._document_of(valid_facts[i])

            for j in range(i + 1, num_facts):
                doc_b = self._document_of(valid_facts[j])

                # Never match a document against itself.
                if doc_a == doc_b:
                    continue

                sim = float(similarity_matrix[i, j])
                if sim >= similarity_threshold:
                    candidates.append(
                        CandidatePair(
                            fact_a=valid_facts[i],
                            fact_b=valid_facts[j],
                            similarity=sim,
                        )
                    )

        candidates.sort(
            key=lambda pair: pair.similarity,
            reverse=True,
        )

        print(f"[Embeddings] Found {len(candidates)} candidate pairs.")
        return candidates