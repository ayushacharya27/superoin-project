import re
from typing import Any, Dict, List, Optional, Set, Tuple

from src.matcher.embeddings import CandidatePair, FactEmbeddingIndex


class FactMatcher:
    def __init__(self, similarity_threshold: float = 0.75) -> None:
        self.similarity_threshold = similarity_threshold
        self.embedding_index = FactEmbeddingIndex()

    @staticmethod
    def _value_number(raw_value: Optional[str]) -> Optional[float]:
        """
        Convert common numeric representations into a comparable float.

        Examples:
            ₹81,415Mn
            Rs. 127 Cr
            1.6%
            -Rs. 452 Cr
            (452 Cr)
            2.8 Billion
        """
        if not raw_value:
            return None

        text = str(raw_value).lower().strip()

        # Handle accounting parentheses: (123.45)
        if text.startswith("(") and text.endswith(")"):
            text = f"-{text[1:-1]}"

        negative = bool(re.search(r"(^|[\s(])-", text))

        text = text.replace(",", "")
        text = text.replace("₹", "")
        text = text.replace("rs.", "")
        text = text.replace("rs", "")
        text = text.replace("%", "")

        match = re.search(r"(\d+(?:\.\d+)?)", text)
        if not match:
            return None

        value = float(match.group(1))

        if "billion" in text or re.search(r"\bbn\b", text):
            value *= 1_000_000_000
        elif "million" in text or re.search(r"\bmn\b", text):
            value *= 1_000_000
        elif "crore" in text or re.search(r"\bcr\b", text):
            value *= 10_000_000
        elif "lakh" in text or re.search(r"\blac\b", text):
            value *= 100_000
        elif "thousand" in text or re.search(r"\bk\b", text):
            value *= 1_000
        elif re.search(r"\bm\b", text):
            value *= 1_000_000

        return -value if negative else value

    @staticmethod
    def _norm_text(value: Optional[str]) -> str:
        if not value:
            return ""

        value = str(value).lower().strip()
        value = re.sub(r"[^a-z0-9%]+", " ", value)
        value = re.sub(r"\s+", " ", value)

        return value.strip()

    @staticmethod
    def _document_of(fact: Any) -> str:
        """
        Recover the originating document from provenance information.
        """
        source_file = getattr(fact, "source_file", None)
        if source_file:
            return str(source_file)

        table_ref = getattr(fact, "table_ref", None)
        if table_ref:
            return str(table_ref).split(":", 1)[0]

        evidence_id = getattr(fact, "evidence_id", None)
        if evidence_id:
            return str(evidence_id).split(":", 1)[0]

        return "unknown"

    @classmethod
    def _attribute_tokens(cls, value: Optional[str]) -> Set[str]:
        """
        Normalize an attribute into semantic tokens.

        Time/context words are removed, while meaningful metric concepts
        are preserved for compatibility checking.
        """
        text = cls._norm_text(value)

        stop = {
            # Grammar
            "the",
            "of",
            "from",
            "for",
            "in",
            "on",
            "at",
            "to",
            "and",
            "by",
            # Time/context
            "fy",
            "q1",
            "q2",
            "q3",
            "q4",
            "annual",
            "year",
            "quarter",
            "yoy",
            "qoq",
            "mom",
            "since",
            "inception",
            # Generic modifiers
            "total",
            "overall",
            "adjusted",
            "adj",
        }

        return {
            token
            for token in text.split()
            if token and token not in stop
        }

    @classmethod
    def _entity_compatible(cls, a: Any, b: Any) -> bool:
        """
        Require entities to be identical or obvious name variants.
        """
        entity_a = cls._norm_text(getattr(a, "entity", None))
        entity_b = cls._norm_text(getattr(b, "entity", None))

        if not entity_a or not entity_b:
            return True

        return (
            entity_a == entity_b
            or entity_a in entity_b
            or entity_b in entity_a
        )

    @classmethod
    def _attribute_compatible(cls, a: Any, b: Any) -> bool:
        """
        Determine whether two attributes describe the same metric.

        Conservative policy:
        - exact semantic token sets are compatible
        - one token set can be a subset of the other only when the
          additional tokens are non-semantic
        - genuinely different metric dimensions are rejected
        """
        attr_a = cls._attribute_tokens(getattr(a, "attribute", None))
        attr_b = cls._attribute_tokens(getattr(b, "attribute", None))

        if not attr_a or not attr_b:
            return False

        # Exact semantic match
        if attr_a == attr_b:
            return True

        # One side must at least be a subset
        if not (attr_a.issubset(attr_b) or attr_b.issubset(attr_a)):
            return False

        extra = (attr_b - attr_a) if attr_a.issubset(attr_b) else (attr_a - attr_b)

        # Distinct metric dimensions
        semantic_dimensions = {
            "margin",
            "growth",
            "rate",
            "ratio",
            "share",
            "yield",
            "volume",
            "value",
            "count",
            "size",
            "tonnage",
            "tonnes",
            "tons",
            "shipments",
            "shipment",
            "revenue",
            "sales",
            "ebitda",
            "pat",
            "profit",
            "loss",
            "fleet",
            "centers",
            "centres",
            "facilities",
            "hubs",
        }

        # A differing core metric dimension implies a distinct business metric
        if extra & semantic_dimensions:
            return False

        return True

    @classmethod
    def _same_context(cls, a: Any, b: Any) -> bool:
        """
        True only when both explicit time and scope are non-empty and match.
        """
        time_a = cls._norm_text(getattr(a, "time", None))
        time_b = cls._norm_text(getattr(b, "time", None))

        scope_a = cls._norm_text(getattr(a, "scope", None))
        scope_b = cls._norm_text(getattr(b, "scope", None))

        if not time_a or not time_b or not scope_a or not scope_b:
            return False

        return time_a == time_b and scope_a == scope_b

    @classmethod
    def _context_differs(cls, a: Any, b: Any) -> bool:
        """
        True when at least one explicit context dimension disagrees.
        """
        time_a = cls._norm_text(getattr(a, "time", None))
        time_b = cls._norm_text(getattr(b, "time", None))

        scope_a = cls._norm_text(getattr(a, "scope", None))
        scope_b = cls._norm_text(getattr(b, "scope", None))

        time_differs = bool(time_a and time_b and time_a != time_b)
        scope_differs = bool(scope_a and scope_b and scope_a != scope_b)

        return time_differs or scope_differs

    @classmethod
    def classify(
        cls,
        fact_a: Any,
        fact_b: Any,
        similarity: float,
    ) -> Dict[str, Any]:
        """
        Classify a pair that has already passed candidate blocking.
        """
        value_a = cls._value_number(getattr(fact_a, "raw_value", None))
        value_b = cls._value_number(getattr(fact_b, "raw_value", None))

        # Non-numeric comparisons are deferred to the LLM arbiter
        if value_a is None or value_b is None:
            return {
                "status": "UNCERTAIN",
                "reasoning": (
                    "The facts are semantically similar, but at least "
                    "one value is non-numeric and requires semantic "
                    "comparison."
                ),
                "confidence": 4,
                "resolution_method": "DETERMINISTIC",
            }

        # Numerically equivalent values
        tolerance = max(abs(value_a), abs(value_b), 1.0) * 0.001

        if abs(value_a - value_b) <= tolerance:
            return {
                "status": "CORROBORATION",
                "reasoning": (
                    "The facts refer to the same semantic metric and "
                    "have equivalent normalized numeric values."
                ),
                "confidence": 9,
                "resolution_method": "DETERMINISTIC",
            }

        # Different time/scope explains different values
        if cls._context_differs(fact_a, fact_b):
            return {
                "status": "CONTEXTUAL_RESOLUTION",
                "reasoning": (
                    "The values differ, but the facts have different "
                    "explicit time or scope context."
                ),
                "confidence": 8,
                "resolution_method": "DETERMINISTIC",
            }

        # Same explicit context + materially different values
        if cls._same_context(fact_a, fact_b):
            return {
                "status": "CONTRADICTION",
                "reasoning": (
                    "The facts refer to the same semantic metric under "
                    "the same explicit context but have materially "
                    "different normalized numeric values."
                ),
                "confidence": 9,
                "resolution_method": "DETERMINISTIC",
            }

        return {
            "status": "UNCERTAIN",
            "reasoning": (
                "The facts appear comparable, but their context is "
                "insufficient for deterministic comparison."
            ),
            "confidence": 4,
            "resolution_method": "DETERMINISTIC",
        }

    def find_and_classify(
        self, facts: List[Any]
    ) -> List[Tuple[CandidatePair, Dict[str, Any]]]:
        """
        Pipeline:
            1. Embedding similarity generates broad candidates.
            2. Entity compatibility removes unrelated entities.
            3. Attribute compatibility removes different metrics.
            4. Deterministic classification handles obvious relationships.
        """
        candidates = self.embedding_index.find_candidates(
            facts,
            similarity_threshold=self.similarity_threshold,
        )

        results: List[Tuple[CandidatePair, Dict[str, Any]]] = []

        for pair in candidates:
            fact_a = pair.fact_a
            fact_b = pair.fact_b

            # Never match facts that failed extraction/grounding
            if getattr(fact_a, "status", "VALID") != "VALID":
                continue

            if getattr(fact_b, "status", "VALID") != "VALID":
                continue

            # Entity blocking
            if not self._entity_compatible(fact_a, fact_b):
                continue

            # Metric/attribute blocking
            if not self._attribute_compatible(fact_a, fact_b):
                continue

            classification = self.classify(
                fact_a,
                fact_b,
                pair.similarity,
            )

            results.append((pair, classification))

        return results