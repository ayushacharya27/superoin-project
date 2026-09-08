import re
from typing import Any, Dict, List, Optional, Set, Tuple

from src.matcher.embeddings import CandidatePair, FactEmbeddingIndex


class FactMatcher:
    def __init__(
        self,
        similarity_threshold: float = 0.75,
    ) -> None:
        self.similarity_threshold = similarity_threshold
        self.embedding_index = FactEmbeddingIndex()

    @staticmethod
    def _value_number(
        raw_value: Optional[str],
    ) -> Optional[float]:
        """
        Parse common numeric representations into comparable floats.
        """
        if not raw_value:
            return None

        text = str(raw_value).lower().strip()

        # Handle accounting negative parenthesized notation: (123.45)
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
        elif "thousand" in text or re.search(r"\bk\b", text):
            value *= 1_000
        elif re.search(r"\bcr\b", text):
            value *= 10_000_000
        elif re.search(r"\bm\b", text):
            value *= 1_000_000

        if negative:
            value = -value

        return value

    @staticmethod
    def _norm_text(
        value: Optional[str],
    ) -> str:
        if not value:
            return ""

        value = str(value).lower().strip()
        value = re.sub(r"[^a-z0-9%]+", " ", value)
        value = re.sub(r"\s+", " ", value)
        return value.strip()

    @staticmethod
    def _document_of(
        fact: Any,
    ) -> str:
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
    def _attribute_tokens(
        cls,
        value: Optional[str],
    ) -> Set[str]:
        """
        Normalize an attribute into semantic tokens.

        Time markers are excluded from metric identity; meaningful
        dimensions such as adjusted, margin, and growth remain.
        """
        text = cls._norm_text(value)

        stop = {
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
        }

        return {
            token
            for token in text.split()
            if token and token not in stop
        }

    @classmethod
    def _entity_compatible(
        cls,
        a: Any,
        b: Any,
    ) -> bool:
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
    def _attribute_compatible(
        cls,
        a: Any,
        b: Any,
    ) -> bool:
        """
        Conservative metric identity check.

        Examples:
            EBITDA margin vs FY24 EBITDA margin -> compatible
            EBITDA margin vs Adjusted EBITDA margin -> incompatible
            Express parcel shipment vs Express parcel shipment YoY change -> incompatible
        """
        attr_a = cls._attribute_tokens(getattr(a, "attribute", None))
        attr_b = cls._attribute_tokens(getattr(b, "attribute", None))

        if not attr_a or not attr_b:
            return False

        if attr_a == attr_b:
            return True

        # A strict subset is only allowed when extra tokens are purely contextual
        if not (attr_a.issubset(attr_b) or attr_b.issubset(attr_a)):
            return False

        extra = (attr_b - attr_a) if attr_a.issubset(attr_b) else (attr_a - attr_b)

        semantic_dimensions = {
            "adjusted",
            "adj",
            "margin",
            "growth",
            "change",
            "rate",
            "ratio",
            "share",
            "yield",
            "volume",
            "value",
            "count",
            "size",
            "ton",
            "tonnage",
            "shipment",
            "revenue",
            "sales",
            "ebitda",
            "pat",
            "profit",
            "loss",
            "fleet",
            "customer",
            "order",
            "center",
            "facility",
            "hub",
        }

        if extra & semantic_dimensions:
            return False

        return True

    @classmethod
    def _same_time(
        cls,
        a: Any,
        b: Any,
    ) -> bool:
        time_a = cls._norm_text(getattr(a, "time", None))
        time_b = cls._norm_text(getattr(b, "time", None))

        # Both must be explicitly present and equal
        return bool(time_a and time_b and time_a == time_b)

    @classmethod
    def _time_differs(
        cls,
        a: Any,
        b: Any,
    ) -> bool:
        time_a = cls._norm_text(getattr(a, "time", None))
        time_b = cls._norm_text(getattr(b, "time", None))

        return bool(time_a and time_b and time_a != time_b)

    @classmethod
    def _same_scope(
        cls,
        a: Any,
        b: Any,
    ) -> bool:
        scope_a = cls._norm_text(getattr(a, "scope", None))
        scope_b = cls._norm_text(getattr(b, "scope", None))

        return bool(scope_a and scope_b and scope_a == scope_b)

    @classmethod
    def _scope_differs(
        cls,
        a: Any,
        b: Any,
    ) -> bool:
        scope_a = cls._norm_text(getattr(a, "scope", None))
        scope_b = cls._norm_text(getattr(b, "scope", None))

        return bool(scope_a and scope_b and scope_a != scope_b)

    @classmethod
    def _same_context(
        cls,
        a: Any,
        b: Any,
    ) -> bool:
        """
        Require every explicitly supplied context dimension to agree.
        Missing context does not automatically count as equal.
        """
        times_present = bool(getattr(a, "time", None) or getattr(b, "time", None))
        scopes_present = bool(getattr(a, "scope", None) or getattr(b, "scope", None))

        time_same = cls._same_time(a, b) if times_present else True
        scope_same = cls._same_scope(a, b) if scopes_present else True

        return time_same and scope_same

    @classmethod
    def _has_explicit_context_difference(
        cls,
        a: Any,
        b: Any,
    ) -> bool:
        return cls._time_differs(a, b) or cls._scope_differs(a, b)

    @classmethod
    def _unit_type(
        cls,
        fact: Any,
    ) -> str:
        """
        Coarse value type to prevent comparisons such as percentage vs currency.
        """
        raw = getattr(fact, "raw_value", None) or ""
        text = str(raw).lower()

        if "%" in text:
            return "PERCENT"

        if (
            re.search(r"\b(cr|mn|million|billion|thousand|rs|₹)\b", text)
            or "₹" in str(raw)
        ):
            return "CURRENCY"

        if re.search(r"\b(ton|tons|tonne|tonnes|shipment|shipments)\b", text):
            return "VOLUME"

        if cls._value_number(raw) is not None:
            return "NUMBER"

        return "TEXT"

    @classmethod
    def classify(
        cls,
        fact_a: Any,
        fact_b: Any,
        similarity: float,
    ) -> Dict[str, Any]:
        """
        Classify candidate pair after entity and metric compatibility checks.
        """
        value_a = cls._value_number(getattr(fact_a, "raw_value", None))
        value_b = cls._value_number(getattr(fact_b, "raw_value", None))

        # Non-numeric facts remain unresolved
        if value_a is None or value_b is None:
            return {
                "status": "UNCERTAIN",
                "reasoning": (
                    "The facts share a metric identity, but at least "
                    "one value is non-numeric."
                ),
                "confidence": 4,
                "resolution_method": "DETERMINISTIC",
            }

        unit_a = cls._unit_type(fact_a)
        unit_b = cls._unit_type(fact_b)

        # Percentages cannot corroborate currency or raw counts
        if unit_a != unit_b:
            return {
                "status": "UNCERTAIN",
                "reasoning": (
                    "The metric names are similar, but the values "
                    "use incompatible unit types."
                ),
                "confidence": 7,
                "resolution_method": "DETERMINISTIC",
            }

        tolerance = max(abs(value_a), abs(value_b), 1.0) * 0.001
        values_equal = abs(value_a - value_b) <= tolerance

        # Same explicit context + same value
        if values_equal and cls._same_context(fact_a, fact_b):
            return {
                "status": "CORROBORATION",
                "reasoning": (
                    "The facts refer to the same semantic metric under "
                    "the same explicit context and have equivalent normalized values."
                ),
                "confidence": 9,
                "resolution_method": "DETERMINISTIC",
            }

        # Same explicit context + different value
        has_explicit_context = bool(
            getattr(fact_a, "time", None)
            or getattr(fact_b, "time", None)
            or getattr(fact_a, "scope", None)
            or getattr(fact_b, "scope", None)
        )

        if not values_equal and cls._same_context(fact_a, fact_b) and has_explicit_context:
            return {
                "status": "CONTRADICTION",
                "reasoning": (
                    "The facts refer to the same semantic metric under "
                    "the same explicit context but have materially different values."
                ),
                "confidence": 9,
                "resolution_method": "DETERMINISTIC",
            }

        # Contextual resolution: explicit disagreement in time or scope
        if not values_equal and cls._has_explicit_context_difference(fact_a, fact_b):
            return {
                "status": "CONTEXTUAL_RESOLUTION",
                "reasoning": (
                    "The same metric has different values under "
                    "explicitly different time or scope contexts."
                ),
                "confidence": 7,
                "resolution_method": "DETERMINISTIC",
            }

        # Equal values with incomplete context
        if values_equal:
            return {
                "status": "UNCERTAIN",
                "reasoning": (
                    "The normalized values are equivalent, "
                    "but explicit context is incomplete."
                ),
                "confidence": 6,
                "resolution_method": "DETERMINISTIC",
            }

        return {
            "status": "UNCERTAIN",
            "reasoning": (
                "The facts appear comparable, but there is not enough "
                "explicit context to determine whether the difference is a "
                "contradiction or a contextual difference."
            ),
            "confidence": 4,
            "resolution_method": "DETERMINISTIC",
        }

    def find_and_classify(
        self,
        facts: List[Any],
    ) -> List[Tuple[CandidatePair, Dict[str, Any]]]:
        """
        Generate embedding candidates, then apply conservative
        entity/metric blocking before relationship classification.
        """
        candidates = self.embedding_index.find_candidates(
            facts,
            similarity_threshold=self.similarity_threshold,
        )

        results: List[Tuple[CandidatePair, Dict[str, Any]]] = []

        for pair in candidates:
            fact_a = pair.fact_a
            fact_b = pair.fact_b

            if getattr(fact_a, "status", "VALID") != "VALID":
                continue

            if getattr(fact_b, "status", "VALID") != "VALID":
                continue

            if not self._entity_compatible(fact_a, fact_b):
                continue

            if not self._attribute_compatible(fact_a, fact_b):
                continue

            classification = self.classify(
                fact_a,
                fact_b,
                pair.similarity,
            )

            results.append((pair, classification))

        return results