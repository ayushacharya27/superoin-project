import re
from typing import List, Optional, Set, Tuple

from src.schemas.extraction import ExtractedFact


GENERIC_ATTRIBUTES: Set[str] = {
    "value",
    "amount",
    "number",
    "metric",
    "figure",
    "data",
    "result",
    "total value",
    "aggregated value",
}

TIME_PATTERNS: List[str] = [
    r"\bq[1-4]\s*fy\s*\d{2,4}\b",
    r"\bfy\s*\d{2,4}\b",
    r"\bq[1-4]\b",
    r"\bfiscal\s+\d{4}\b",
    r"\bnine months(?: period)? ended [a-z]+\s+\d{1,2},\s+\d{4}\b",
    r"\bsix months(?: period)? ended [a-z]+\s+\d{1,2},\s+\d{4}\b",
    r"\bthree months(?: period)? ended [a-z]+\s+\d{1,2},\s+\d{4}\b",
]

GENERIC_CONTEXT_WORDS: Set[str] = {
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
    "as",
    "basis",
    "period",
    "ended",
    "during",
    "year",
    "quarter",
    "fiscal",
    "financial",
}

REMOVABLE_MODIFIERS: Set[str] = {
    "total",
    "overall",
    "annual",
}

SEMANTIC_TERMS: Set[str] = {
    "margin",
    "growth",
    "change",
    "yoy",
    "qoq",
    "mom",
    "rate",
    "ratio",
    "share",
    "yield",
    "volume",
    "count",
    "ton",
    "tons",
    "tonnage",
    "shipment",
    "shipments",
    "revenue",
    "ebitda",
    "pat",
    "profit",
    "loss",
    "fleet",
    "size",
    "customer",
    "customers",
    "order",
    "orders",
    "center",
    "centers",
    "centre",
    "centres",
    "facility",
    "facilities",
    "adjusted",
    "adj",
}

NUMERIC_PATTERN = re.compile(
    r"""
    (?P<sign>-|\()?
    \s*
    (?P<number>
        \d{1,3}(?:,\d{3})+(?:\.\d+)?
        |
        \d+(?:\.\d+)?
    )
    \s*
    (?P<unit>
        %
        |
        cr
        |
        mn
        |
        bn
        |
        million
        |
        billion
        |
        thousand
        |
        k
        |
        m
        |
        b
        |
        crore
        |
        lakhs?
        |
        lacs?
    )?
    """,
    re.IGNORECASE | re.VERBOSE,
)


def _norm_text(value: Optional[str]) -> str:
    if not value:
        return ""

    value = str(value).lower().strip()
    value = value.replace("_", " ")
    value = re.sub(r"[^a-z0-9%()/.,\-\s]", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def _extract_time(text: str) -> Optional[str]:
    text = _norm_text(text)
    matches: List[Tuple[int, str]] = []

    for pattern in TIME_PATTERNS:
        for match in re.finditer(pattern, text):
            matches.append((match.start(), match.group(0).strip()))

    if not matches:
        return None

    matches.sort(key=lambda item: item[0])
    return matches[0][1]


def _remove_time(text: str) -> str:
    result = _norm_text(text)
    for pattern in TIME_PATTERNS:
        result = re.sub(pattern, " ", result)

    result = re.sub(r"\s+", " ", result)
    return result.strip()


def _clean_attribute(text: Optional[str]) -> str:
    result = _remove_time(text or "")
    tokens: List[str] = []

    for token in result.split():
        if token in GENERIC_CONTEXT_WORDS:
            continue
        if token in REMOVABLE_MODIFIERS:
            continue
        tokens.append(token)

    return " ".join(tokens).strip()


def _has_semantic_content(text: str) -> bool:
    tokens = set(_norm_text(text).split())
    return bool(tokens & SEMANTIC_TERMS)


def _derive_attribute_from_fact(fact: ExtractedFact) -> str:
    """
    Recover a useful metric name from information already emitted
    by the model. Never invent a metric.
    """
    attribute = _clean_attribute(fact.attribute)
    if attribute and attribute.lower() not in GENERIC_ATTRIBUTES:
        return attribute

    candidates: List[str] = []

    column_name = getattr(fact, "column_name", None)
    if column_name:
        candidates.append(_clean_attribute(column_name))

    if fact.qualifier:
        candidates.append(_clean_attribute(fact.qualifier))

    if fact.exact_quote:
        candidates.append(_clean_attribute(fact.exact_quote))

    for candidate in candidates:
        if not candidate:
            continue
        if candidate.lower() in GENERIC_ATTRIBUTES:
            continue
        if _has_semantic_content(candidate):
            return candidate

    return ""


def _numeric_candidates(text: Optional[str]) -> List[str]:
    """
    Extract plausible numeric source representations from text.
    Bare calendar years (1900-2100) without units are excluded.
    """
    if not text:
        return []

    matches = NUMERIC_PATTERN.finditer(str(text))
    results: List[str] = []

    for match in matches:
        raw = match.group(0).strip()
        number_text = (match.group("number") or "").replace(",", "")
        unit = (match.group("unit") or "").lower()

        try:
            number = float(number_text)
        except ValueError:
            continue

        if not unit and number.is_integer() and 1900 <= number <= 2100:
            continue

        results.append(raw)

    return results


def recover_numeric_raw_value(fact: ExtractedFact) -> Optional[str]:
    """
    Recover raw_value only from grounded source text.
    Requires exactly one plausible numeric candidate to avoid ambiguity.
    """
    quote_candidates = _numeric_candidates(fact.exact_quote)
    if len(quote_candidates) == 1:
        return quote_candidates[0]

    return None


def _normalize_singular_forms(attribute: str) -> str:
    replacements = {
        "shipments": "shipment",
        "customers": "customer",
        "orders": "order",
        "centres": "center",
        "centers": "center",
        "facilities": "facility",
        "tonnes": "ton",
        "tons": "ton",
    }

    normalized_tokens = [
        replacements.get(token, token) for token in attribute.split()
    ]
    return " ".join(normalized_tokens)


def canonicalize_fact(fact: ExtractedFact) -> ExtractedFact:
    """
    Deterministically normalize a grounded extracted fact.

    Rules:
      - clean entity/attribute text
      - move explicit periods from attribute to time
      - preserve meaningful semantic modifiers
      - normalize singular/plural forms
      - recover a numeric raw_value only from an exact grounded quote
      - never invent missing values
    """
    if fact.entity:
        fact.entity = fact.entity.strip()

    original_attribute = fact.attribute or ""
    fact.attribute = _derive_attribute_from_fact(fact)

    if fact.attribute and not fact.time:
        extracted_time = _extract_time(original_attribute)
        if extracted_time:
            fact.time = extracted_time

    if fact.time:
        fact.time = _norm_text(fact.time)

    if fact.attribute:
        fact.attribute = _clean_attribute(fact.attribute)
        fact.attribute = _normalize_singular_forms(fact.attribute)

    # Inspect the exact quote only when valid and missing a normalized value
    if (
        getattr(fact, "status", "VALID") == "VALID"
        and fact.raw_value
        and getattr(fact, "normalized_value", None) is None
    ):
        current_candidates = _numeric_candidates(fact.raw_value)

        if not current_candidates:
            recovered = recover_numeric_raw_value(fact)
            if recovered:
                print(
                    f"[Canonicalizer] Recovered numeric value: "
                    f"attribute={fact.attribute!r} "
                    f"raw_value={fact.raw_value!r} -> {recovered!r}"
                )
                fact.raw_value = recovered

    # Generic or empty attributes fail closed
    if not fact.attribute or fact.attribute.lower() in GENERIC_ATTRIBUTES:
        fact.status = "UNCERTAIN"

    return fact


def canonicalize_facts(facts: List[ExtractedFact]) -> List[ExtractedFact]:
    result: List[ExtractedFact] = []

    for fact in facts:
        if getattr(fact, "status", "VALID") != "VALID":
            result.append(fact)
            continue

        result.append(canonicalize_fact(fact))

    return result