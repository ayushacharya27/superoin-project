from typing import List

from src.schemas.evidence import EvidenceUnit


def select_fact_candidates(
    evidence: List[EvidenceUnit],
    min_score: int = 5,
) -> List[EvidenceUnit]:
    """
    Select evidence that is likely to contain a meaningful fact.

    This is intentionally schema-agnostic and document-agnostic.
    It does not contain company-specific metric names.
    """
    candidates = []

    for item in evidence:
        score = _fact_worthiness_score(item)
        if score >= min_score:
            candidates.append(item)

    return candidates


def _fact_worthiness_score(
    item: EvidenceUnit,
) -> int:
    score = 0
    text = item.text.strip()

    if not text:
        return 0

    # Strongest signals: explicit quantitative information.
    if item.number_hints:
        score += 3

    if item.unit_hint:
        score += 3

    if item.time_hint:
        score += 2

    if item.entity_hint:
        score += 2

    # Tables are generally information-dense.
    if item.evidence_type == "TABLE_ROW":
        score += 3

    # Metric-like language is useful even when extraction did not
    # identify a number/unit explicitly.
    if _has_metric_language(item):
        score += 2

    # A sentence containing both a number and reasonably long
    # explanatory text is more likely to express a real fact.
    if item.number_hints and len(text) >= 40:
        score += 1

    # Very short fragments are usually table/header noise.
    if len(text) < 20:
        score -= 3

    # Obvious metadata should not survive merely because it contains numbers.
    if _looks_like_metadata(text):
        score -= 5

    return score


def _has_metric_language(
    item: EvidenceUnit,
) -> bool:
    """
    Generic metric detection.

    This deliberately uses linguistic patterns rather than a
    company-specific metric dictionary.
    """
    text = item.text.lower()

    metric_patterns = (
        "revenue",
        "income",
        "profit",
        "loss",
        "ebitda",
        "margin",
        "growth",
        "cost",
        "expense",
        "cash flow",
        "assets",
        "liabilities",
        "equity",
        "debt",
        "shares",
        "customers",
        "employees",
        "shipments",
        "volume",
        "rate",
        "ratio",
        "percentage",
        "holding",
        "capital",
        "turnover",
        "acquisition",
        "valuation",
    )

    return any(
        pattern in text
        for pattern in metric_patterns
    )


def _looks_like_metadata(
    text: str,
) -> bool:
    text = text.lower()

    metadata_patterns = (
        "cin:",
        "registered office",
        "corporate identity number",
        "table of contents",
        "contents",
        "page no",
        "all amounts in indian rupees",
        "unless otherwise stated",
    )

    return any(
        pattern in text
        for pattern in metadata_patterns
    )