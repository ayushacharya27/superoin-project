import hashlib
import re
from typing import List, Optional, Tuple

from src.schemas.evidence import EvidenceUnit
from src.schemas.fact import CanonicalFact


NUMBER_RE = re.compile(r"^\(?-?\d[\d,]*(?:\.\d+)?\)?$")


def extract_deterministic_table_facts(
    evidence: List[EvidenceUnit],
) -> List[CanonicalFact]:
    facts = []

    for item in evidence:
        if item.evidence_type != "TABLE_ROW":
            continue

        row_facts = _extract_row_facts(item)
        facts.extend(row_facts)

    return facts


def _extract_row_facts(
    item: EvidenceUnit,
) -> List[CanonicalFact]:
    pairs = _extract_pairs(item.text)

    if len(pairs) < 2:
        return []

    row_label = _find_row_label(pairs)
    if not row_label:
        return []

    unit = _find_unit(pairs)
    facts = []

    for key, value in pairs:
        if key == row_label:
            continue

        numeric = _parse_number(value)
        if numeric is None:
            continue

        if _is_metadata_column(key):
            continue

        facts.append(
            CanonicalFact(
                fact_id=_stable_fact_id(
                    item,
                    key,
                    value,
                ),
                entity=(
                    item.entity_hint
                    or "UNKNOWN"
                ),
                attribute=row_label,
                raw_value=value,
                unit=unit,
                normalized_value=numeric,
                time=key,
                scope=None,
                qualifier=None,
                exact_quote=item.text,
                source_file=item.source_file,
                page_number=item.page_number,
                evidence_id=item.evidence_id,
                extraction_confidence=9,
                status="VALID",
            )
        )

    return facts


def _extract_pairs(
    text: str,
) -> List[Tuple[str, str]]:
    pairs = []

    for segment in text.split(";"):
        segment = segment.strip()
        if "=" not in segment:
            continue

        key, value = segment.split("=", 1)
        key = key.strip()
        value = value.strip()

        if key and value:
            pairs.append((key, value))

    return pairs


def _find_row_label(
    pairs: List[Tuple[str, str]],
) -> Optional[str]:
    """
    Find the first key whose value is clearly descriptive rather
    than numeric.

    Example:
        ₹ Cr = Revenue from customers
        Q1 FY24 = 1,930
    """
    for key, value in pairs:
        if _is_numeric(value):
            continue

        if _is_metadata_column(key):
            continue

        if _looks_like_period(key):
            continue

        if _looks_like_unit(key):
            continue

        return value

    return None


def _find_unit(
    pairs: List[Tuple[str, str]],
) -> Optional[str]:
    for key, _ in pairs:
        if _looks_like_unit(key):
            return key

    return None


def _parse_number(
    value: str,
) -> Optional[float]:
    cleaned = (
        value.strip()
        .replace(",", "")
        .replace("₹", "")
    )

    # Accounting negative: (217) -> -217
    if cleaned.startswith("(") and cleaned.endswith(")"):
        cleaned = "-" + cleaned[1:-1]

    cleaned = cleaned.replace("%", "")

    if cleaned in {
        "",
        "-",
        "—",
        "na",
        "n.a.",
        "n/a",
    }:
        return None

    try:
        return float(cleaned)
    except ValueError:
        return None


def _is_numeric(
    value: str,
) -> bool:
    return _parse_number(value) is not None


def _is_metadata_column(
    key: str,
) -> bool:
    lower = key.lower().strip()
    return lower in {
        "table",
        "row",
        "column",
        "s. no.",
        "s.no.",
        "serial",
        "no.",
    }


def _looks_like_period(
    value: str,
) -> bool:
    lower = value.lower()
    return any(
        token in lower
        for token in (
            "fy",
            "q1",
            "q2",
            "q3",
            "q4",
            "march",
            "june",
            "september",
            "december",
        )
    )


def _looks_like_unit(
    value: str,
) -> bool:
    lower = value.lower()
    return any(
        token in lower
        for token in (
            "₹",
            "rs",
            "inr",
            "usd",
            "million",
            "crore",
            "cr",
            "lakhs",
        )
    )


def _stable_fact_id(
    item: EvidenceUnit,
    column: str,
    value: str,
) -> str:
    raw = (
        f"{item.evidence_id}|"
        f"{column}|"
        f"{value}"
    )
    return hashlib.sha1(
        raw.encode("utf-8")
    ).hexdigest()[:16]