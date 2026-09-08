from typing import List, Literal, Tuple

from pydantic import BaseModel, Field

from src.schemas.evidence import EvidenceUnit


class TableDecision(BaseModel):
    evidence_id: str
    decision: Literal["DETERMINISTIC", "LLM"]
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str


def route_table_rows(
    evidence: List[EvidenceUnit],
) -> List[TableDecision]:
    """
    Decide whether a table row can be processed deterministically.

    This is deliberately schema-agnostic. We do not know in advance
    what columns or metrics a PDF contains.
    """
    decisions = []

    for item in evidence:
        if item.evidence_type != "TABLE_ROW":
            continue

        decision, confidence, reason = _classify_table_row(item)

        decisions.append(
            TableDecision(
                evidence_id=item.evidence_id,
                decision=decision,
                confidence=confidence,
                reason=reason,
            )
        )

    return decisions


def _classify_table_row(
    item: EvidenceUnit,
) -> Tuple[str, float, str]:
    text = " ".join(item.text.split())
    pairs = _extract_pairs(text)

    if len(pairs) < 2:
        return (
            "LLM",
            0.95,
            "Insufficient key-value structure",
        )

    value_pairs = [
        (key, value)
        for key, value in pairs
        if _looks_like_value(value)
    ]

    if len(value_pairs) < 1:
        return (
            "LLM",
            0.90,
            "No reliable value-bearing columns detected",
        )

    structural_noise = 0

    for key, value in pairs:
        if _looks_like_period(key):
            continue

        if _looks_like_percentage_column(key):
            continue

        if _looks_like_unit_header(key):
            continue

        if _looks_like_serial_column(key):
            structural_noise += 1

    ratio = len(value_pairs) / max(len(pairs), 1)

    # Simple rows with several obvious value columns are safe to
    # parse without an LLM.
    if len(value_pairs) >= 2 and ratio >= 0.50:
        return (
            "DETERMINISTIC",
            min(0.99, 0.75 + 0.05 * len(value_pairs)),
            "Regular key-value table row",
        )

    if structural_noise > len(pairs) // 2:
        return (
            "LLM",
            0.85,
            "Mostly structural/header information",
        )

    return (
        "LLM",
        0.70,
        "Table structure is ambiguous",
    )


def _extract_pairs(text: str) -> List[Tuple[str, str]]:
    pairs = []

    for segment in text.split(";"):
        segment = segment.strip()

        if "=" not in segment:
            continue

        key, value = segment.split("=", 1)
        key = key.strip()
        value = value.strip()

        if not key or not value:
            continue

        pairs.append((key, value))

    return pairs


def _looks_like_value(value: str) -> bool:
    value = value.strip()

    if not value:
        return False

    cleaned = (
        value.replace(",", "")
        .replace("₹", "")
        .replace("%", "")
        .replace("(", "")
        .replace(")", "")
        .replace("-", "")
        .strip()
    )

    if not cleaned:
        return False

    # Numeric / numeric-like value.
    try:
        float(cleaned)
        return True
    except ValueError:
        pass

    # Common textual table values.
    textual_values = {
        "yes",
        "no",
        "nil",
        "none",
        "n.a.",
        "na",
        "n/a",
    }

    return cleaned.lower() in textual_values


def _looks_like_period(value: str) -> bool:
    lower = value.lower()

    period_signals = (
        "fy",
        "q1",
        "q2",
        "q3",
        "q4",
        "march",
        "june",
        "september",
        "december",
        "year",
        "month",
        "ended",
    )

    return any(signal in lower for signal in period_signals)


def _looks_like_percentage_column(value: str) -> bool:
    lower = value.lower()

    return (
        "%" in lower
        or "margin" in lower
        or "growth" in lower
    )


def _looks_like_unit_header(value: str) -> bool:
    lower = value.lower()

    unit_signals = (
        "₹",
        "rs",
        "inr",
        "usd",
        "million",
        "crore",
        "cr",
        "lakhs",
        "units",
    )

    return any(signal in lower for signal in unit_signals)


def _looks_like_serial_column(value: str) -> bool:
    lower = value.lower()
    return lower in {"s. no.", "s.no.", "no.", "serial", "sr. no."}