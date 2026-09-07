import re
from typing import Optional

from src.schemas.fact_schema import ExtractedFact


def normalize_numeric_value(
    raw_value: str,
    unit: Optional[str],
) -> Optional[float]:
    """
    Convert a numeric value into a canonical base value.

    Examples:

        216.68 + ₹ million
            -> 216680000

        1.2 + USD billion
            -> 1200000000

        15% + percentage
            -> 0.15

    Returns None when the value cannot be safely normalized.
    """

    if not raw_value:
        return None

    text = raw_value.strip().lower()

    # ---------------------------------------------------------
    # 1. Remove common formatting
    # ---------------------------------------------------------

    text = text.replace(",", "")
    text = text.replace("₹", "")
    text = text.replace("$", "")
    text = text.replace("€", "")
    text = text.replace("£", "")

    # ---------------------------------------------------------
    # 2. Extract the first numeric value
    # ---------------------------------------------------------

    match = re.search(
        r"-?\d+(?:\.\d+)?",
        text,
    )

    if not match:
        return None

    value = float(match.group())

    # ---------------------------------------------------------
    # 3. Normalize percentages
    # ---------------------------------------------------------

    if "%" in text or (
        unit and "percent" in unit.lower()
    ):
        return value / 100

    # ---------------------------------------------------------
    # 4. Normalize scale
    # ---------------------------------------------------------

    combined = f"{text} {unit or ''}".lower()

    if "trillion" in combined:
        value *= 1_000_000_000_000

    elif "billion" in combined:
        value *= 1_000_000_000

    elif "million" in combined:
        value *= 1_000_000

    elif "thousand" in combined:
        value *= 1_000

    return value


def normalize_fact(
    fact: ExtractedFact,
) -> ExtractedFact:
    """
    Normalize a fact's numeric value when possible.

    Semantic values that cannot safely be converted to a
    number are left unchanged.
    """

    normalized = normalize_numeric_value(
        fact.raw_value,
        fact.unit,
    )

    if normalized is not None:
        fact.normalized_value = normalized

    return fact