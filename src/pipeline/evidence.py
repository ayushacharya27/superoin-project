import hashlib
from pathlib import Path
import re
from typing import List, Optional

import pymupdf

from src.schemas.evidence import EvidenceUnit


NUMBER_PATTERN = re.compile(
    r"""
    (?:
        [$€£₹]\s*
    )?
    [-+]?
    (?:
        \d{1,3}(?:,\d{3})+
        |
        \d+
    )
    (?:\.\d+)?
    %?
    """,
    re.VERBOSE,
)

TIME_PATTERN = re.compile(
    r"""
    (?:
        FY\s*\d{2,4}
        |
        Q[1-4]\s*(?:FY\s*)?\d{2,4}
        |
        \b(?:19|20)\d{2}\b
        |
        (?:January|February|March|April|May|June|July|August|
        September|October|November|December)\s+\d{4}
        |
        (?:March|June|September|December)\s+\d{1,2},?\s+\d{4}
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

UNIT_PATTERN = re.compile(
    r"""
    (?:
        INR|USD|EUR|GBP|JPY
        |
        ₹|\$|€|£
        |
        million|billion|thousand|crore|lakh
        |
        %|percent|percentage
        |
        tonnes?|kg|kilograms?|units?
        |
        employees?|people|customers?
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

NOISE_PATTERNS = [
    r"copyright",
    r"all rights reserved",
    r"safe harbour",
    r"safe harbor",
    r"disclaimer",
    r"registered office",
    r"corporate office",
    r"telephone and e-mail",
    r"company secretary",
    r"compliance officer",
    r"membership no",
    r"digitally signed",
    r"encl\s*:",
]

METRIC_TERMS = (
    "revenue",
    "sales",
    "income",
    "profit",
    "loss",
    "ebitda",
    "margin",
    "expense",
    "cost",
    "asset",
    "liabilit",
    "debt",
    "cash",
    "investment",
    "employees",
    "workforce",
    "customers",
    "shipments",
    "tonnes",
    "volume",
    "market share",
    "growth",
    "rate",
    "amount",
    "value",
    "orders",
    "parcels",
    "delivery",
    "deliveries",
)


def _stable_id(
    source_file: str,
    page_number: int,
    evidence_type: str,
    text: str,
) -> str:
    raw = f"{source_file}|{page_number}|{evidence_type}|{text.strip()}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _number_hints(text: str) -> List[str]:
    return [match.group(0).strip() for match in NUMBER_PATTERN.finditer(text)]


def _unit_hint(text: str) -> Optional[str]:
    match = UNIT_PATTERN.search(text)
    return match.group(0) if match else None


def _time_hint(text: str) -> Optional[str]:
    match = TIME_PATTERN.search(text)
    return match.group(0) if match else None


def _is_noise(text: str) -> bool:
    text = _clean_text(text)
    if not text:
        return True

    lower = text.lower()
    for pattern in NOISE_PATTERNS:
        if re.search(pattern, lower):
            return True

    return False


def _has_metric_language(text: str) -> bool:
    lower = text.lower()
    return any(term in lower for term in METRIC_TERMS)


def _looks_like_claim(text: str) -> bool:
    text = _clean_text(text)
    if len(text) < 15:
        return False

    if _is_noise(text):
        return False

    numbers = _number_hints(text)

    # Strongest signal: a meaningful statement containing a number.
    if numbers and (
        _has_metric_language(text)
        or _unit_hint(text)
        or _time_hint(text)
    ):
        return True

    # Comparative/quantitative language can express facts without an explicit numeric literal.
    lower = text.lower()
    comparative_terms = (
        "increased",
        "decreased",
        "grew",
        "declined",
        "rose",
        "fell",
        "higher",
        "lower",
        "more than",
        "less than",
    )

    return _has_metric_language(text) and any(
        term in lower for term in comparative_terms
    )


def _words_to_lines(
    words: List[dict],
    y_tolerance: float = 3.0,
) -> List[List[dict]]:
    """
    Reconstruct visual lines using word coordinates.

    PDF text extraction frequently inserts artificial line breaks.
    Coordinates give us a better approximation of what the reader sees.
    """
    sorted_words = sorted(
        words,
        key=lambda word: (
            round(word["y0"] / y_tolerance),
            word["x0"],
        ),
    )

    lines: List[List[dict]] = []

    for word in sorted_words:
        placed = False
        word_y = word["y0"]

        for line in lines:
            reference_y = line[0]["y0"]
            if abs(word_y - reference_y) <= y_tolerance:
                line.append(word)
                placed = True
                break

        if not placed:
            lines.append([word])

    for line in lines:
        line.sort(key=lambda word: word["x0"])

    lines.sort(
        key=lambda line: (
            min(word["y0"] for word in line),
            min(word["x0"] for word in line),
        )
    )

    return lines


def _line_text(line: List[dict]) -> str:
    return _clean_text(" ".join(word["text"] for word in line))


def _merge_nearby_claim_lines(lines: List[List[dict]]) -> List[str]:
    """
    Merge visually adjacent lines when they form one claim.

    Example:
        >2.8Bn
        Express parcel shipments
        delivered since inception
    becomes one evidence statement.
    """
    result: List[str] = []
    i = 0

    while i < len(lines):
        current = _line_text(lines[i])
        if not current:
            i += 1
            continue

        # Start with a potential claim.
        merged = current
        j = i + 1

        while j < len(lines):
            candidate = _line_text(lines[j])
            if not candidate:
                break

            # Stop if the next line already looks like an independent claim.
            if (
                j > i + 1
                and _looks_like_claim(candidate)
                and _number_hints(candidate)
            ):
                break

            combined = f"{merged} {candidate}"
            if len(combined) > 500:
                break

            # Continue merging if the current material is incomplete.
            if (
                _number_hints(merged)
                or _has_metric_language(merged)
                or not re.search(r"[.!?]$", merged)
            ):
                merged = combined
                j += 1
            else:
                break

        result.append(_clean_text(merged))
        i = max(j, i + 1)

    return result


def _extract_prose_evidence(
    page: pymupdf.Page,
    source_file: str,
    page_number: int,
) -> List[EvidenceUnit]:
    words_raw = page.get_text("words")
    words = []

    for item in words_raw:
        x0, y0, x1, y1, text = item[:5]
        text = _clean_text(text)
        if not text:
            continue

        words.append(
            {
                "x0": x0,
                "y0": y0,
                "x1": x1,
                "y1": y1,
                "text": text,
            }
        )

    lines = _words_to_lines(words)
    claims = _merge_nearby_claim_lines(lines)
    evidence: List[EvidenceUnit] = []

    for claim in claims:
        if not _looks_like_claim(claim):
            continue

        evidence.append(
            EvidenceUnit(
                evidence_id=_stable_id(
                    source_file,
                    page_number,
                    "PROSE",
                    claim,
                ),
                source_file=source_file,
                page_number=page_number,
                evidence_type="PROSE",
                text=claim,
                number_hints=_number_hints(claim),
                unit_hint=_unit_hint(claim),
                time_hint=_time_hint(claim),
            )
        )

    return evidence


def _table_to_rows(table) -> List[dict]:
    try:
        rows = table.extract()
    except Exception:
        return []

    if not rows:
        return []

    cleaned = [
        [_clean_text(str(cell)) if cell is not None else "" for cell in row]
        for row in rows
    ]

    # Remove completely empty rows.
    cleaned = [row for row in cleaned if any(row)]
    if not cleaned:
        return []

    headers = cleaned[0]
    result = []

    for row in cleaned[1:]:
        if not any(row):
            continue

        result.append(
            {
                "headers": headers,
                "values": row,
            }
        )

    return result


def _is_fact_bearing_table_row(
    headers: List[str],
    values: List[str],
) -> bool:
    text = " ".join(headers + values)
    if _is_noise(text):
        return False

    # A row needs either a measurable value or metric language.
    if _number_hints(text):
        return True

    return _has_metric_language(text)


def _extract_table_evidence(
    page: pymupdf.Page,
    source_file: str,
    page_number: int,
) -> List[EvidenceUnit]:
    evidence: List[EvidenceUnit] = []

    try:
        tables = page.find_tables()
    except Exception:
        return evidence

    for table_index, table in enumerate(tables.tables):
        rows = _table_to_rows(table)

        for row_index, row_data in enumerate(rows):
            headers = row_data["headers"]
            values = row_data["values"]

            if not _is_fact_bearing_table_row(headers, values):
                continue

            pairs = []
            for index, value in enumerate(values):
                if not value:
                    continue

                header = (
                    headers[index]
                    if index < len(headers) and headers[index]
                    else f"column_{index + 1}"
                )
                pairs.append(f"{header}={value}")

            if not pairs:
                continue

            row_text = (
                f"Table {table_index + 1}; "
                f"row {row_index + 1}; "
                f"{'; '.join(pairs)}"
            )

            row_label = values[0] if values and values[0] else None

            evidence.append(
                EvidenceUnit(
                    evidence_id=_stable_id(
                        source_file,
                        page_number,
                        "TABLE_ROW",
                        row_text,
                    ),
                    source_file=source_file,
                    page_number=page_number,
                    evidence_type="TABLE_ROW",
                    text=row_text,
                    row_label=row_label,
                    number_hints=_number_hints(row_text),
                    unit_hint=_unit_hint(row_text),
                    time_hint=_time_hint(row_text),
                )
            )

    return evidence


def extract_evidence(pdf_path: str) -> List[EvidenceUnit]:
    source_file = Path(pdf_path).name
    document = pymupdf.open(pdf_path)
    all_evidence: List[EvidenceUnit] = []

    try:
        for page_number, page in enumerate(document, start=1):
            all_evidence.extend(
                _extract_table_evidence(
                    page,
                    source_file,
                    page_number,
                )
            )
            all_evidence.extend(
                _extract_prose_evidence(
                    page,
                    source_file,
                    page_number,
                )
            )
    finally:
        document.close()

    return all_evidence