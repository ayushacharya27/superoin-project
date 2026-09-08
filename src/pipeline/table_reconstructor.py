from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from src.schemas.evidence import EvidenceUnit
from src.schemas.table import (
    ReconstructedTableRow,
    TableContext,
)


def reconstruct_tables(
    evidence: List[EvidenceUnit],
) -> List[TableContext]:
    """
    Reconstruct TABLE_ROW evidence into logical tables.

    The grouping key intentionally uses only information actually
    present in the parsed evidence. No document-specific table IDs
    or schemas are assumed.
    """
    table_rows = [
        item for item in evidence if item.evidence_type == "TABLE_ROW"
    ]

    if not table_rows:
        return []

    buckets: Dict[Tuple[str, str, int], List[EvidenceUnit]] = defaultdict(list)

    for item in table_rows:
        key = (
            item.source_file,
            _table_identity(item),
            item.page_number,
        )
        buckets[key].append(item)

    tables: List[TableContext] = []

    for index, ((source, table_identity, page), rows) in enumerate(
        buckets.items(),
        start=1,
    ):
        table = _build_table(
            table_id=f"table-{index:05d}",
            source_file=source,
            page_number=page,
            table_identity=table_identity,
            rows=rows,
        )
        tables.append(table)

    return sorted(
        tables,
        key=lambda table: (
            table.source_file,
            table.start_page,
            table.table_id,
        ),
    )


def _table_identity(
    item: EvidenceUnit,
) -> str:
    """
    Derive a stable table identity.

    Prefer an explicit caption. Otherwise use the table label
    encoded by the parser. Fall back to a conservative structural
    signature.
    """
    if item.table_caption:
        return _normalize(item.table_caption)

    text = _normalize(item.text)

    # Existing extractor commonly emits:
    # "Table 1; row 6; ..."
    first_segment = text.split(";", 1)[0].strip()

    if first_segment.startswith("table "):
        return first_segment

    return (
        _normalize(item.section_heading)
        or first_segment
        or "unknown-table"
    )


def _build_table(
    table_id: str,
    source_file: str,
    page_number: int,
    table_identity: str,
    rows: List[EvidenceUnit],
) -> TableContext:
    rows = sorted(
        rows,
        key=lambda item: (
            _row_number(item),
            item.evidence_id,
        ),
    )

    reconstructed_rows = [_reconstruct_row(item) for item in rows]
    headers = _infer_headers(rows)
    unit = _infer_unit(rows)
    caption = _infer_caption(rows, table_identity)

    deterministic, confidence = _assess_structure(
        headers,
        reconstructed_rows,
        unit,
    )

    return TableContext(
        table_id=table_id,
        source_file=source_file,
        start_page=page_number,
        end_page=max(item.page_number for item in rows),
        caption=caption,
        unit=unit,
        headers=headers,
        rows=reconstructed_rows,
        deterministic=deterministic,
        confidence=confidence,
    )


def _reconstruct_row(
    item: EvidenceUnit,
) -> ReconstructedTableRow:
    pairs = _parse_pairs(item.text)
    row_label = item.row_label

    if not row_label:
        row_label = _infer_row_label(pairs)

    values = []
    for key, value in pairs:
        if row_label and value == row_label:
            continue

        if _looks_like_structural_metadata(key):
            continue

        values.append(f"{key}={value}")

    return ReconstructedTableRow(
        evidence_ids=[item.evidence_id],
        values=values,
        row_label=row_label,
        source_file=item.source_file,
        page_number=item.page_number,
    )


def _infer_headers(
    rows: List[EvidenceUnit],
) -> List[str]:
    headers = []

    for row in rows:
        pairs = _parse_pairs(row.text)

        for key, _ in pairs:
            candidate = key.strip()

            if not candidate:
                continue

            if _is_period(candidate):
                headers.append(candidate)
                continue

            if _looks_like_percentage(candidate):
                headers.append(candidate)
                continue

            if candidate.lower() in {
                "column_1",
                "column_2",
                "column_3",
                "column_4",
                "column_5",
                "column_6",
                "column_7",
                "column_8",
                "column_9",
                "column_10",
                "column_11",
                "column_12",
                "column_13",
                "column_14",
                "column_15",
            }:
                headers.append(candidate)

    return _unique(headers)


def _infer_unit(
    rows: List[EvidenceUnit],
) -> Optional[str]:
    for row in rows:
        if row.unit_hint:
            return row.unit_hint

        for key, _ in _parse_pairs(row.text):
            if _looks_like_unit(key):
                return key

    return None


def _infer_caption(
    rows: List[EvidenceUnit],
    table_identity: str,
) -> Optional[str]:
    for row in rows:
        if row.table_caption:
            return row.table_caption

    if table_identity != "unknown-table":
        return table_identity

    return None


def _assess_structure(
    headers: List[str],
    rows: List[ReconstructedTableRow],
    unit: Optional[str],
) -> Tuple[bool, float]:
    """
    Conservative structural assessment.

    A table is deterministic only when its rows show repeated,
    machine-readable key/value structure and the header shape is
    reasonably consistent.
    """
    if not rows:
        return False, 0.0

    rows_with_values = [row for row in rows if row.values]

    if not rows_with_values:
        return False, 0.95

    value_ratios = []
    for row in rows_with_values:
        numeric_count = sum(
            1
            for value in row.values
            if _value_is_numeric(value.split("=", 1)[-1])
        )
        value_ratios.append(numeric_count / max(len(row.values), 1))

    average_numeric_ratio = sum(value_ratios) / len(value_ratios)
    has_headers = len(headers) >= 2

    if has_headers and average_numeric_ratio >= 0.50:
        confidence = min(
            0.99,
            0.70 + 0.15 * min(len(headers), 2) + 0.10 * average_numeric_ratio,
        )
        return True, confidence

    if unit and average_numeric_ratio >= 0.60:
        return True, 0.82

    return False, 0.65


def _parse_pairs(
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


def _infer_row_label(
    pairs: List[Tuple[str, str]],
) -> Optional[str]:
    for key, value in pairs:
        if _value_is_numeric(value):
            continue

        if _is_period(key):
            continue

        if _looks_like_unit(key):
            continue

        if _looks_like_structural_metadata(key):
            continue

        return value

    return None


def _row_number(
    item: EvidenceUnit,
) -> int:
    text = _normalize(item.text)

    for segment in text.split(";"):
        segment = segment.strip()
        if not segment.lower().startswith("row "):
            continue

        if "=" not in segment:
            continue

        try:
            return int(segment.split("=", 1)[1])
        except ValueError:
            pass

    return 10**9


def _value_is_numeric(
    value: str,
) -> bool:
    cleaned = (
        value.strip()
        .replace(",", "")
        .replace("₹", "")
        .replace("%", "")
    )

    if cleaned.startswith("(") and cleaned.endswith(")"):
        cleaned = cleaned[1:-1]

    if cleaned in {
        "",
        "-",
        "—",
        "na",
        "n.a.",
        "n/a",
    }:
        return False

    try:
        float(cleaned)
        return True
    except ValueError:
        return False


def _is_period(
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
            "year ended",
            "months ended",
        )
    )


def _looks_like_percentage(
    value: str,
) -> bool:
    lower = value.lower()
    return (
        "%" in lower
        or "margin" in lower
        or "growth" in lower
        or "yoy" in lower
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
            "units",
        )
    )


def _looks_like_structural_metadata(
    value: str,
) -> bool:
    lower = value.lower().strip()
    return (
        lower.startswith("column_")
        or lower in {
            "table",
            "row",
            "s. no.",
            "s.no.",
            "serial",
        }
    )


def _normalize(
    value: Optional[str],
) -> str:
    if not value:
        return ""

    return " ".join(value.lower().split())


def _unique(
    values: List[str],
) -> List[str]:
    seen = set()
    result = []

    for value in values:
        if value in seen:
            continue

        seen.add(value)
        result.append(value)

    return result