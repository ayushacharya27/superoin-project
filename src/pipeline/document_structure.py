import hashlib
from pathlib import Path
from typing import List, Optional, Tuple

import pymupdf

from src.schemas.document import (
    DocumentElement,
    StructuredTable,
)


def extract_document_elements(
    pdf_path: str,
) -> List[DocumentElement]:
    """
    Build a structure-aware, reading-order representation of a PDF.

    Tables are extracted as first-class objects from PyMuPDF rather
    than reconstructed later from flattened table-row evidence.
    """
    path = Path(pdf_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file: {pdf_path}")

    document = pymupdf.open(pdf_path)
    all_elements: List[DocumentElement] = []

    try:
        for page_number, page in enumerate(
            document,
            start=1,
        ):
            page_elements = _extract_page_elements(
                page=page,
                source_file=path.name,
                page_number=page_number,
            )
            all_elements.extend(page_elements)

    finally:
        document.close()

    return _sort_elements(all_elements)


def _extract_page_elements(
    page,
    source_file: str,
    page_number: int,
) -> List[DocumentElement]:
    tables = _extract_tables(
        page=page,
        source_file=source_file,
        page_number=page_number,
    )

    table_bboxes = [
        table.bbox
        for table in tables
        if table.bbox is not None
    ]

    prose_and_headings = _extract_text_elements(
        page=page,
        source_file=source_file,
        page_number=page_number,
        table_bboxes=table_bboxes,
    )

    elements = list(prose_and_headings)

    for table in tables:
        top = (
            float(table.bbox[1])
            if table.bbox
            else 0.0
        )

        elements.append(
            DocumentElement(
                element_id=(
                    f"{source_file}:"
                    f"{page_number}:"
                    f"table:{table.table_id}"
                ),
                source_file=source_file,
                page_number=page_number,
                element_type="TABLE",
                text=_table_to_text(table),
                heading=None,
                table=table,
                evidence_ids=table.evidence_ids,
                top=top,
            )
        )

    return _sort_elements(elements)


def _extract_tables(
    page,
    source_file: str,
    page_number: int,
) -> List[StructuredTable]:
    results: List[StructuredTable] = []

    try:
        finder = page.find_tables()
    except Exception:
        return results

    raw_tables = getattr(
        finder,
        "tables",
        [],
    )

    for table_index, raw_table in enumerate(
        raw_tables,
        start=1,
    ):
        extracted = _safe_extract_table(raw_table)

        if not extracted:
            continue

        cleaned_rows = _clean_rows(extracted)

        if not cleaned_rows:
            continue

        headers, body_rows = _split_headers(cleaned_rows)
        bbox = _extract_bbox(raw_table)
        evidence_ids = []

        for row_index, row in enumerate(
            body_rows,
            start=1,
        ):
            evidence_ids.append(
                _stable_table_row_id(
                    source_file=source_file,
                    page_number=page_number,
                    table_index=table_index,
                    row_index=row_index,
                    row=row,
                )
            )

        results.append(
            StructuredTable(
                table_id=str(table_index),
                page_number=page_number,
                caption=None,
                unit=None,
                headers=headers,
                rows=body_rows,
                evidence_ids=evidence_ids,
                bbox=bbox,
            )
        )

    return results


def _safe_extract_table(
    raw_table,
):
    try:
        return raw_table.extract()
    except Exception:
        return None


def _clean_rows(
    rows,
) -> List[List[str]]:
    cleaned = []

    for row in rows:
        if not row:
            continue

        cleaned_row = [
            _clean_cell(cell)
            for cell in row
        ]

        if not any(cleaned_row):
            continue

        cleaned.append(cleaned_row)

    return cleaned


def _split_headers(
    rows: List[List[str]],
) -> Tuple[List[str], List[List[str]]]:
    """
    Use the first sufficiently populated row as the header row.

    We avoid assuming specific column names.
    """
    if len(rows) == 1:
        return [], rows

    first = rows[0]
    non_empty = sum(bool(cell) for cell in first)

    # A sparse first row is less likely to be a real header.
    if non_empty < 2:
        return [], rows

    return first, rows[1:]


def _extract_bbox(
    raw_table,
) -> Optional[List[float]]:
    try:
        rect = raw_table.bbox
        return [
            float(rect[0]),
            float(rect[1]),
            float(rect[2]),
            float(rect[3]),
        ]
    except Exception:
        return None


def _extract_text_elements(
    page,
    source_file: str,
    page_number: int,
    table_bboxes: List[Optional[List[float]]],
) -> List[DocumentElement]:
    words = page.get_text("words")

    if not words:
        return []

    lines = _words_to_lines(words)
    elements = []

    for index, line in enumerate(
        lines,
        start=1,
    ):
        text = _line_text(line)

        if not text:
            continue

        bbox = _line_bbox(line)

        if _inside_any_table(
            bbox=bbox,
            table_bboxes=table_bboxes,
        ):
            continue

        if _is_noise(text):
            continue

        element_type = (
            "HEADING"
            if _looks_like_heading(text)
            else "PROSE"
        )

        elements.append(
            DocumentElement(
                element_id=(
                    f"{source_file}:"
                    f"{page_number}:"
                    f"{element_type.lower()}:"
                    f"{index}"
                ),
                source_file=source_file,
                page_number=page_number,
                element_type=element_type,
                text=text,
                heading=(
                    text
                    if element_type == "HEADING"
                    else None
                ),
                table=None,
                evidence_ids=[],
                top=float(bbox[1]),
            )
        )

    return elements


def _words_to_lines(
    words,
) -> List[dict]:
    """
    Reconstruct visual lines from PyMuPDF word coordinates.
    """
    lines = []

    for word in words:
        x0, y0, x1, y1, text, block, line, word_index = word[:8]
        matched = None

        for candidate in lines:
            if abs(float(candidate["y"]) - float(y0)) <= 3.0:
                matched = candidate
                break

        if matched is None:
            lines.append(
                {
                    "y": float(y0),
                    "words": [word],
                }
            )
        else:
            matched["words"].append(word)

    for line in lines:
        line["words"].sort(key=lambda item: float(item[0]))

    lines.sort(key=lambda item: item["y"])
    return lines


def _line_text(
    line: dict,
) -> str:
    return " ".join(
        str(word[4])
        for word in line["words"]
    ).strip()


def _line_bbox(
    line: dict,
) -> List[float]:
    words = line["words"]
    return [
        min(float(word[0]) for word in words),
        min(float(word[1]) for word in words),
        max(float(word[2]) for word in words),
        max(float(word[3]) for word in words),
    ]


def _inside_any_table(
    bbox: List[float],
    table_bboxes: List[Optional[List[float]]],
) -> bool:
    if not table_bboxes:
        return False

    x0, y0, x1, y1 = bbox

    for table_bbox in table_bboxes:
        if table_bbox is None:
            continue

        tx0, ty0, tx1, ty1 = table_bbox
        overlap_x = max(x0, tx0) < min(x1, tx1)
        overlap_y = max(y0, ty0) < min(y1, ty1)

        if overlap_x and overlap_y:
            return True

    return False


def _looks_like_heading(
    text: str,
) -> bool:
    cleaned = " ".join(text.split())

    if not cleaned:
        return False

    if len(cleaned) > 140:
        return False

    word_count = len(cleaned.split())

    if word_count > 14:
        return False

    if cleaned.endswith("."):
        return False

    alpha_chars = [char for char in cleaned if char.isalpha()]

    if not alpha_chars:
        return False

    uppercase_ratio = sum(
        char.isupper()
        for char in alpha_chars
    ) / len(alpha_chars)

    if uppercase_ratio >= 0.65:
        return True

    # Numbered section headings like: "2. Financial Performance"
    if cleaned[0].isdigit() and len(cleaned) <= 100:
        return True

    return False


def _is_noise(
    text: str,
) -> bool:
    lower = " ".join(text.lower().split())

    if not lower:
        return True

    exact_noise = {
        "www.delhivery.com",
        "www.bseindia.com",
        "www.nseindia.com",
        "this page is intentionally left blank",
    }

    if lower in exact_noise:
        return True

    prefix_noise = (
        "cin:",
        "registered office:",
        "corporate identity number:",
        "page no.",
        "page:",
    )

    if lower.startswith(prefix_noise):
        return True

    # Isolated page numbers.
    if lower.isdigit() and len(lower) <= 4:
        return True

    return False


def _clean_cell(
    value,
) -> str:
    if value is None:
        return ""

    return " ".join(str(value).split())


def _table_to_text(
    table: StructuredTable,
) -> str:
    parts = ["[TABLE]"]

    if table.caption:
        parts.append(f"Caption: {table.caption}")

    if table.unit:
        parts.append(f"Unit: {table.unit}")

    if table.headers:
        parts.append("Headers: " + " | ".join(table.headers))

    for row in table.rows:
        parts.append(" | ".join(row))

    return "\n".join(parts)


def _stable_table_row_id(
    source_file: str,
    page_number: int,
    table_index: int,
    row_index: int,
    row: List[str],
) -> str:
    raw = (
        f"{source_file}|"
        f"{page_number}|"
        f"{table_index}|"
        f"{row_index}|"
        f"{'|'.join(row)}"
    )

    return hashlib.sha1(
        raw.encode("utf-8")
    ).hexdigest()[:16]


def _sort_elements(
    elements: List[DocumentElement],
) -> List[DocumentElement]:
    return sorted(
        elements,
        key=lambda element: (
            element.source_file,
            element.page_number,
            element.top,
            element.element_id,
        ),
    )