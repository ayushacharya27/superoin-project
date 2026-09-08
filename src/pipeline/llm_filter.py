from collections import Counter
from typing import Dict, List, Tuple

from src.schemas.document import DocumentElement


def filter_for_llm(
    elements: List[DocumentElement],
) -> List[DocumentElement]:
    """
    Prepare structured document elements for LLM extraction.

    This layer is intentionally conservative.

    It removes:
    - obvious table-of-contents pages
    - address/contact tables
    - obvious cover-page furniture
    - isolated administrative fragments

    It preserves:
    - headings
    - prose
    - substantive tables
    - quantitative content
    """
    if not elements:
        return []

    pages = _group_by_page(elements)
    retained: List[DocumentElement] = []

    for page_elements in pages:
        if _is_toc_page(page_elements):
            continue

        for element in page_elements:
            if _drop_element(element):
                continue

            retained.append(element)

    return sorted(
        retained,
        key=lambda element: (
            element.source_file,
            element.page_number,
            element.top,
            element.element_id,
        ),
    )


def _group_by_page(
    elements: List[DocumentElement],
) -> List[List[DocumentElement]]:
    groups: Dict[Tuple[str, int], List[DocumentElement]] = {}
    order: List[Tuple[str, int]] = []

    for element in sorted(
        elements,
        key=lambda item: (
            item.source_file,
            item.page_number,
            item.top,
            item.element_id,
        ),
    ):
        key = (
            element.source_file,
            element.page_number,
        )

        if key not in groups:
            groups[key] = []
            order.append(key)

        groups[key].append(element)

    return [groups[key] for key in order]


def _is_toc_page(
    elements: List[DocumentElement],
) -> bool:
    """
    Detect table-of-contents style pages from layout/text patterns.

    No filename/page-number assumptions are used.
    """
    if not elements:
        return False

    texts = [
        element.text.strip()
        for element in elements
        if element.text.strip()
    ]

    if len(texts) < 4:
        return False

    toc_like = sum(
        _looks_like_toc_line(text)
        for text in texts
    )

    heading_text = " ".join(texts).lower()

    explicit_toc = any(
        marker in heading_text
        for marker in (
            "table of contents",
            "contents",
        )
    )

    ratio = toc_like / len(texts)
    return explicit_toc or ratio >= 0.55


def _looks_like_toc_line(
    text: str,
) -> bool:
    lower = text.lower()

    if (
        "................................................................" in text
        or "…" in text
    ):
        return True

    # TOC lines frequently terminate in a page number.
    parts = text.replace("...", " ").split()

    if len(parts) >= 3:
        last = parts[-1].replace(".", "")
        if last.isdigit() and len(last) <= 4:
            return True

    # Generic section navigation.
    if lower.startswith(
        (
            "section i:",
            "section ii:",
            "section iii:",
            "section iv:",
            "section v:",
            "section vi:",
            "section vii:",
            "section viii:",
        )
    ):
        return True

    return False


def _drop_element(
    element: DocumentElement,
) -> bool:
    text = " ".join(element.text.split()).strip()

    if not text:
        return True

    if element.element_type == "TABLE":
        return _is_admin_table(element)

    if element.element_type == "HEADING":
        return _is_low_value_heading(text)

    return _is_low_value_prose(text)


def _is_admin_table(
    element: DocumentElement,
) -> bool:
    """
    Remove contact/address tables that don't contain meaningful
    business facts.
    """
    if element.table is None:
        return False

    text = " ".join(element.text.lower().split())

    administrative_signals = (
        "registered office",
        "corporate office",
        "contact person",
        "telephone and e-mail",
        "telephone",
        "email",
        "website",
        "company secretary",
        "registrar",
    )

    signal_count = sum(
        signal in text
        for signal in administrative_signals
    )

    numeric_values = sum(
        _contains_numeric_value(row)
        for row in element.table.rows
    )

    # Administrative contact tables with no meaningful numeric
    # business data are safe to omit from extraction.
    return (
        signal_count >= 2
        and numeric_values <= 1
    )


def _contains_numeric_value(
    row: List[str],
) -> bool:
    for cell in row:
        cleaned = (
            cell.replace(",", "")
            .replace(".", "")
            .replace("-", "")
            .replace(" ", "")
        )

        if cleaned.isdigit() and len(cleaned) >= 2:
            return True

    return False


def _is_low_value_heading(
    text: str,
) -> bool:
    lower = text.lower()

    if lower in {
        "prospectus",
        "index",
        "contents",
        "important notice",
    }:
        return True

    # Address/contact headings.
    if any(
        phrase in lower
        for phrase in (
            "registered office",
            "corporate office",
            "contact details",
            "telephone and e-mail",
        )
    ):
        return True

    return False


def _is_low_value_prose(
    text: str,
) -> bool:
    lower = text.lower()

    # QR / scanning / administrative instructions.
    administrative_patterns = (
        "please scan this qr code",
        "upi mandate end time",
        "for identification purposes only",
        "this page is intentionally left blank",
    )

    if any(
        pattern in lower
        for pattern in administrative_patterns
    ):
        return True

    # Very short isolated administrative fragments.
    if len(text) < 18:
        return True

    return False