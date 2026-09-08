from typing import List, Optional

from src.schemas.document import DocumentElement

# Maximum vertical gap, in PDF points, between visual prose lines
# that may belong to the same paragraph.
MAX_PARAGRAPH_GAP = 12.0

# A large horizontal reset often indicates a new visual block.
MAX_HORIZONTAL_RESET = 120.0


def clean_document_elements(
    elements: List[DocumentElement],
) -> List[DocumentElement]:
    """
    Convert low-level visual elements into useful document blocks.

    Rules:
    - tables are always preserved
    - headings are always preserved unless obvious noise
    - prose is merged only when consecutive lines are visually close
      and belong to the same page
    - obvious PDF furniture is removed
    """
    ordered = sorted(
        elements,
        key=lambda element: (
            element.source_file,
            element.page_number,
            element.top,
            element.element_id,
        ),
    )

    cleaned: List[DocumentElement] = []
    prose_buffer: List[DocumentElement] = []

    def flush_prose():
        if not prose_buffer:
            return

        merged = _merge_prose(prose_buffer)
        if merged is not None:
            cleaned.append(merged)

        prose_buffer.clear()

    previous_prose: Optional[DocumentElement] = None

    for element in ordered:
        if _should_drop(element):
            continue

        if element.element_type != "PROSE":
            flush_prose()
            previous_prose = None
            cleaned.append(element)
            continue

        if previous_prose is None:
            prose_buffer.append(element)
            previous_prose = element
            continue

        if _can_merge(previous_prose, element):
            prose_buffer.append(element)
        else:
            flush_prose()
            prose_buffer.append(element)

        previous_prose = element

    flush_prose()

    return sorted(
        cleaned,
        key=lambda element: (
            element.source_file,
            element.page_number,
            element.top,
            element.element_id,
        ),
    )


def _can_merge(
    previous: DocumentElement,
    current: DocumentElement,
) -> bool:
    """
    Determine whether two visual prose lines belong to the same
    paragraph-sized block.
    """
    if previous.source_file != current.source_file:
        return False

    if previous.page_number != current.page_number:
        return False

    vertical_gap = current.top - previous.top

    if vertical_gap < 0:
        return False

    if vertical_gap > MAX_PARAGRAPH_GAP:
        return False

    # A likely new section/list item should remain separate.
    current_text = current.text.strip()
    if _looks_like_new_block(current_text):
        return False

    return True


def _merge_prose(
    elements: List[DocumentElement],
) -> Optional[DocumentElement]:
    if not elements:
        return None

    meaningful = [
        element
        for element in elements
        if element.text.strip()
    ]

    if not meaningful:
        return None

    first = meaningful[0]
    text = _join_lines([element.text for element in meaningful])

    if not text:
        return None

    evidence_ids = []
    for element in meaningful:
        evidence_ids.extend(element.evidence_ids)

    return DocumentElement(
        element_id=(
            f"{first.source_file}:"
            f"{first.page_number}:"
            f"merged:{first.element_id}"
        ),
        source_file=first.source_file,
        page_number=first.page_number,
        element_type="PROSE",
        text=text,
        heading=None,
        table=None,
        evidence_ids=_unique(evidence_ids),
        # CRITICAL: retain the original reading position.
        top=first.top,
    )


def _join_lines(
    lines: List[str],
) -> str:
    result = ""

    for raw_line in lines:
        current = " ".join(raw_line.split())

        if not current:
            continue

        if not result:
            result = current
            continue

        # PDF extraction can split a word across visual lines.
        if result.endswith("-"):
            result += current
            continue

        if current.startswith((".", ",", ";", ":", ")", "%")):
            result += current
            continue

        result += " " + current

    return result.strip()


def _looks_like_new_block(
    text: str,
) -> bool:
    if not text:
        return True

    stripped = text.strip()

    # Numbered sections / list items.
    if len(stripped) >= 2:
        if stripped[0].isdigit() and stripped[1] in {".", ")"}:
            return True

    # Common bullet markers.
    if stripped.startswith(
        (
            "•",
            "▪",
            "●",
            "–",
            "—",
            "- ",
        )
    ):
        return True

    # A heading-looking line should not be swallowed into prose.
    if _looks_like_heading_text(stripped):
        return True

    return False


def _should_drop(
    element: DocumentElement,
) -> bool:
    text = " ".join(element.text.split()).strip()

    if not text:
        return True

    if element.element_type == "TABLE":
        return False

    if element.element_type == "HEADING":
        return _is_noise_heading(text)

    return _is_noise_prose(text)


def _is_noise_heading(
    text: str,
) -> bool:
    lower = text.lower()
    return lower in {
        "table of contents",
        "contents",
        "index",
    }


def _is_noise_prose(
    text: str,
) -> bool:
    lower = text.lower()

    if lower.startswith(
        (
            "cin:",
            "registered office:",
            "corporate identity number:",
            "page no.",
            "page:",
        )
    ):
        return True

    if lower in {
        "www.delhivery.com",
        "www.bseindia.com",
        "www.nseindia.com",
        "this page is intentionally left blank",
    }:
        return True

    if text.isdigit() and len(text) <= 4:
        return True

    return False


def _looks_like_heading_text(
    text: str,
) -> bool:
    if len(text) > 140:
        return False

    words = text.split()
    if len(words) > 14:
        return False

    if text.endswith("."):
        return False

    alpha = [char for char in text if char.isalpha()]
    if not alpha:
        return False

    uppercase_ratio = sum(char.isupper() for char in alpha) / len(alpha)

    if uppercase_ratio >= 0.75:
        return True

    if text[:1].isdigit() and len(text) <= 100:
        return True

    return False


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