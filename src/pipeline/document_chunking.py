from typing import List, Optional

from config import settings
from src.schemas.context import CompactContext, ContextEvidence
from src.schemas.document import DocumentElement

DEFAULT_TARGET_TOKENS = 6500


def estimate_tokens(text: str) -> int:
    """
    Lightweight token estimate.

    We intentionally use a cheap approximation here because chunking is
    a pre-processing step. The actual model tokenizer is not required.
    """
    if not text:
        return 0

    return max(1, len(text) // settings.CHARS_PER_TOKEN)


def _serialize_table(element: DocumentElement) -> str:
    table = element.table

    if table is None:
        return ""

    lines = [
        "[TABLE]",
        f"TABLE_REF: {element.element_id}",
        f"PAGE: {element.page_number}",
    ]

    if table.caption:
        lines.append(f"CAPTION: {table.caption}")

    if table.unit:
        lines.append(f"UNIT: {table.unit}")

    for column_index, header in enumerate(table.headers):
        lines.append(f"HEADER {column_index}: {header}")

    for row_index, row in enumerate(table.rows):
        lines.append(f"ROW {row_index}:")
        for column_index, value in enumerate(row):
            lines.append(f"CELL {column_index}: {value}")

    lines.append("[/TABLE]")

    return "\n".join(lines)


def _serialize_element(element: DocumentElement) -> str:
    if element.element_type == "TABLE":
        return _serialize_table(element)

    if element.element_type == "HEADING":
        return f"[HEADING] {element.text}"

    return f"[PROSE] {element.text}"


def _element_to_context_evidence(
    element: DocumentElement,
) -> ContextEvidence:
    """
    Converts a parsed document element into context evidence while
    preserving enough structured information for deterministic grounding.
    """
    if element.element_type == "TABLE" and element.table is not None:
        return ContextEvidence(
            evidence_id=element.element_id,
            source_file=element.source_file,
            page_number=element.page_number,
            text=_serialize_element(element),
            element_type="TABLE",
            table_ref=element.element_id,
            table_headers=list(element.table.headers),
            table_rows=[list(row) for row in element.table.rows],
            table_evidence_ids=list(element.table.evidence_ids),
        )

    return ContextEvidence(
        evidence_id=element.element_id,
        source_file=element.source_file,
        page_number=element.page_number,
        text=_serialize_element(element),
        element_type=element.element_type,
    )


def _build_context(
    elements: List[DocumentElement],
    context_index: int,
) -> CompactContext:
    evidence = [
        _element_to_context_evidence(element)
        for element in elements
    ]

    serialized = "\n\n".join(item.text for item in evidence)

    return CompactContext(
        context_id=f"context-{context_index:05d}",
        topic="document",
        evidence=evidence,
        estimated_tokens=estimate_tokens(serialized),
    )


def build_structured_chunks(
    elements: List[DocumentElement],
    target_tokens: int = DEFAULT_TARGET_TOKENS,
) -> List[CompactContext]:
    """
    Build ordered, bounded contexts without mixing documents.

    Tables are kept intact and their structured rows are carried forward
    into ContextEvidence for deterministic grounding after extraction.
    """
    if not elements:
        return []

    contexts: List[CompactContext] = []

    current_elements: List[DocumentElement] = []
    current_tokens = 0
    context_index = 0
    current_source: Optional[str] = None

    def flush() -> None:
        nonlocal current_elements
        nonlocal current_tokens
        nonlocal context_index

        if not current_elements:
            return

        contexts.append(
            _build_context(
                current_elements,
                context_index,
            )
        )

        context_index += 1
        current_elements = []
        current_tokens = 0

    for element in elements:
        element_text = _serialize_element(element)
        element_tokens = estimate_tokens(element_text)

        # Never mix source PDFs.
        if (
            current_elements
            and current_source != element.source_file
        ):
            flush()

        current_source = element.source_file

        # Keep an individual oversized table/element intact.
        if element_tokens > target_tokens:
            flush()

            current_elements = [element]
            current_tokens = element_tokens

            flush()
            current_source = None
            continue

        if not current_elements:
            current_elements = [element]
            current_tokens = element_tokens
            continue

        if current_tokens + element_tokens > target_tokens:
            flush()

            current_elements = [element]
            current_tokens = element_tokens
            continue

        current_elements.append(element)
        current_tokens += element_tokens

    flush()

    return contexts