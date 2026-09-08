from typing import List

from config import settings
from src.schemas.context import CompactContext, ContextEvidence
from src.schemas.evidence import EvidenceUnit

TARGET_TOKENS = 2200


def build_document_chunks(
    evidence: List[EvidenceUnit],
    target_tokens: int = TARGET_TOKENS,
) -> List[CompactContext]:
    """
    Build large, structure-preserving extraction contexts.

    Unlike the old semantic-group approach, this is document-oriented:
    it follows source order and keeps nearby evidence together.
    """
    if not evidence:
        return []

    ordered = sorted(
        evidence,
        key=lambda item: (
            item.source_file,
            item.page_number,
            item.evidence_id,
        ),
    )

    chunks = []
    current: List[EvidenceUnit] = []
    current_tokens = 0
    current_source = None

    for item in ordered:
        item_tokens = _estimate_tokens(item)

        if current_source is None:
            current_source = item.source_file

        # Don't mix documents in one extraction packet.
        if current and item.source_file != current_source:
            chunks.append(
                _make_context(
                    current,
                    len(chunks) + 1,
                )
            )
            current = []
            current_tokens = 0
            current_source = item.source_file

        # Prefer to cut at evidence boundaries.
        if current and current_tokens + item_tokens > target_tokens:
            chunks.append(
                _make_context(
                    current,
                    len(chunks) + 1,
                )
            )
            current = []
            current_tokens = 0

        current.append(item)
        current_tokens += item_tokens

    if current:
        chunks.append(
            _make_context(
                current,
                len(chunks) + 1,
            )
        )

    return chunks


def _make_context(
    evidence: List[EvidenceUnit],
    index: int,
) -> CompactContext:
    source = evidence[0].source_file
    first_page = min(item.page_number for item in evidence)
    last_page = max(item.page_number for item in evidence)

    topic = f"{source} | pages {first_page}-{last_page}"

    return CompactContext(
        context_id=f"{_source_prefix(source)}-chunk-{index:03d}",
        topic=topic,
        evidence=[
            ContextEvidence(
                evidence_id=item.evidence_id,
                source_file=item.source_file,
                page_number=item.page_number,
                text=item.text,
            )
            for item in evidence
        ],
        estimated_tokens=sum(_estimate_tokens(item) for item in evidence),
    )


def _estimate_tokens(
    item: EvidenceUnit,
) -> int:
    metadata = f"[{item.evidence_id}] p.{item.page_number}"
    return max(
        1,
        (len(item.text) + len(metadata)) // settings.CHARS_PER_TOKEN,
    )


def _source_prefix(
    filename: str,
) -> str:
    return (
        filename.replace(".pdf", "")
        .replace(" ", "-")
        .replace("/", "-")
    )