from typing import List, Optional

import numpy as np
from sklearn.cluster import AgglomerativeClustering

from src.schemas.evidence import EvidenceUnit
from src.schemas.group import EvidenceGroup

DEFAULT_SIMILARITY_THRESHOLD = 0.72
STRUCTURE_PENALTY = 0.20


def group_evidence(
    evidence: List[EvidenceUnit],
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> List[EvidenceGroup]:
    if not evidence:
        return []

    cleaned = _deduplicate_exact(evidence)
    usable = [item for item in cleaned if not _is_low_information(item)]

    if not usable:
        return []

    missing_embeddings = [
        item.evidence_id for item in usable if item.embedding is None
    ]

    if missing_embeddings:
        raise ValueError(
            "All evidence must be embedded before grouping. "
            f"Missing embeddings: {len(missing_embeddings)}"
        )

    vectors = np.asarray(
        [item.embedding for item in usable],
        dtype=np.float32,
    )
    vectors = _normalize(vectors)

    similarity = np.clip(vectors @ vectors.T, -1.0, 1.0)
    distances = 1.0 - similarity

    for i in range(len(usable)):
        for j in range(i + 1, len(usable)):
            if not _structurally_compatible(usable[i], usable[j]):
                distances[i, j] += STRUCTURE_PENALTY
                distances[j, i] += STRUCTURE_PENALTY

    np.fill_diagonal(distances, 0.0)

    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=1.0 - similarity_threshold,
        metric="precomputed",
        linkage="average",
    )

    labels = clustering.fit_predict(distances)

    grouped = {}
    for item, label in zip(usable, labels):
        grouped.setdefault(int(label), []).append(item)

    groups = []
    for group_number, members in enumerate(grouped.values(), start=1):
        members.sort(
            key=lambda item: (
                item.source_file,
                item.page_number,
                item.evidence_id,
            )
        )

        groups.append(
            EvidenceGroup(
                group_id=f"group-{group_number:05d}",
                topic_label=_make_topic_label(members),
                evidence=members,
                representative_evidence_ids=[
                    item.evidence_id for item in members[:5]
                ],
            )
        )

    groups.sort(
        key=lambda group: (
            -len(group.evidence),
            group.group_id,
        )
    )

    return groups


def _deduplicate_exact(
    evidence: List[EvidenceUnit],
) -> List[EvidenceUnit]:
    """
    Remove repeated extraction artifacts without destroying
    cross-document provenance.

    The same fact appearing in two PDFs must remain twice because
    the second occurrence may be evidence of corroboration.
    """
    seen = set()
    result = []

    for item in evidence:
        key = (
            _normalize_text(item.text),
            item.evidence_type,
            _normalize_text(item.section_heading),
            _normalize_text(item.table_caption),
            _normalize_text(item.row_label),
            _normalize_text(item.column_label),
            item.source_file,
            item.page_number,
        )

        if key in seen:
            continue

        seen.add(key)
        result.append(item)

    return result


def _is_low_information(item: EvidenceUnit) -> bool:
    text = _normalize_text(item.text)

    if not text:
        return True

    if "annual report" in text and len(text) < 120:
        return True

    if "cin:" in text and len(text) < 250:
        return True

    if "all amounts in indian rupees" in text and len(text) < 250:
        return True

    if item.evidence_type == "TABLE_ROW" and len(text) < 20:
        return True

    return False


def _structural_class(item: EvidenceUnit) -> str:
    if item.evidence_type == "HEADING":
        return "heading"

    has_number = bool(item.number_hints)
    has_unit = bool(item.unit_hint)
    has_time = bool(item.time_hint)

    if has_number and (has_unit or has_time):
        return "quantitative"

    if has_number:
        return "numeric"

    if item.evidence_type == "TABLE_ROW":
        return "table"

    return "prose"


def _structurally_compatible(
    a: EvidenceUnit,
    b: EvidenceUnit,
) -> bool:
    class_a = _structural_class(a)
    class_b = _structural_class(b)

    if class_a == class_b:
        return True

    quantitative_classes = {
        "quantitative",
        "numeric",
    }

    if class_a in quantitative_classes and class_b in quantitative_classes:
        return True

    if {class_a, class_b} == {"table", "quantitative"}:
        return True

    if {class_a, class_b} == {"table", "numeric"}:
        return True

    return False


def _normalize_text(value: Optional[str]) -> str:
    if not value:
        return ""
    return " ".join(value.lower().split())


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(
        vectors,
        axis=1,
        keepdims=True,
    )
    return vectors / np.maximum(norms, 1e-12)


def _make_topic_label(evidence: List[EvidenceUnit]) -> str:
    headings = [
        item.section_heading
        for item in evidence
        if item.section_heading
    ]

    if headings:
        counts = {}
        for heading in headings:
            counts[heading] = counts.get(heading, 0) + 1
        return max(counts, key=counts.get)

    captions = [
        item.table_caption
        for item in evidence
        if item.table_caption
    ]

    if captions:
        return captions[0]

    return evidence[0].text[:100]