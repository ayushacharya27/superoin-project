from typing import List, Optional, Tuple

import numpy as np

from config import settings
from src.schemas.context import (
    CompactContext,
    ContextEvidence,
)
from src.schemas.evidence import EvidenceUnit
from src.schemas.group import EvidenceGroup


GROUP_SIMILARITY_THRESHOLD = 0.58
DIFFERENT_SOURCE_BONUS = 0.08


def estimate_tokens(text: str) -> int:
    if not text:
        return 0

    return max(
        1,
        int(
            np.ceil(
                len(text)
                / settings.CHARS_PER_TOKEN
            )
        ),
    )


def pack_contexts(
    groups: List[EvidenceGroup],
) -> List[CompactContext]:
    if not groups:
        return []

    prepared = [
        (
            group,
            _prepare_evidence(group.evidence),
        )
        for group in groups
        if group.evidence
    ]

    centroids = {
        group.group_id: _group_centroid(group)
        for group, _ in prepared
    }

    unassigned = {
        group.group_id
        for group, _ in prepared
    }

    group_lookup = {
        group.group_id: (
            group,
            evidence,
        )
        for group, evidence in prepared
    }

    contexts = []
    context_number = 1

    while unassigned:
        seed_id = _choose_seed(
            unassigned,
            group_lookup,
        )

        seed_group, seed_evidence = group_lookup[seed_id]

        selected = [
            (
                seed_group,
                seed_evidence,
            )
        ]

        unassigned.remove(seed_id)
        selected_tokens = _estimate_groups_tokens(selected)

        while True:
            candidate = _best_candidate(
                seed_group=seed_group,
                selected_groups=selected,
                candidate_ids=unassigned,
                group_lookup=group_lookup,
                centroids=centroids,
            )

            if candidate is None:
                break

            candidate_group, candidate_evidence = group_lookup[candidate]

            candidate_tokens = _estimate_groups_tokens(
                selected + [
                    (
                        candidate_group,
                        candidate_evidence,
                    )
                ]
            )

            if candidate_tokens > settings.MAX_CONTEXT_TOKENS:
                break

            selected.append(
                (
                    candidate_group,
                    candidate_evidence,
                )
            )

            selected_tokens = candidate_tokens
            unassigned.remove(candidate)

        evidence = _select_context_evidence(selected)

        if not evidence:
            continue

        contexts.append(
            CompactContext(
                context_id=f"context-{context_number:05d}",
                topic=_make_context_topic(selected),
                evidence=[
                    ContextEvidence(
                        evidence_id=item.evidence_id,
                        source_file=item.source_file,
                        page_number=item.page_number,
                        text=item.text,
                    )
                    for item in evidence
                ],
                estimated_tokens=_estimate_context_tokens(evidence),
            )
        )

        context_number += 1

    return contexts


def _group_centroid(
    group: EvidenceGroup,
) -> np.ndarray:
    vectors = [
        item.embedding
        for item in group.evidence
        if item.embedding is not None
    ]

    if not vectors:
        raise ValueError(
            f"Group {group.group_id} contains no embeddings."
        )

    vector = np.mean(
        np.asarray(
            vectors,
            dtype=np.float32,
        ),
        axis=0,
    )

    norm = np.linalg.norm(vector)
    if norm <= 1e-12:
        return vector

    return vector / norm


def _choose_seed(
    ids,
    group_lookup,
) -> str:
    """
    Start with the largest evidence group.

    This tends to consume the most context first and prevents
    large groups being stranded at the end.
    """
    return max(
        ids,
        key=lambda group_id: len(group_lookup[group_id][1]),
    )


def _best_candidate(
    seed_group,
    selected_groups,
    candidate_ids,
    group_lookup,
    centroids,
) -> Optional[str]:
    if not candidate_ids:
        return None

    selected_sources = {
        item.source_file
        for _, evidence in selected_groups
        for item in evidence
    }

    best_id = None
    best_score = GROUP_SIMILARITY_THRESHOLD - 0.01

    for candidate_id in candidate_ids:
        candidate_group, candidate_evidence = group_lookup[candidate_id]
        candidate_centroid = centroids[candidate_id]

        # Compare against the strongest semantic match already
        # present in this context.
        similarities = [
            float(np.dot(centroids[selected_group.group_id], candidate_centroid))
            for selected_group, _ in selected_groups
        ]

        similarity = max(similarities)
        if similarity < GROUP_SIMILARITY_THRESHOLD:
            continue

        candidate_sources = {
            item.source_file
            for item in candidate_evidence
        }

        score = similarity
        if selected_sources and candidate_sources.isdisjoint(selected_sources):
            score += DIFFERENT_SOURCE_BONUS

        if score > best_score:
            best_score = score
            best_id = candidate_id

    return best_id


def _prepare_evidence(
    evidence: List[EvidenceUnit],
) -> List[EvidenceUnit]:
    seen = set()
    result = []

    for item in evidence:
        key = (
            _normalize_text(item.text),
            item.source_file,
            item.page_number,
        )

        if key in seen:
            continue

        seen.add(key)
        result.append(item)

    return _rank_evidence(result)


def _rank_evidence(
    evidence: List[EvidenceUnit],
) -> List[EvidenceUnit]:
    """
    Rank evidence by factual usefulness.

    We prefer evidence containing explicit numerical,
    temporal, unit, and entity signals.
    """
    def score(item):
        value = 0
        value += 4 * bool(item.number_hints)
        value += 3 * bool(item.unit_hint)
        value += 2 * bool(item.time_hint)
        value += 2 * bool(item.entity_hint)

        if item.evidence_type == "TABLE_ROW":
            value += 2
        if item.evidence_type == "PROSE":
            value += 1

        return value

    return sorted(
        evidence,
        key=lambda item: (
            -score(item),
            item.source_file,
            item.page_number,
            item.evidence_id,
        ),
    )


def _select_context_evidence(
    selected_groups,
) -> List[EvidenceUnit]:
    """Select high-value evidence while maintaining source diversity."""
    candidates = []

    for group, evidence in selected_groups:
        limit = settings.MAX_EVIDENCE_PER_GROUP
        for item in evidence[:limit]:
            candidates.append(item)

    # Prefer one pass of source-diverse evidence first.
    by_source = {}
    for item in candidates:
        by_source.setdefault(
            item.source_file,
            [],
        ).append(item)

    sources = list(by_source)
    result = []
    seen = set()

    max_rounds = max(
        (len(items) for items in by_source.values()),
        default=0,
    )

    for index in range(max_rounds):
        for source in sources:
            items = by_source[source]
            if index >= len(items):
                continue

            item = items[index]
            if item.evidence_id in seen:
                continue

            seen.add(item.evidence_id)
            result.append(item)

    # Enforce the final token budget.
    packed = []
    tokens = 0

    for item in result:
        item_tokens = _evidence_tokens(item)

        if (
            packed
            and tokens + item_tokens > settings.MAX_CONTEXT_TOKENS
        ):
            break

        packed.append(item)
        tokens += item_tokens

    return packed


def _estimate_groups_tokens(
    groups,
) -> int:
    evidence = []
    for _, items in groups:
        evidence.extend(items)

    return _estimate_context_tokens(evidence)


def _estimate_context_tokens(
    evidence,
) -> int:
    return sum(
        _evidence_tokens(item)
        for item in evidence
    )


def _evidence_tokens(
    item: EvidenceUnit,
) -> int:
    metadata = (
        f"[{item.evidence_id}] "
        f"{item.source_file} "
        f"p.{item.page_number}"
    )

    return (
        estimate_tokens(item.text)
        + estimate_tokens(metadata)
        + 5
    )


def _make_context_topic(
    selected_groups,
) -> str:
    labels = [
        group.topic_label
        for group, _ in selected_groups
    ]

    if len(labels) == 1:
        return labels[0]

    return " | ".join(labels[:3])


def _normalize_text(value: Optional[str]) -> str:
    if not value:
        return ""

    return " ".join(value.lower().split())