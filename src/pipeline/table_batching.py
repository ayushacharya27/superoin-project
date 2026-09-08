from typing import List

from src.schemas.table import ReconstructedTableRow, TableContext

MAX_TABLE_TOKENS = 6000


def build_table_llm_contexts(
    tables: List[TableContext],
    max_tokens: int = MAX_TABLE_TOKENS,
) -> List[TableContext]:
    """
    Tables are the unit of batching.

    A table may contain many rows, but those rows are presented
    to the LLM together because headers/caption/unit provide the
    semantic context required for extraction.
    """
    contexts = []

    for table in tables:
        if table.deterministic:
            continue

        chunks = _split_table(table, max_tokens)
        contexts.extend(chunks)

    return contexts


def _split_table(
    table: TableContext,
    max_tokens: int,
) -> List[TableContext]:
    result = []
    current_rows = []
    current_tokens = _table_header_tokens(table)

    for row in table.rows:
        row_tokens = _row_tokens(row)

        if current_rows and current_tokens + row_tokens > max_tokens:
            result.append(
                table.model_copy(
                    update={"rows": current_rows}
                )
            )
            current_rows = []
            current_tokens = _table_header_tokens(table)

        current_rows.append(row)
        current_tokens += row_tokens

    if current_rows:
        result.append(
            table.model_copy(
                update={"rows": current_rows}
            )
        )

    return result


def _table_header_tokens(
    table: TableContext,
) -> int:
    text = " ".join(
        part
        for part in (
            table.caption,
            table.unit,
            *table.headers,
        )
        if part
    )
    return max(1, len(text) // 4)


def _row_tokens(
    row: ReconstructedTableRow,
) -> int:
    text = " ".join(
        [
            row.row_label or "",
            *row.values,
        ]
    )
    return max(1, len(text) // 4)