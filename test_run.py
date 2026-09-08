from pathlib import Path

from src.llm import FactExtractor
from src.pipeline import (
    build_structured_chunks,
    clean_document_elements,
    extract_document_elements,
)
from src.pipeline.llm_filter import filter_for_llm


PDF_PATH = (
    Path("data/starter_pdfs")
    / "01-delhivery-prospectus-2022-excerpt.pdf"
)


def main():
    raw = extract_document_elements(
        str(PDF_PATH)
    )

    clean = clean_document_elements(raw)

    filtered = filter_for_llm(clean)

    chunks = build_structured_chunks(
        filtered,
        target_tokens=2000,
    )

    context = chunks[0]

    print("=" * 80)
    print("ONE MISTRAL API CALL")
    print("=" * 80)
    print(
        f"Context: {context.context_id}"
    )
    print(
        f"Tokens: {context.estimated_tokens}"
    )
    print(
        f"Evidence elements: "
        f"{len(context.evidence)}"
    )

    extractor = FactExtractor()

    facts = extractor.extract(
        context
    )

    print()
    print("=" * 80)
    print("RESULT")
    print("=" * 80)

    print(
        f"Facts: {len(facts)}"
    )

    for fact in facts:
        print()
        print(
            f"[{fact.status}] "
            f"{fact.entity} | "
            f"{fact.attribute}"
        )
        print(
            f"value={fact.raw_value} "
            f"unit={fact.unit}"
        )
        print(
            f"time={fact.time} "
            f"scope={fact.scope}"
        )
        print(
            f"confidence="
            f"{fact.extraction_confidence}"
        )
        print(
    f"evidence_id={fact.evidence_id} "
    f"table_ref={fact.table_ref} "
    f"row={fact.row_index} "
    f"column={fact.column_index}"
)
        print(
            f"quote={fact.exact_quote[:300]}"
        )


if __name__ == "__main__":
    main()