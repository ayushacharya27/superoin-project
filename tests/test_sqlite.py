import os

from src.db.sqlite_db import (
    init_db,
    create_document,
    create_chunk,
    create_fact,
    get_valid_facts,
)

from src.schemas.fact_schema import ExtractedFact


def main():

    # ---------------------------------------------------------
    # 1. Initialize database
    # ---------------------------------------------------------

    init_db()

    print("Database initialized.")

    # ---------------------------------------------------------
    # 2. Create test document
    # ---------------------------------------------------------

    document_id = create_document(
        file_name="test.pdf",
        document_title="Test Document",
        reporting_period="FY2025",
    )

    print(
        f"Created document with ID: {document_id}"
    )

    # ---------------------------------------------------------
    # 3. Create test chunk
    # ---------------------------------------------------------

    chunk_id = create_chunk(
        document_id=document_id,
        page_number=1,
        chunk_index=0,
        content="Revenue was $1.2 billion in FY2025.",
    )

    print(
        f"Created chunk with ID: {chunk_id}"
    )

    # ---------------------------------------------------------
    # 4. Create test fact
    # ---------------------------------------------------------

    fact = ExtractedFact(
        entity="Example Company",
        attribute="Revenue",
        raw_value="1.2 billion",
        unit="USD billion",
        time="FY2025",
        scope=None,
        qualifier=None,
        exact_quote="Revenue was $1.2 billion in FY2025.",
        confidence_score=10,
        normalized_value=1_200_000_000,
        status="VALID",
    )

    fact_id = create_fact(
        document_id=document_id,
        chunk_id=chunk_id,
        fact=fact,
    )

    print(
        f"Created fact with ID: {fact_id}"
    )

    # ---------------------------------------------------------
    # 5. Retrieve facts
    # ---------------------------------------------------------

    facts = get_valid_facts()

    print("\n===== VALID FACTS =====")

    for stored_fact in facts:

        print(
            f"\nID: {stored_fact.id}"
        )

        print(
            f"Entity: {stored_fact.entity}"
        )

        print(
            f"Attribute: {stored_fact.attribute}"
        )

        print(
            f"Value: {stored_fact.raw_value}"
        )

        print(
            f"Normalized: {stored_fact.normalized_value}"
        )

        print(
            f"Time: {stored_fact.time}"
        )

        print(
            f"Status: {stored_fact.status}"
        )


if __name__ == "__main__":
    main()