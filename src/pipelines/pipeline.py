import os
from typing import Dict, Optional

from src.db.sqlite_db import (
    init_db,
    create_document,
    create_chunk,
    create_fact,
    get_document_by_filename,
    get_processed_chunk_keys,
)

from src.pipelines.ingestion import ingest_pdf
from src.pipelines.metadata import extract_global_metadata
from src.pipelines.extraction import extract_facts_from_chunk
from src.pipelines.validation import validate_fact_grounding
from src.pipelines.normalization import normalize_fact


def process_pdf(
    file_path: str,
    max_chunks: Optional[int] = None,
) -> Dict:
    """
    Process one PDF through the complete fact extraction pipeline.

    Pipeline:

        PDF
        ↓
        Ingestion
        ↓
        Global metadata
        ↓
        Chunk processing
        ↓
        Fact extraction
        ↓
        Grounding validation
        ↓
        Numeric normalization
        ↓
        SQLite persistence

    The pipeline is incremental/idempotent:
    if a document has already been processed, chunks that already
    exist in SQLite are skipped.

    max_chunks:
        Optional limit used during development/testing.
        If None, all chunks are processed.
    """

    # =========================================================
    # 1. Validate input
    # =========================================================

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"PDF not found: {file_path}"
        )

    if not file_path.lower().endswith(".pdf"):
        raise ValueError(
            f"Expected a PDF file: {file_path}"
        )

    # =========================================================
    # 2. Initialize database
    # =========================================================

    init_db()

    # =========================================================
    # 3. Ingest PDF
    # =========================================================

    print("\n[Pipeline] Starting ingestion...")

    chunks = ingest_pdf(file_path)

    print(
        f"[Pipeline] Received {len(chunks)} chunks."
    )

    # =========================================================
    # 4. Optional chunk limit
    # =========================================================

    if max_chunks is not None:

        if max_chunks <= 0:
            raise ValueError(
                "max_chunks must be greater than 0."
            )

        chunks = chunks[:max_chunks]

        print(
            f"[Pipeline] Limited processing to "
            f"{len(chunks)} chunks."
        )

    # =========================================================
    # 5. Extract global metadata
    # =========================================================

    print(
        "\n[Pipeline] Extracting global metadata..."
    )

    metadata = extract_global_metadata(chunks)

    print(
        f"[Pipeline] Document: "
        f"{metadata.document_title}"
    )

    print(
        f"[Pipeline] Entities: "
        f"{metadata.primary_entities}"
    )

    print(
        f"[Pipeline] Reporting period: "
        f"{metadata.reporting_period}"
    )

    # =========================================================
    # 6. Find or create document
    # =========================================================

    file_name = os.path.basename(file_path)

    existing_document = get_document_by_filename(
        file_name
    )

    if existing_document:

        document_id = existing_document.id

        processed_chunk_keys = (
            get_processed_chunk_keys(
                document_id
            )
        )

        print(
            f"\n[Pipeline] Existing document found "
            f"(ID={document_id})."
        )

        print(
            f"[Pipeline] Already processed chunks: "
            f"{len(processed_chunk_keys)}"
        )

    else:

        document_id = create_document(
            file_name=file_name,
            document_title=metadata.document_title,
            reporting_period=metadata.reporting_period,
        )

        processed_chunk_keys = set()

        print(
            f"\n[Pipeline] Created document ID: "
            f"{document_id}"
        )

    # =========================================================
    # 7. Process chunks
    # =========================================================

    total_facts = 0
    valid_facts = 0
    grounding_failures = 0
    skipped_chunks = 0
    newly_processed_chunks = 0

    for index, chunk in enumerate(
        chunks,
        start=1,
    ):

        page_number = chunk.metadata.get(
            "page_number"
        )

        chunk_index = chunk.metadata.get(
            "chunk_index"
        )

        # -----------------------------------------------------
        # Skip already processed chunks
        # -----------------------------------------------------

        if (
            page_number,
            chunk_index,
        ) in processed_chunk_keys:

            skipped_chunks += 1

            print(
                f"\n[Pipeline] Skipping already processed "
                f"chunk {index}/{len(chunks)} "
                f"(page={page_number}, "
                f"chunk={chunk_index})"
            )

            continue

        # -----------------------------------------------------
        # Process new chunk
        # -----------------------------------------------------

        newly_processed_chunks += 1

        print(
            f"\n[Pipeline] Processing chunk "
            f"{index}/{len(chunks)} "
            f"(page={page_number}, "
            f"chunk={chunk_index})"
        )

        # -----------------------------------------------------
        # Store chunk
        # -----------------------------------------------------

        chunk_id = create_chunk(
            document_id=document_id,
            page_number=page_number,
            chunk_index=chunk_index,
            content=chunk.page_content,
        )

        # -----------------------------------------------------
        # Extract facts
        # -----------------------------------------------------

        facts = extract_facts_from_chunk(
            chunk,
            metadata,
        )

        if not facts:

            print(
                "[Pipeline] No facts extracted."
            )

            continue

        # -----------------------------------------------------
        # Validate grounding
        # -----------------------------------------------------

        facts = validate_fact_grounding(
            facts,
            chunk,
        )

        # -----------------------------------------------------
        # Normalize and persist facts
        # -----------------------------------------------------

        for fact in facts:

            total_facts += 1

            # Normalize numerical values
            fact = normalize_fact(
                fact
            )

            # Track validation result
            if fact.status == "VALID":

                valid_facts += 1

            else:

                grounding_failures += 1

            # Save fact to SQLite
            create_fact(
                document_id=document_id,
                chunk_id=chunk_id,
                fact=fact,
            )

        print(
            f"[Pipeline] Extracted "
            f"{len(facts)} facts."
        )

    # =========================================================
    # 8. Create summary
    # =========================================================

    summary = {
        "document_id": document_id,
        "file_name": file_name,
        "document_title": metadata.document_title,
        "primary_entities": metadata.primary_entities,
        "reporting_period": metadata.reporting_period,
        "chunks_total": len(chunks),
        "chunks_processed": newly_processed_chunks,
        "chunks_skipped": skipped_chunks,
        "facts_extracted": total_facts,
        "valid_facts": valid_facts,
        "grounding_failures": grounding_failures,
    }

    # =========================================================
    # 9. Print summary
    # =========================================================

    print("\n==============================")
    print("PIPELINE COMPLETE")
    print("==============================")

    for key, value in summary.items():

        print(
            f"{key}: {value}"
        )

    return summary