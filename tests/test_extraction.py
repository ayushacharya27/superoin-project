from src.pipelines.ingestion import ingest_pdf
from src.pipelines.metadata import extract_global_metadata
from src.pipelines.extraction import extract_facts_from_chunk
from src.pipelines.validation import validate_fact_grounding


PDF_PATH = (
    "data/starter_pdfs/"
    "01-delhivery-prospectus-2022-excerpt.pdf"
)


def main():

    # ---------------------------------------------------------
    # Step 1: Ingest
    # ---------------------------------------------------------

    chunks = ingest_pdf(PDF_PATH)

    print(f"\nTotal chunks: {len(chunks)}")

    # ---------------------------------------------------------
    # Step 2: Global metadata
    # ---------------------------------------------------------

    metadata = extract_global_metadata(chunks)

    print("\n===== GLOBAL METADATA =====")
    print(f"Title: {metadata.document_title}")
    print(f"Entities: {metadata.primary_entities}")
    print(f"Period: {metadata.reporting_period}")

    # ---------------------------------------------------------
    # Step 3: Test extraction on ONE chunk
    # ---------------------------------------------------------

    chunk = chunks[20]

    print("\n===== TEST CHUNK =====")
    print(f"Page: {chunk.metadata.get('page_number')}")
    print(f"Chunk: {chunk.metadata.get('chunk_index')}")
    print(chunk.page_content[:1000])

    # ---------------------------------------------------------
    # Step 4: Extract facts
    # ---------------------------------------------------------

    facts = extract_facts_from_chunk(
        chunk,
        metadata,
    )

    facts = validate_fact_grounding(
    facts,
    chunk,
)

    # ---------------------------------------------------------
    # Step 5: Display facts
    # ---------------------------------------------------------

    print("\n===== EXTRACTED FACTS =====")

    for i, fact in enumerate(facts, start=1):

        print(f"\nFact {i}")
        print(f"Entity: {fact.entity}")
        print(f"Attribute: {fact.attribute}")
        print(f"Raw Value: {fact.raw_value}")
        print(f"Unit: {fact.unit}")
        print(f"Time: {fact.time}")
        print(f"Scope: {fact.scope}")
        print(f"Qualifier: {fact.qualifier}")
        print(f"Quote: {fact.exact_quote}")
        print(f"Confidence: {fact.confidence_score}")
        print(f"Source: {fact.source_file}")
        print(f"Page: {fact.page_number}")
        print(f"Chunk: {fact.chunk_index}")
        print(f"Status: {fact.status}")


if __name__ == "__main__":
    main()