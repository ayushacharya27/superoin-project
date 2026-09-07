from src.pipelines.ingestion import ingest_pdf
from src.pipelines.metadata import extract_global_metadata


PDF_PATH = "data/starter_pdfs/01-delhivery-prospectus-2022-excerpt.pdf"


def main():
    # Step 1: Ingest PDF
    chunks = ingest_pdf(PDF_PATH)

    print(f"\nTotal chunks: {len(chunks)}")

    # Step 2: Extract global metadata
    metadata = extract_global_metadata(chunks)

    # Step 3: Display result
    print("\n===== GLOBAL METADATA =====")

    print(f"Document Title: {metadata.document_title}")

    print("\nPrimary Entities:")
    for entity in metadata.primary_entities:
        print(f"  - {entity}")

    print(f"\nReporting Period: {metadata.reporting_period}")


if __name__ == "__main__":
    main()