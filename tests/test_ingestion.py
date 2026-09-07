from src.pipelines.ingestion import ingest_pdf


PDF_PATH = "data/starter_pdfs/01-delhivery-prospectus-2022-excerpt.pdf"


if __name__ == "__main__":

    chunks = ingest_pdf(PDF_PATH)

    print("\n" + "=" * 80)
    print("FIRST 3 CHUNKS")
    print("=" * 80)

    for chunk in chunks[:3]:

        print("\n--- METADATA ---")
        print(chunk.metadata)

        print("\n--- CONTENT ---")
        print(chunk.page_content[:1500])

        print("\n" + "-" * 80)