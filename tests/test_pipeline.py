from src.pipelines.pipeline import process_pdf

PDF_PATH = (
    "data/starter_pdfs/"
    "01-delhivery-prospectus-2022-excerpt.pdf"
)


def main():
    summary = process_pdf(
        PDF_PATH,
        max_chunks=5
    )

    print("\n===== FINAL SUMMARY =====")

    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()