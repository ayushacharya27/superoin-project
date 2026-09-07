from src.schemas.fact_schema import ExtractedFact
from src.pipelines.normalization import normalize_fact


def main():

    test_facts = [
        ExtractedFact(
            entity="DELHIVERY LIMITED",
            attribute="Equity share capital",
            raw_value="216.68",
            unit="₹ million",
            time="December 31, 2021",
            exact_quote="Equity share capital|216.68",
            confidence_score=10,
        ),
        ExtractedFact(
            entity="Example Company",
            attribute="Revenue",
            raw_value="1.2 billion",
            unit="USD billion",
            time="FY2025",
            exact_quote="Revenue was 1.2 billion",
            confidence_score=10,
        ),
        ExtractedFact(
            entity="Example Company",
            attribute="Growth",
            raw_value="15%",
            unit="percentage",
            time="FY2025",
            exact_quote="Growth was 15%",
            confidence_score=10,
        ),
    ]

    print("\n===== NORMALIZATION TEST =====")

    for fact in test_facts:

        normalize_fact(fact)

        print("\nFact")
        print(f"Raw value: {fact.raw_value}")
        print(f"Unit: {fact.unit}")
        print(f"Normalized: {fact.normalized_value}")


if __name__ == "__main__":
    main()