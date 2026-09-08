import os
from typing import List

from src.llm.extractor import FactExtractor
from src.matcher.resolve import FactMatcher
from src.pipeline.document_chunking import build_structured_chunks
from src.pipeline.document_cleanup import clean_document_elements
from src.pipeline.document_structure import extract_document_elements
from src.pipeline.llm_filter import filter_for_llm
from src.schemas.extraction import ExtractedFact

PDFS = [
    "data/starter_pdfs/01-delhivery-prospectus-2022-excerpt.pdf",
    "data/starter_pdfs/02-delhivery-annual-report-fy24-excerpt.pdf",
    "data/starter_pdfs/03-delhivery-q4-fy24-earnings-presentation.pdf",
]


def process_one_pdf(
    pdf_path: str, extractor: FactExtractor
) -> List[ExtractedFact]:
    print("\n" + "=" * 80)
    print(f"PROCESSING: {os.path.basename(pdf_path)}")
    print("=" * 80)

    raw = extract_document_elements(pdf_path)
    clean = clean_document_elements(raw)
    filtered = filter_for_llm(clean)

    contexts = build_structured_chunks(
        filtered,
        target_tokens=2000,
    )

    if not contexts:
        print("[Demo] No contexts produced.")
        return []

    # Exactly ONE context from this PDF for demonstration.
    context = contexts[0]

    print(
        f"[Demo] Using context={context.context_id} "
        f"tokens={context.estimated_tokens} "
        f"evidence={len(context.evidence)}"
    )

    facts = extractor.extract(context)
    valid = [fact for fact in facts if fact.status == "VALID"]

    print(f"[Demo] Extracted={len(facts)} VALID={len(valid)}")
    return valid


def main() -> None:
    for pdf in PDFS:
        if not os.path.exists(pdf):
            print(f"[Demo] Missing PDF: {pdf}")
            return

    extractor = FactExtractor()
    all_facts: List[ExtractedFact] = []

    for pdf_path in PDFS:
        facts = process_one_pdf(pdf_path, extractor)
        all_facts.extend(facts)

    print("\n" + "=" * 80)
    print("EMBEDDING + MATCHING")
    print("=" * 80)
    print(f"[Demo] Total VALID facts: {len(all_facts)}")

    matcher = FactMatcher(similarity_threshold=0.75)
    results = matcher.find_and_classify(all_facts)

    print(f"\n[Demo] Candidate relationships: {len(results)}")

    for index, (candidate, relationship) in enumerate(results[:30], start=1):
        fact_a = candidate.fact_a
        fact_b = candidate.fact_b

        print("\n" + "-" * 80)
        print(f"PAIR {index}")
        print(f"Similarity: {candidate.similarity:.3f}")
        print(f"A: {fact_a.entity} | {fact_a.attribute} | {fact_a.raw_value}")
        print(f"B: {fact_b.entity} | {fact_b.attribute} | {fact_b.raw_value}")
        print(f"Relationship: {relationship['status']}")
        print(f"Reasoning: {relationship['reasoning']}")

    print("\n" + "=" * 80)
    print("DEMO COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()