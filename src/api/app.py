import os
import shutil
import tempfile
import uuid
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from src.llm.extractor import FactExtractor
from src.llm.semantic_selector import SemanticSelector
from src.matcher.resolve import FactMatcher
from src.pipeline.document_chunking import build_structured_chunks
from src.pipeline.document_cleanup import clean_document_elements
from src.pipeline.document_structure import extract_document_elements
from src.pipeline.fact_canonicalizer import canonicalize_facts
from src.pipeline.llm_filter import filter_for_llm
from src.schemas.context import CompactContext
from src.schemas.extraction import ExtractedFact

app = FastAPI(
    title="Fact Knowledge Layer",
    version="0.1.0",
    description="PDF fact extraction and cross-document relationship analysis.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

extractor = FactExtractor()
matcher = FactMatcher(similarity_threshold=0.75)
semantic_selector = SemanticSelector()

ANALYSES: Dict[str, Dict[str, Any]] = {}


class AskRequest(BaseModel):
    question: str
    analysis_id: str


def fact_to_dict(fact: Any) -> Dict[str, Any]:
    return {
        "entity": getattr(fact, "entity", "") or "",
        "attribute": getattr(fact, "attribute", "") or "",
        "raw_value": getattr(fact, "raw_value", "") or "",
        "unit": getattr(fact, "unit", None),
        "normalized_value": getattr(fact, "normalized_value", None),
        "time": getattr(fact, "time", None),
        "scope": getattr(fact, "scope", None),
        "qualifier": getattr(fact, "qualifier", None),
        "exact_quote": getattr(fact, "exact_quote", "") or "",
        "evidence_id": getattr(fact, "evidence_id", "") or "",
        "table_ref": getattr(fact, "table_ref", None),
        "row_index": getattr(fact, "row_index", None),
        "column_index": getattr(fact, "column_index", None),
        "column_name": getattr(fact, "column_name", None),
        "extraction_confidence": getattr(fact, "extraction_confidence", 5),
        "status": getattr(fact, "status", "VALID"),
    }


def relationship_to_dict(
    pair: Any,
    classification: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "similarity": round(pair.similarity, 3),
        "status": classification["status"],
        "reasoning": classification["reasoning"],
        "confidence": classification["confidence"],
        "resolution_method": classification["resolution_method"],
        "fact_a": fact_to_dict(pair.fact_a),
        "fact_b": fact_to_dict(pair.fact_b),
    }


def _context_keyword_score(
    context: CompactContext,
    keywords: List[str],
) -> int:
    """
    Score a context against semantic keywords while rewarding numeric density.
    """
    if not keywords:
        return 0

    text_parts: List[str] = []

    for evidence in context.evidence:
        text_parts.append(evidence.text or "")

        if getattr(evidence, "table_headers", None):
            text_parts.extend(evidence.table_headers)

        if getattr(evidence, "table_rows", None):
            for row in evidence.table_rows:
                text_parts.extend(row)

    text = " ".join(text_parts).lower()
    score = 0

    for keyword in keywords:
        cleaned = keyword.lower().strip()
        if cleaned and cleaned in text:
            score += 1

    # Dense numeric tokens correlate with financial tables/excerpts
    numeric_count = sum(1 for token in text.split() if any(c.isdigit() for c in token))
    score += min(numeric_count // 3, 10)

    return score


def _select_demo_contexts(
    contexts: List[CompactContext],
    keywords: List[str],
    limit: int = 1,
) -> List[CompactContext]:
    """
    Select top-N contexts per document based on semantic relevance scores.
    """
    if not contexts:
        return []

    ranked = sorted(
        enumerate(contexts),
        key=lambda item: (
            _context_keyword_score(item[1], keywords),
            -item[0],
        ),
        reverse=True,
    )

    selected: List[CompactContext] = []

    for index, context in ranked[:limit]:
        score = _context_keyword_score(context, keywords)
        print(
            f"[Demo] Selected context {context.context_id} "
            f"(index={index}, keyword_score={score})"
        )
        selected.append(context)

    return selected


def process_pdf(
    pdf_path: str,
    filename: str,
) -> Dict[str, Any]:
    """
    Parse a PDF into structural elements and build structured bounded chunks.
    Does not execute LLM extraction.
    """
    raw = extract_document_elements(pdf_path)
    clean = clean_document_elements(raw)
    filtered = filter_for_llm(clean)

    contexts = build_structured_chunks(
        filtered,
        target_tokens=2000,
    )

    return {
        "filename": filename,
        "raw_elements": len(raw),
        "clean_elements": len(clean),
        "filtered_elements": len(filtered),
        "contexts": contexts,
    }


@app.get("/api/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "service": "fact-knowledge-layer",
        "llm_provider": "mistral",
        "semantic_selector": "gemini",
        "demo_mode": True,
        "contexts_per_pdf": 2,
        "selection": "Gemini-derived semantic keywords; top 2 contexts per PDF",
    }


@app.post("/api/analyze")
async def analyze(
    files: List[UploadFile] = File(...),
) -> Dict[str, Any]:
    if not files:
        raise HTTPException(
            status_code=400,
            detail="Upload at least one PDF.",
        )

    analysis_id = str(uuid.uuid4())

    with tempfile.TemporaryDirectory() as temp_dir:
        # STEP 1: Parse and structure all uploaded PDFs
        parsed_documents: List[Dict[str, Any]] = []

        for upload in files:
            filename = upload.filename or "document.pdf"

            if not filename.lower().endswith(".pdf"):
                raise HTTPException(
                    status_code=400,
                    detail=f"{filename} is not a PDF.",
                )

            safe_name = os.path.basename(filename)
            pdf_path = os.path.join(temp_dir, safe_name)

            with open(pdf_path, "wb") as output:
                shutil.copyfileobj(upload.file, output)

            result = process_pdf(pdf_path, safe_name)
            parsed_documents.append(result)

        # STEP 2: Extract cross-document semantic concepts using Gemini
        contexts_by_document = {
            result["filename"]: result["contexts"]
            for result in parsed_documents
        }

        keywords = semantic_selector.extract_keywords(contexts_by_document)
        print(f"[Demo] Gemini comparison keywords: {', '.join(keywords)}")

        # STEP 3: Route targeted contexts and perform grounded extraction
        all_results: List[Dict[str, Any]] = []
        all_facts: List[ExtractedFact] = []

        for result in parsed_documents:
            filename = result["filename"]
            contexts = result["contexts"]

            selected_contexts = _select_demo_contexts(contexts, keywords, limit=1)
            facts: List[ExtractedFact] = []

            if selected_contexts:
                for context_number, selected_context in enumerate(
                    selected_contexts,
                    start=1,
                ):
                    print(
                        f"[Demo] {filename}: "
                        f"context={selected_context.context_id} "
                        f"({context_number}/{len(selected_contexts)}) "
                        f"tokens={selected_context.estimated_tokens} "
                        f"evidence={len(selected_context.evidence)}"
                    )

                    extracted = extractor.extract(selected_context)
                    facts.extend(canonicalize_facts(extracted))

                selected_context_ids = [c.context_id for c in selected_contexts]
                processed_contexts = len(selected_contexts)
            else:
                selected_context_ids = []
                processed_contexts = 0

            all_results.append(
                {
                    "filename": filename,
                    "raw_elements": result["raw_elements"],
                    "clean_elements": result["clean_elements"],
                    "filtered_elements": result["filtered_elements"],
                    "context_count": len(contexts),
                    "processed_contexts": processed_contexts,
                    "selected_contexts": selected_context_ids,
                    "mistral_extraction_calls": processed_contexts,
                    "fact_count": len(facts),
                    "valid_fact_count": sum(
                        1
                        for fact in facts
                        if getattr(fact, "status", "VALID") == "VALID"
                    ),
                    "uncertain_fact_count": sum(
                        1
                        for fact in facts
                        if getattr(fact, "status", "VALID") == "UNCERTAIN"
                    ),
                    "grounding_failure_count": sum(
                        1
                        for fact in facts
                        if getattr(fact, "status", "VALID") == "GROUNDING_FAILURE"
                    ),
                }
            )

            all_facts.extend(facts)

    # STEP 4: Classify relationships across grounded facts
    valid_facts = [
        fact
        for fact in all_facts
        if getattr(fact, "status", "VALID") == "VALID"
    ]

    relationships = matcher.find_and_classify(valid_facts)
    relationship_rows = [
        relationship_to_dict(pair, classification)
        for pair, classification in relationships
    ]

    stats = {
        "documents": len(all_results),
        "facts": len(valid_facts),
        "candidate_relationships": len(relationship_rows),
        "corroborations": sum(
            1
            for row in relationship_rows
            if row["status"] == "CORROBORATION"
        ),
        "contradictions": sum(
            1
            for row in relationship_rows
            if row["status"] == "CONTRADICTION"
        ),
        "contextual_resolutions": sum(
            1
            for row in relationship_rows
            if row["status"] == "CONTEXTUAL_RESOLUTION"
        ),
        "uncertain": sum(
            1
            for row in relationship_rows
            if row["status"] == "UNCERTAIN"
        ),
        "uncertain_extractions": sum(
            doc["uncertain_fact_count"] for doc in all_results
        ),
        "grounding_failures": sum(
            doc["grounding_failure_count"] for doc in all_results
        ),
    }

    analysis: Dict[str, Any] = {
        "analysis_id": analysis_id,
        "demo_mode": True,
        "contexts_per_pdf": 2,
        "selection_strategy": "Gemini-derived semantic keywords; top 2 contexts per PDF",
        "selection_keywords": keywords,
        "documents": all_results,
        "facts": [fact_to_dict(fact) for fact in valid_facts],
        "relationships": relationship_rows,
        "stats": stats,
    }

    ANALYSES[analysis_id] = analysis
    return analysis


@app.get("/api/analysis/{analysis_id}")
def get_analysis(analysis_id: str) -> Dict[str, Any]:
    analysis = ANALYSES.get(analysis_id)
    if analysis is None:
        raise HTTPException(
            status_code=404,
            detail="Analysis not found.",
        )
    return analysis


@app.post("/api/ask")
def ask(request: AskRequest) -> Dict[str, Any]:
    analysis = ANALYSES.get(request.analysis_id)
    if analysis is None:
        raise HTTPException(
            status_code=404,
            detail="Analysis not found.",
        )

    question = request.question.strip().lower()
    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    words = {word for word in question.split() if len(word) > 2}
    scored = []

    for fact in analysis["facts"]:
        text = " ".join(
            [
                fact.get("entity") or "",
                fact.get("attribute") or "",
                fact.get("raw_value") or "",
                fact.get("time") or "",
                fact.get("scope") or "",
            ]
        ).lower()

        score = sum(1 for word in words if word in text)
        if score:
            scored.append((score, fact))

    scored.sort(key=lambda item: item[0], reverse=True)
    top_facts = [fact for _, fact in scored[:8]]

    return {
        "question": request.question,
        "answer": (
            f"I found {len(top_facts)} relevant fact(s) in the analyzed documents."
            if top_facts
            else "I couldn't find a directly matching fact."
        ),
        "facts": top_facts,
    }


@app.get("/")
def index():
    static_file = "src/api/static/index.html"
    if not os.path.exists(static_file):
        return {"message": "Fact Knowledge Layer API is running."}
    return FileResponse(static_file)