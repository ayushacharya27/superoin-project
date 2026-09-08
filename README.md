# Fact Knowledge Layer

A document intelligence system that extracts **grounded facts** from PDFs and identifies cross-document relationships such as **corroboration, contradiction, contextual resolution, and uncertainty**. fileciteturn18file0L3-L8

The system is designed to generalize to new PDFs without hardcoded facts, filenames, or document-specific schemas/rules. fileciteturn18file0L10-L10

## Demo

**[Watch the 3-minute demo](YOUR_VIDEO_LINK)**

Run locally:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn src.api.app:app --reload
```

Open: `http://127.0.0.1:8000`

API docs: `http://127.0.0.1:8000/docs`

## Architecture

```text
PDF
  ↓
Structure-aware parsing
  ↓
PROSE / HEADING / TABLE
  ↓
Cleanup + filtering
  ↓
Bounded structured context
  ↓
Mistral + LangChain extraction
  ↓
Evidence grounding
  ↓
Fact canonicalization
  ↓
Embeddings
  ↓
Entity / attribute candidate blocking
  ↓
Deterministic relationship reasoning
  ↓
FastAPI
  ↓
ChatGPT-style UI
```

For the current demo, **one relationship-rich bounded context and one Mistral extraction call are used per PDF**.

## Key Design

**LLM:** extracts structured facts from supplied evidence.

**Grounding:** verifies source quotes, table coordinates, and provenance.

**Canonicalization:** normalizes metric names, time, and common value representations without inventing facts.

**Embeddings:** generate semantic candidates only; they are not the final source of truth.

**Relationship engine:** classifies compatible facts as `CORROBORATION`, `CONTRADICTION`, `CONTEXTUAL_RESOLUTION`, or `UNCERTAIN`.

Every fact retains source information such as file, page, evidence ID, and quote. Unsupported or malformed extractions are rejected or marked uncertain instead of being silently accepted. fileciteturn18file0L22-L29

## Project Structure

```text
fact-knowledge-layer/
├── data/
│   └── starter_pdfs/
├── src/
│   ├── api/
│   │   ├── app.py
│   │   └── static/index.html
│   ├── llm/
│   │   ├── extractor.py
│   │   └── prompts.py
│   ├── matcher/
│   │   ├── embeddings.py
│   │   └── resolve.py
│   ├── pipeline/
│   │   ├── document_structure.py
│   │   ├── document_cleanup.py
│   │   ├── llm_filter.py
│   │   ├── document_chunking.py
│   │   └── fact_canonicalizer.py
│   └── schemas/
├── config.py
├── test_run.py
├── test_matcher.py
├── requirements.txt
└── README.md
```

## Limitations / Next Steps

The demo intentionally processes one selected context per PDF. A production version should add:

- multi-context processing for large PDFs
- cached and incremental ingestion
- persistent fact/evidence storage
- deeper reasoning for ambiguous relationships
- richer source/document navigation

## Design Principle

> **LLMs extract; evidence validates; embeddings retrieve candidates; deterministic logic establishes relationships.**
