# Fact Knowledge Layer

A document intelligence prototype that turns PDF content into grounded, comparable facts and identifies relationships across documents.

The system extracts meaningful numerical and semantic facts, links them to source evidence, and classifies cross-document relationships as corroboration, contradiction, or contextual resolution.

## Setup and Run Instructions

### 1. Clone and enter the repository

```bash
git clone <your-github-repo-url>
cd fact-knowledge-layer
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate
```

On Windows:

```powershell
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```env
LLM_PROVIDER=mistral
LLM_MODEL=ministral-3b-latest
MISTRAL_API_KEY=your_mistral_key

GEMINI_API_KEY=your_gemini_key
GEMINI_MODEL=gemini-2.5-flash

EMBEDDING_MODEL=all-MiniLM-L6-v2
EMBEDDING_BATCH_SIZE=64

MAX_CONTEXT_TOKENS=2500
CHARS_PER_TOKEN=4
MAX_EVIDENCE_PER_GROUP=30

SQLITE_DB_PATH=data/facts.sqlite
```

Never commit `.env` or API keys.

### 5. Start the application

```bash
uvicorn src.api.app:app --reload
```

Open `http://127.0.0.1:8000` in a browser.

The application accepts multiple PDFs and processes them into grounded facts and cross-document relationships.

## Video Demo

**Demo video (≤3 minutes):** `<add-demo-video-link>`

The demo shows:

1. Uploading and processing the PDF documents.
2. Extracted facts with source evidence.
3. A corroborated fact across documents.
4. A genuine or likely contradiction.
5. An apparent contradiction resolved using context such as time, scope, or units.
6. An extraction/grounding failure and how the system handles it.

## Approach

### Pipeline

```text
PDF upload
    ↓
Structure-aware PDF parsing
    ↓
PROSE / HEADING / TABLE elements
    ↓
Deterministic cleanup
    ↓
Bounded document contexts
    ↓
Semantic context selection
    ↓
LLM fact extraction
    ↓
Evidence grounding / validation
    ↓
Canonical fact normalization
    ↓
Fact embeddings
    ↓
Cross-document candidate matching
    ↓
Relationship classification
    ↓
SQLite + FastAPI + UI
```

### 1. Structure-aware parsing

PDFs are parsed with layout information rather than treating the document as one large text blob.

The parser preserves headings, prose, tables, table headers and rows, page numbers, and source/evidence identifiers. This allows facts extracted from tables to retain row/column context.

### 2. Context construction

Documents are cleaned and divided into bounded, ordered contexts. Contexts do not mix content from different PDFs. Tables are kept intact where possible so numerical facts retain their surrounding structure.

### 3. Semantic context selection

A lightweight Gemini step identifies semantic concepts that are useful for finding comparable information across the uploaded documents.

The selector examines partial document evidence and produces comparison-oriented keywords. These keywords are then used to select the most relevant full context from each document.

This reduces the amount of document content sent to the extraction model while keeping the selection generic rather than hard-coding facts or document names.

### 4. Fact extraction

Mistral is used through LangChain to extract structured facts.

A fact can contain entity, attribute, raw value, unit, normalized value, time, scope, qualifier, exact source quote, evidence ID, and extraction confidence.

The extractor is instructed to use only supplied evidence and not invent facts.

### 5. Grounding

Every extracted fact is checked against its referenced evidence.

For prose, the quoted evidence must be supported by the corresponding source text. For tables, the table reference and row/column location are validated.

Unsupported facts are rejected instead of being exposed as trusted knowledge.

This also provides an explicit way to handle LLM extraction failures.

### 6. Canonicalization

Extracted facts are converted into a common representation so differently worded statements can be compared.

Numerical normalization is deterministic where possible rather than relying on the LLM to perform arithmetic or unit conversion.

Time, scope, and qualifiers are retained because two values that look different may describe different contexts.

### 7. Relationship matching

Sentence-transformer embeddings are generated **after** fact extraction.

Embeddings are used to find semantically similar facts from different documents and generate candidate pairs. Numerical values are not included in the identity embedding, because two different values for the same metric are precisely what may indicate a contradiction.

Candidate pairs are then evaluated using the canonical fact fields and contextual information.

Relationships include:

- **CORROBORATION** — the facts describe the same underlying claim/value.
- **CONTRADICTION** — the facts describe the same context but contain materially different values.
- **CONTEXTUAL_RESOLUTION** — the values differ, but the difference can be explained by context such as time or scope.
- **UNCERTAIN** — the system cannot confidently determine the relationship.

### 8. Storage and API/UI

SQLite stores the prototype's fact/relationship data.

FastAPI exposes the processing and inspection interface, while the web UI provides a simple way to upload PDFs and inspect extracted results.

The design intentionally avoids requiring a graph database or vector database for the starter dataset.

## Key Engineering Decisions

### Ground evidence before trusting a fact

An LLM can produce a plausible-looking statement that is not actually present in the source. The grounding layer therefore acts as a trust boundary between extraction and the knowledge layer.

### Separate semantic similarity from numerical comparison

Embeddings answer: "Are these facts about the same kind of thing?" They do not decide whether the numerical values agree.

The relationship layer compares values and context separately.

### Preserve context

Time, scope, units, qualifiers, and table structure are retained because apparent contradictions are often caused by different reporting periods, populations, or measurement definitions.

### Keep extraction and relationship reasoning separate

The first LLM extracts facts from evidence. Cross-document reasoning happens afterward on canonical facts rather than repeatedly sending raw PDFs to an LLM.

This keeps the architecture understandable and limits unnecessary LLM calls.

## Project Structure

```text
fact-knowledge-layer/
├── data/
│   └── starter_pdfs/
├── src/
│   ├── api/
│   │   └── app.py
│   ├── db/
│   ├── llm/
│   │   ├── extractor.py
│   │   └── semantic_selector.py
│   ├── matcher/
│   ├── pipeline/
│   │   ├── document_structure.py
│   │   ├── document_cleanup.py
│   │   ├── document_chunking.py
│   │   └── llm_filter.py
│   └── schemas/
├── config.py
├── requirements.txt
├── README.md
└── test_run.py
```

## Limitations and Next Steps

The current system is a prototype and is intentionally optimized for clarity and demonstration rather than production scale.

Known limitations:

- PDF extraction quality depends on the source document's layout.
- Some LLM outputs fail evidence validation and are rejected.
- Semantic candidate matching can still produce ambiguous pairs.
- Relationship classification is currently conservative and deterministic.
- The prototype uses bounded contexts and therefore does not yet provide a full large-document indexing strategy.
- SQLite is sufficient for the starter dataset but would not be the ideal storage layer at large scale.
- Incremental document ingestion and persistent caching are not yet fully implemented.

Potential next steps:

1. Cache parsed contexts and LLM results using content hashes.
2. Add stronger table reconstruction for complex PDFs.
3. Introduce a second reasoning pass only for ambiguous relationship candidates.
4. Improve confidence calibration by separating extraction confidence from relationship confidence.
5. Add incremental ingestion so new documents can be added without rebuilding existing knowledge.
6. Scale storage and retrieval for substantially larger document collections.

## Additional Notes

The system deliberately does not hard-code the starter PDFs, their filenames, individual facts, or document-specific extraction rules.

The fact representation is generic enough to accommodate new entities, attributes, metrics, periods, scopes, and qualifiers.

The intended workflow is:

```text
new PDF → parse → extract → ground → canonicalize → compare with existing facts
```

A rejected fact is treated as a useful failure signal rather than silently accepted. This makes extraction errors visible and provides a clear path for improving the system.

The project uses LangChain for LLM integration, Gemini for semantic context selection, Mistral for structured fact extraction, sentence-transformer embeddings for candidate generation, and FastAPI/SQLite for the application layer.
