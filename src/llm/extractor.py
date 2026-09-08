from typing import Dict, List, Optional, Tuple

from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI

from config import settings
from src.llm.prompts import SYSTEM_PROMPT, USER_PROMPT
from src.schemas.context import CompactContext, ContextEvidence
from src.schemas.extraction import ExtractedFact, FactExtractionBatch


class FactExtractor:
    def __init__(self) -> None:
        if not settings.MISTRAL_API_KEY:
            raise ValueError(
                "MISTRAL_API_KEY is not configured in .env"
            )

        print("[Extractor] Initializing Mistral API...")

        self.model_name = settings.LLM_MODEL

        self.llm = ChatMistralAI(
            model=settings.LLM_MODEL,
            api_key=settings.MISTRAL_API_KEY,
            temperature=0,
        ).with_structured_output(FactExtractionBatch)

        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                ("human", USER_PROMPT),
            ]
        )

        self.chain = self.prompt | self.llm

    @staticmethod
    def _serialize_context(context: CompactContext) -> str:
        """
        Convert structured context into the text format expected by
        the extraction prompt.
        """
        parts: List[str] = []

        for evidence in context.evidence:
            header = (
                f"EVIDENCE_ID: {evidence.evidence_id}\n"
                f"SOURCE_FILE: {evidence.source_file}\n"
                f"PAGE: {evidence.page_number}\n"
            )

            if evidence.element_type:
                header += f"ELEMENT_TYPE: {evidence.element_type}\n"

            if evidence.table_ref:
                header += f"TABLE_REF: {evidence.table_ref}\n"

            if evidence.table_headers:
                header += (
                    "TABLE_HEADERS:\n"
                    + "\n".join(
                        f"  HEADER {i}: {header_value}"
                        for i, header_value in enumerate(evidence.table_headers)
                    )
                    + "\n"
                )

            if evidence.table_rows:
                header += "TABLE_ROWS:\n"
                for row_index, row in enumerate(evidence.table_rows):
                    header += f"  ROW {row_index}:\n"
                    for column_index, cell in enumerate(row):
                        header += f"    CELL {column_index}: {cell}\n"

            header += f"TEXT:\n{evidence.text}\n"
            parts.append(header)

        return "\n\n".join(parts)

    @staticmethod
    def _build_evidence_lookup(
        context: CompactContext,
    ) -> Dict[str, ContextEvidence]:
        return {
            evidence.evidence_id: evidence
            for evidence in context.evidence
        }

    @staticmethod
    def _build_table_lookup(
        context: CompactContext,
    ) -> Dict[str, ContextEvidence]:
        table_lookup: Dict[str, ContextEvidence] = {}
        for evidence in context.evidence:
            if evidence.table_ref:
                table_lookup[evidence.table_ref] = evidence
        return table_lookup

    @staticmethod
    def _ground_table_fact(
        fact: ExtractedFact,
        table_lookup: Dict[str, ContextEvidence],
    ) -> bool:
        """
        Validate table coordinates against the supplied table evidence
        and ensure deterministic value attribution.
        """
        if not fact.table_ref:
            return False

        evidence = table_lookup.get(fact.table_ref)
        if evidence is None or not evidence.table_rows:
            return False

        if fact.row_index is None or fact.column_index is None:
            return False

        if not (0 <= fact.row_index < len(evidence.table_rows)):
            return False

        row = evidence.table_rows[fact.row_index]

        if not (0 <= fact.column_index < len(row)):
            return False

        cell_value = row[fact.column_index].strip()
        if not cell_value:
            return False

        # Enforce application-owned actual value and column name
        fact.raw_value = cell_value

        if evidence.table_headers and fact.column_index < len(evidence.table_headers):
            fact.column_name = evidence.table_headers[fact.column_index]

        if (
            evidence.table_evidence_ids
            and fact.row_index < len(evidence.table_evidence_ids)
        ):
            fact.evidence_id = evidence.table_evidence_ids[fact.row_index]
        else:
            fact.evidence_id = f"{fact.table_ref}:row:{fact.row_index}"

        fact.exact_quote = " | ".join(row)

        return True

    @staticmethod
    def _ground_prose_fact(
        fact: ExtractedFact,
        evidence_lookup: Dict[str, ContextEvidence],
    ) -> bool:
        """
        Validate that a prose fact points to an existing evidence unit
        and that the quoted text is grounded in that evidence.
        """
        if not fact.evidence_id:
            return False

        evidence = evidence_lookup.get(fact.evidence_id)
        if evidence is None:
            return False

        if not fact.exact_quote:
            return False

        quote = fact.exact_quote.strip()
        source_text = (evidence.text or "").strip()

        if not quote or not source_text:
            return False

        if quote in source_text:
            return True

        normalized_quote = " ".join(quote.split()).lower()
        normalized_source = " ".join(source_text.split()).lower()

        return normalized_quote in normalized_source

    def _ground(
        self,
        facts: List[ExtractedFact],
        context: CompactContext,
    ) -> List[ExtractedFact]:
        """
        Validate every extracted fact against the original context.

        Invalid or incomplete facts are retained with an explicit status
        so the caller can inspect extraction failures without allowing
        them into downstream matching.
        """
        evidence_lookup = self._build_evidence_lookup(context)
        table_lookup = self._build_table_lookup(context)

        grounded: List[ExtractedFact] = []

        for fact in facts:
            # Drop untrusted LLM-generated normalized values
            fact.normalized_value = None

            # Basic required-field validation
            if not fact.entity or not fact.attribute or not fact.raw_value:
                fact.status = "UNCERTAIN"
                grounded.append(fact)
                continue

            # Table fact grounding
            if fact.table_ref:
                if self._ground_table_fact(fact, table_lookup):
                    fact.status = "VALID"
                else:
                    fact.status = "GROUNDING_FAILURE"
                    print(
                        f"[Grounding] Table fact rejected: "
                        f"table_ref={fact.table_ref}, "
                        f"row={fact.row_index}, "
                        f"column={fact.column_index}"
                    )
                grounded.append(fact)
                continue

            # Prose fact grounding
            if fact.evidence_id:
                if self._ground_prose_fact(fact, evidence_lookup):
                    fact.status = "VALID"
                else:
                    fact.status = "GROUNDING_FAILURE"
                    print(
                        f"[Grounding] Prose fact rejected: "
                        f"evidence_id={fact.evidence_id}"
                    )
                grounded.append(fact)
                continue

            # Missing all provenance
            fact.status = "GROUNDING_FAILURE"
            print(
                "[Grounding] Fact rejected: missing evidence_id/table_ref"
            )
            grounded.append(fact)

        return grounded

    def extract(
        self,
        context: CompactContext,
    ) -> List[ExtractedFact]:
        """
        Run one structured Mistral extraction call and validate
        provenance deterministically.

        Malformed structured output is treated as an extraction
        failure for this context rather than crashing the API.
        """
        evidence_text = self._serialize_context(context)

        try:
            response: FactExtractionBatch = self.chain.invoke(
                {
                    "evidence": evidence_text,
                }
            )
        except Exception as exc:
            print(
                "[Extractor] Structured output failed: "
                f"context={context.context_id} "
                f"error={exc}"
            )
            return []

        if not response or not response.facts:
            print(
                f"[Extractor] No facts returned: context={context.context_id}"
            )
            return []

        return self._ground(
            response.facts,
            context,
        )