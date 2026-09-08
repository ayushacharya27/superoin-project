import logging
from typing import Any, Dict, List

from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI

from config import settings
from src.llm.prompts import SYSTEM_PROMPT, USER_PROMPT
from src.schemas.context import CompactContext, ContextEvidence
from src.schemas.extraction import ExtractedFact, FactExtractionBatch

logger = logging.getLogger(__name__)


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
    def _serialize_context(
        context: CompactContext,
    ) -> str:
        """
        Serialize structured document evidence into the exact
        text representation expected by the extraction prompt.
        """
        parts: List[str] = []

        for evidence in context.evidence:
            header = (
                f"EVIDENCE_ID: {evidence.evidence_id}\n"
                f"SOURCE_FILE: {evidence.source_file}\n"
                f"PAGE: {evidence.page_number}\n"
            )

            if getattr(evidence, "element_type", None):
                header += f"ELEMENT_TYPE: {evidence.element_type}\n"

            if getattr(evidence, "table_ref", None):
                header += f"TABLE_REF: {evidence.table_ref}\n"

            if getattr(evidence, "table_headers", None):
                header += (
                    "TABLE_HEADERS:\n"
                    + "\n".join(
                        f"  HEADER {i}: {header_value}"
                        for i, header_value in enumerate(evidence.table_headers)
                    )
                    + "\n"
                )

            if getattr(evidence, "table_rows", None):
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
            if getattr(evidence, "table_ref", None):
                table_lookup[evidence.table_ref] = evidence
        return table_lookup

    @staticmethod
    def _ground_table_fact(
        fact: ExtractedFact,
        table_lookup: Dict[str, ContextEvidence],
    ) -> bool:
        """
        Validate coordinates against structured table evidence, then bind
        the ground-truth cell value and provenance deterministically.
        """
        if not fact.table_ref:
            return False

        evidence = table_lookup.get(fact.table_ref)
        if evidence is None or not getattr(evidence, "table_rows", None):
            return False

        if fact.row_index is None or fact.column_index is None:
            return False

        if not (0 <= fact.row_index < len(evidence.table_rows)):
            return False

        row = evidence.table_rows[fact.row_index]

        if not (0 <= fact.column_index < len(row)):
            return False

        cell_value = str(row[fact.column_index]).strip()
        if not cell_value:
            return False

        # Bind application-controlled structural values
        fact.raw_value = cell_value

        if (
            getattr(evidence, "table_headers", None)
            and fact.column_index < len(evidence.table_headers)
        ):
            fact.column_name = evidence.table_headers[fact.column_index]

        if (
            getattr(evidence, "table_evidence_ids", None)
            and fact.row_index < len(evidence.table_evidence_ids)
        ):
            fact.evidence_id = evidence.table_evidence_ids[fact.row_index]
        else:
            fact.evidence_id = f"{fact.table_ref}:row:{fact.row_index}"

        fact.exact_quote = " | ".join(str(cell) for cell in row)
        return True

    @staticmethod
    def _ground_prose_fact(
        fact: ExtractedFact,
        evidence_lookup: Dict[str, ContextEvidence],
    ) -> bool:
        """
        Validate prose provenance and require the quote to be an
        exact substring of the referenced evidence.

        Also require raw_value to occur in the exact quote whenever
        a raw_value exists to eliminate ungrounded textual paraphrases.
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

        # Support exact match first, then whitespace-normalized match
        if quote not in source_text:
            norm_quote = " ".join(quote.split()).lower()
            norm_source = " ".join(source_text.split()).lower()
            if norm_quote not in norm_source:
                return False

        if (
            fact.raw_value
            and fact.raw_value.strip()
            and fact.raw_value.strip() not in quote
        ):
            norm_val = " ".join(fact.raw_value.split()).lower()
            norm_quote = " ".join(quote.split()).lower()
            if norm_val not in norm_quote:
                return False

        return True

    def _ground(
        self,
        facts: List[ExtractedFact],
        context: CompactContext,
    ) -> List[ExtractedFact]:
        """
        Validate extracted facts against the original context.

        Facts are retained for inspection even when invalid, but only
        VALID facts are eligible for downstream matching.
        """
        evidence_lookup = self._build_evidence_lookup(context)
        table_lookup = self._build_table_lookup(context)

        grounded: List[ExtractedFact] = []

        for fact in facts:
            # Model-provided normalized values are not authoritative
            fact.normalized_value = None

            # Basic field presence validation
            if (
                not fact.entity
                or not fact.attribute
                or not fact.raw_value
            ):
                fact.status = "UNCERTAIN"
                grounded.append(fact)
                continue

            # Table fact verification
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

            # Prose fact verification
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
        Execute exactly one structured Mistral extraction call
        for the supplied context.

        Malformed structured output is handled as an extraction
        failure instead of crashing the API.
        """
        evidence_text = self._serialize_context(context)

        try:
            response: FactExtractionBatch = self.chain.invoke(
                {"evidence": evidence_text}
            )
        except Exception as exc:
            print(
                f"[Extractor] Structured output failed: "
                f"context={context.context_id} error={exc}"
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