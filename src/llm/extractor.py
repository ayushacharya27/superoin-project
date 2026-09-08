import json
import logging
from typing import Dict, List, Optional, Tuple

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
            raise ValueError("MISTRAL_API_KEY is not configured.")

        logger.info("[Extractor] Initializing Mistral API...")

        self.model_name = settings.LLM_MODEL

        self.llm = ChatMistralAI(
            model=self.model_name,
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
        parts: List[str] = []

        for item in context.evidence:
            parts.append(
                "\n".join(
                    [
                        f"EVIDENCE_ID: {item.evidence_id}",
                        f"SOURCE_FILE: {item.source_file}",
                        f"PAGE: {item.page_number}",
                        "TEXT:",
                        item.text,
                        "END_EVIDENCE",
                    ]
                )
            )

        return "\n\n".join(parts)

    @staticmethod
    def _build_evidence_lookup(
        context: CompactContext,
    ) -> Dict[str, ContextEvidence]:
        return {
            item.evidence_id: item
            for item in context.evidence
        }

    @staticmethod
    def _build_table_lookup(
        context: CompactContext,
    ) -> Dict[str, ContextEvidence]:
        return {
            item.table_ref: item
            for item in context.evidence
            if (
                item.element_type == "TABLE"
                and item.table_ref
            )
        }

    @staticmethod
    def _ground_table_fact(
        fact: ExtractedFact,
        table_lookup: Dict[str, ContextEvidence],
    ) -> Tuple[bool, ExtractedFact]:
        """
        Resolve the LLM's table coordinates against the actual
        structured table carried by ContextEvidence.

        The LLM proposes coordinates.
        The application owns the actual value and provenance.
        """
        if not fact.table_ref:
            return False, fact

        table_evidence = table_lookup.get(fact.table_ref)

        if table_evidence is None:
            return False, fact

        if fact.row_index is None:
            return False, fact

        if fact.column_index is None:
            return False, fact

        if not (
            0 <= fact.row_index < len(table_evidence.table_rows)
        ):
            return False, fact

        row = table_evidence.table_rows[fact.row_index]

        if not (
            0 <= fact.column_index < len(row)
        ):
            return False, fact

        actual_value = row[fact.column_index].strip()

        if not actual_value:
            return False, fact

        # Application-owned value.
        fact.raw_value = actual_value

        # Application-owned column name.
        if fact.column_index < len(table_evidence.table_headers):
            fact.column_name = table_evidence.table_headers[
                fact.column_index
            ]

        # Application-owned provenance.
        if fact.row_index < len(table_evidence.table_evidence_ids):
            fact.evidence_id = table_evidence.table_evidence_ids[
                fact.row_index
            ]
        else:
            fact.evidence_id = f"{fact.table_ref}:row:{fact.row_index}"

        # Application-generated human-readable quote.
        fact.exact_quote = " | ".join(row)

        return True, fact

    @staticmethod
    def _ground_prose_fact(
        fact: ExtractedFact,
        evidence_lookup: Dict[str, ContextEvidence],
    ) -> Tuple[bool, ExtractedFact]:
        """
        Deterministically validate prose facts using their
        evidence_id and exact_quote.
        """
        if not fact.evidence_id:
            return False, fact

        evidence = evidence_lookup.get(fact.evidence_id)

        if evidence is None:
            return False, fact

        quote = (
            fact.exact_quote.strip()
            if fact.exact_quote
            else ""
        )

        if not quote:
            return False, fact

        source_text = evidence.text

        # First try an exact match.
        if quote in source_text:
            return True, fact

        # Then tolerate harmless whitespace differences.
        normalized_quote = " ".join(quote.split()).lower()
        normalized_source = " ".join(source_text.split()).lower()

        if normalized_quote in normalized_source:
            return True, fact

        return False, fact

    def _ground(
        self,
        facts: List[ExtractedFact],
        context: CompactContext,
    ) -> List[ExtractedFact]:
        evidence_lookup = self._build_evidence_lookup(context)
        table_lookup = self._build_table_lookup(context)

        validated_facts: List[ExtractedFact] = []

        for fact in facts:
            # Never trust model-generated normalization.
            fact.normalized_value = None

            if fact.table_ref:
                valid, fact = self._ground_table_fact(
                    fact,
                    table_lookup,
                )
            else:
                valid, fact = self._ground_prose_fact(
                    fact,
                    evidence_lookup,
                )

            fact.status = (
                "VALID"
                if valid
                else "GROUNDING_FAILURE"
            )

            if not valid:
                if fact.table_ref:
                    logger.warning(
                        "[Grounding] Table fact rejected: "
                        "table_ref=%s, row=%s, column=%s",
                        fact.table_ref,
                        fact.row_index,
                        fact.column_index,
                    )
                else:
                    logger.warning(
                        "[Grounding] Prose fact rejected: evidence_id=%s",
                        fact.evidence_id,
                    )

            validated_facts.append(fact)

        return validated_facts

    def extract(
        self,
        context: CompactContext,
    ) -> List[ExtractedFact]:
        """
        Run Mistral extraction followed by deterministic grounding.
        """
        evidence_text = self._serialize_context(context)

        response: FactExtractionBatch = self.chain.invoke(
            {"evidence": evidence_text}
        )

        if not isinstance(response, FactExtractionBatch):
            raise TypeError("Unexpected structured output from Mistral.")

        return self._ground(
            response.facts,
            context,
        )