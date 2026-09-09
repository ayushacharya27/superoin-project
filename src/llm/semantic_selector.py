import logging
from typing import Any, Dict, List

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from config import settings
from src.schemas.context import CompactContext

logger = logging.getLogger(__name__)


class KeywordResult(BaseModel):
    keywords: List[str] = Field(
        default_factory=list,
        description=(
            "Concise semantic metric/concept keywords that can help identify "
            "facts that are comparable across the supplied documents."
        ),
    )


class SemanticSelector:
    def __init__(self) -> None:
        if not settings.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is not configured in .env")

        logger.info("[SemanticSelector] Initializing Gemini...")

        self.llm = ChatGoogleGenerativeAI(
            model=settings.GEMINI_MODEL,
            google_api_key=settings.GEMINI_API_KEY,
            temperature=0,
        ).with_structured_output(
            KeywordResult,
            method="json_schema",
        )

        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "You are a semantic topic selector for a document fact "
                        "knowledge system.\n\n"
                        "You will receive short partial excerpts from multiple documents.\n\n"
                        "Identify concise semantic concepts that are likely to help find facts "
                        "that can be compared across those documents.\n\n"
                        "Return ONLY useful comparison keywords.\n\n"
                        "Rules:\n"
                        "- Use only concepts visible in the supplied excerpts.\n"
                        "- Prefer metrics, quantities, business concepts, operational concepts, "
                        "events, and dimensions.\n"
                        "- Do not extract final facts.\n"
                        "- Do not invent terminology.\n"
                        "- Avoid generic words such as company, document, information, value, "
                        "amount, data.\n"
                        "- Keep keywords concise.\n"
                        "- Return at most 12 keywords."
                    ),
                ),
                (
                    "human",
                    (
                        "Partial document excerpts:\n\n"
                        "{evidence}\n\n"
                        "Return the comparison keywords."
                    ),
                ),
            ]
        )

        self.chain = self.prompt | self.llm

    @staticmethod
    def _half_context_text(context: CompactContext) -> str:
        """
        Return approximately half of one bounded context to provide
        broad multi-document coverage within tight token budgets.
        """
        parts: List[str] = []

        for evidence in context.evidence:
            text = evidence.text or ""

            if getattr(evidence, "table_headers", None):
                text += " " + " ".join(evidence.table_headers)

            if getattr(evidence, "table_rows", None):
                for row in evidence.table_rows:
                    text += " " + " ".join(row)

            parts.append(text)

        full_text = "\n".join(parts).strip()
        if not full_text:
            return ""

        midpoint = max(len(full_text) // 2, 1)
        return full_text[:midpoint]

    def extract_keywords(
        self,
        contexts_by_document: Dict[str, List[CompactContext]],
    ) -> List[str]:
        """
        Sample partial context from each document and prompt Gemini
        to extract shared comparison concepts across documents.
        """
        sections: List[str] = []

        for filename, contexts in contexts_by_document.items():
            if not contexts:
                continue

            # Sample the initial bounded context from each document
            context = contexts[0]
            partial = self._half_context_text(context)

            if not partial:
                continue

            sections.append(
                f"DOCUMENT: {filename}\n"
                f"CONTEXT: {context.context_id}\n\n"
                f"PARTIAL EVIDENCE:\n"
                f"{partial}\n"
            )

        if not sections:
            return []

        evidence = "\n\n".join(sections)
        logger.info(
            "[SemanticSelector] Querying Gemini for comparison keywords across %d document(s)...",
            len(sections),
        )

        try:
            result: KeywordResult = self.chain.invoke({"evidence": evidence})
        except Exception as exc:
            logger.error(
                "[SemanticSelector] Gemini keyword extraction call failed: %s",
                exc,
            )
            return []

        keywords: List[str] = []
        for keyword in result.keywords:
            cleaned = keyword.strip().lower()
            if cleaned and cleaned not in keywords:
                keywords.append(cleaned)

        logger.info(
            "[SemanticSelector] Comparison keywords: %s",
            ", ".join(keywords),
        )
        return keywords