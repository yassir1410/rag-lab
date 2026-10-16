import time
import uuid
import re
from typing import List, Type
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.rag.agents.tools.base import BaseAgentTool
from app.rag.agents.schemas import ToolResult


class InspectEvidenceArgs(BaseModel):
    source_ids: List[str] = Field(..., min_length=1, description="List of source IDs to inspect (e.g. ['[S1]', '[S2]'])")
    question: str = Field(..., min_length=2, description="Target question or claim to verify against the evidence")


class InspectEvidenceTool(BaseAgentTool):
    """
    Approved Tool: Evidence Inspector & Sufficiency Checker.
    Analyzes whether currently collected evidence snippets contain sufficient information
    to ground an answer to a given question or sub-question without hallucination.
    """

    def __init__(self, evidence_collector=None):
        self._collector = evidence_collector

    def set_collector(self, collector):
        self._collector = collector

    @property
    def name(self) -> str:
        return "inspect_evidence"

    @property
    def description(self) -> str:
        return (
            "Checks whether specified evidence items (e.g. ['[S1]', '[S2]']) adequately address a given question. "
            "Evaluates term coverage, factual cues, and indicates whether further searches are needed."
        )

    @property
    def args_schema(self) -> Type[BaseModel]:
        return InspectEvidenceArgs

    def execute(self, session: Session, **kwargs) -> ToolResult:
        t0 = time.perf_counter()
        call_id = kwargs.pop("call_id", str(uuid.uuid4())[:8])

        try:
            validated = self.args_schema(**kwargs)
            question = validated.question
            source_ids = validated.source_ids

            # Extract terms from question
            q_terms = set(re.findall(r"\b[a-zA-Z0-9_\u00C0-\u017F]{3,}\b", question.lower()))
            # Remove common stopwords
            stopwords = {"what", "when", "where", "which", "who", "whom", "whose", "why", "how",
                         "does", "were", "been", "have", "with", "from", "that", "this", "they",
                         "about", "their", "will", "would", "could", "should"}
            key_terms = q_terms - stopwords

            # Retrieve text for source_ids from collector if available
            combined_text = ""
            matched_sources = []
            if self._collector:
                for sid in source_ids:
                    item = self._collector.get_by_source_id(sid)
                    if item:
                        combined_text += " " + item.content
                        matched_sources.append(sid)

            # In case collector was not injected or source_ids were passed without prefix
            text_lower = combined_text.lower()
            found_terms = [t for t in key_terms if t in text_lower]
            coverage_ratio = len(found_terms) / max(len(key_terms), 1)

            is_sufficient = coverage_ratio >= 0.5 and len(combined_text) > 50

            analysis = {
                "question": question,
                "sources_checked": matched_sources,
                "key_terms": list(key_terms),
                "matched_terms": found_terms,
                "coverage_score": round(coverage_ratio, 2),
                "sufficient": is_sufficient,
                "recommendation": (
                    "Evidence appears sufficient to answer this question."
                    if is_sufficient
                    else "Evidence may be incomplete; consider searching with specific missing terms or read adjacent sections."
                ),
            }

            lat = (time.perf_counter() - t0) * 1000.0
            preview = f"Coverage score: {round(coverage_ratio, 2)} - {'Sufficient' if is_sufficient else 'Needs more search'}"
            return ToolResult(
                call_id=call_id,
                tool_name=self.name,
                success=True,
                data=analysis,
                latency_ms=lat,
                items_count=len(matched_sources),
                raw_preview=preview,
            )

        except Exception as e:
            lat = (time.perf_counter() - t0) * 1000.0
            return ToolResult(
                call_id=call_id,
                tool_name=self.name,
                success=False,
                error=str(e),
                latency_ms=lat,
                items_count=0,
                raw_preview=f"Error: {e}",
            )
