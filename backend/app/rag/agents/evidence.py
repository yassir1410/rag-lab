import hashlib
import re
from typing import List, Dict, Any, Optional, Tuple, Set
from app.rag.agents.schemas import EvidenceItem, ToolResult


class EvidenceCollector:
    """
    Source-aware Evidence Collector for Agentic RAG.
    Aggregates multi-hop retrieval outputs, deduplicates chunks,
    assigns stable citation markers [S1], [S2]..., tracks tool provenance,
    and validates final model citations against hallucinations.
    """

    def __init__(self):
        self._items_by_source_id: Dict[str, EvidenceItem] = {}
        self._seen_chunk_ids: Set[str] = set()
        self._seen_content_hashes: Set[str] = set()
        self._next_source_idx: int = 1
        self._unanswered_sub_questions: List[str] = []

    @property
    def items(self) -> List[EvidenceItem]:
        return list(self._items_by_source_id.values())

    @property
    def total_evidence_count(self) -> int:
        return len(self._items_by_source_id)

    @property
    def total_tokens(self) -> int:
        return sum(item.token_count for item in self._items_by_source_id.values())

    def get_by_source_id(self, source_id: str) -> Optional[EvidenceItem]:
        # Support both "[S1]" and "S1"
        normalized = source_id if source_id.startswith("[") else f"[{source_id}]"
        return self._items_by_source_id.get(normalized)

    def _hash_content(self, text: str) -> str:
        # Normalized snippet hash ignoring minor whitespace differences
        cleaned = "".join(text.lower().split())
        return hashlib.sha256(cleaned.encode("utf-8")).hexdigest()[:16]

    def add_from_tool_result(
        self,
        tool_result: ToolResult,
        step_number: int,
    ) -> List[str]:
        """
        Parses ToolResult data and adds new evidence items if not duplicate.
        Returns the list of newly assigned source IDs (e.g. ['[S1]', '[S2]']).
        """
        if not tool_result.success or not tool_result.data:
            return []

        data = tool_result.data
        if isinstance(data, dict):
            raw_items = [data]
        elif isinstance(data, list):
            raw_items = data
        else:
            return []

        added_source_ids: List[str] = []

        for entry in raw_items:
            if not isinstance(entry, dict):
                continue

            content = entry.get("content", "")
            if not content or len(content.strip()) < 10:
                continue

            chunk_id = entry.get("chunk_id") or str(entry.get("section_id") or "")
            content_hash = self._hash_content(content)

            # Deduplication check
            if chunk_id and chunk_id in self._seen_chunk_ids:
                continue
            if content_hash in self._seen_content_hashes:
                continue

            # Record seen
            if chunk_id:
                self._seen_chunk_ids.add(chunk_id)
            self._seen_content_hashes.add(content_hash)

            source_id = f"[S{self._next_source_idx}]"
            self._next_source_idx += 1

            token_count = entry.get("token_count") or max(1, len(content.split()))

            item = EvidenceItem(
                source_id=source_id,
                document_id=str(entry.get("document_id", "unknown")),
                document_name=entry.get("document_name") or entry.get("filename") or "Unknown Document",
                chunk_id=chunk_id or f"chunk-{self._next_source_idx}",
                section_id=str(entry.get("section_id")) if entry.get("section_id") else None,
                page_number=entry.get("page_number") or entry.get("sequence_index"),
                content=content,
                score=float(entry.get("score", 0.0)),
                retrieved_by_tool=tool_result.tool_name,
                retrieved_at_step=step_number,
                token_count=token_count,
            )

            self._items_by_source_id[source_id] = item
            added_source_ids.append(source_id)

        return added_source_ids

    def build_evidence_context(self, max_tokens: int = 3500) -> str:
        """
        Formats all accumulated documentary evidence into a strictly segregated
        reference block for LLM synthesis.
        """
        if not self._items_by_source_id:
            return "No documentary evidence has been retrieved yet."

        blocks = []
        current_tokens = 0

        for sid, item in self._items_by_source_id.items():
            header = f"--- Source {sid} (Document: {item.document_name}, Page/Seq: {item.page_number or 'N/A'}, Tool: {item.retrieved_by_tool}) ---"
            snippet = f"{header}\n{item.content.strip()}\n"
            token_est = len(snippet.split())

            if current_tokens + token_est > max_tokens and len(blocks) > 0:
                blocks.append(f"\n[Note: Additional evidence truncated to fit context budget of {max_tokens} tokens]")
                break

            blocks.append(snippet)
            current_tokens += token_est

        return "\n".join(blocks)

    def validate_citations(self, answer_text: str) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        Scans generated answer text for citations (e.g. [S1], [S2]).
        Returns:
            - valid_citations: list of citation objects pointing to real evidence
            - hallucinated_citations: list of citation strings not present in evidence
        """
        cited_markers = re.findall(r"\[S\d+\]", answer_text)
        unique_markers = sorted(list(set(cited_markers)))

        valid_citations: List[Dict[str, Any]] = []
        hallucinated_citations: List[str] = []

        for marker in unique_markers:
            item = self._items_by_source_id.get(marker)
            if item:
                valid_citations.append({
                    "source_id": item.source_id,
                    "document_id": item.document_id,
                    "document_name": item.document_name,
                    "chunk_id": item.chunk_id,
                    "page_number": item.page_number,
                    "retrieved_by_tool": item.retrieved_by_tool,
                    "snippet": item.content[:180] + "..." if len(item.content) > 180 else item.content,
                })
            else:
                hallucinated_citations.append(marker)

        return valid_citations, hallucinated_citations

    def set_unanswered_sub_questions(self, sub_qs: List[str]):
        self._unanswered_sub_questions = list(sub_qs)

    @property
    def unanswered_sub_questions(self) -> List[str]:
        return self._unanswered_sub_questions
