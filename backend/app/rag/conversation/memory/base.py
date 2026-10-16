import abc
from typing import List, Optional, Tuple, Dict, Any
from app.models.conversation import ConversationMessage, ConversationSummary
from app.rag.chunking.tokenizer import default_tokenizer
from app.rag.conversation.schemas import (
    MemoryConfig,
    MemorySelectionResult,
    ConversationMessageResponse,
    MemoryInspectionItem,
)


def message_to_schema(msg: ConversationMessage) -> ConversationMessageResponse:
    return ConversationMessageResponse(
        id=msg.id,
        conversation_id=msg.conversation_id,
        role=msg.role,
        content=msg.content,
        sequence_number=msg.sequence_number,
        metadata_json=msg.metadata_json or {},
        trace_id=msg.trace_id,
        created_at=msg.created_at.isoformat() if hasattr(msg.created_at, "isoformat") else str(msg.created_at),
    )


class BaseMemoryStrategy(abc.ABC):
    """
    Abstract interface for conversational memory management.
    Selects, summarizes, or bounds conversation history according to specific criteria.
    """

    strategy_id: str = "base"
    name: str = "Base Memory"
    description: str = "Abstract base memory strategy."

    def __init__(self):
        self.tokenizer = default_tokenizer

    def count_tokens(self, text: str) -> int:
        return self.tokenizer.count_tokens(text)

    @abc.abstractmethod
    def select_memory(
        self,
        messages: List[ConversationMessage],
        summary: Optional[ConversationSummary],
        config: MemoryConfig,
        total_context_budget: int = 4096,
        prompt_and_retrieval_tokens: int = 0,
    ) -> MemorySelectionResult:
        """
        Selects a subset of conversation history messages or rolling summary.
        Returns MemorySelectionResult containing included and excluded messages.
        """
        pass


class NoMemoryStrategy(BaseMemoryStrategy):
    """
    No Memory (Stateless RAG Baseline):
    Excludes all past conversation turns, treating each query in complete isolation.
    """
    strategy_id = "no_memory"
    name = "No Memory (Stateless Baseline)"
    description = "Treats each user message as a fresh, independent turn without conversation history."

    def select_memory(
        self,
        messages: List[ConversationMessage],
        summary: Optional[ConversationSummary],
        config: MemoryConfig,
        total_context_budget: int = 4096,
        prompt_and_retrieval_tokens: int = 0,
    ) -> MemorySelectionResult:
        inspection_items: List[MemoryInspectionItem] = []
        excluded_schemas: List[ConversationMessageResponse] = []

        for m in messages:
            t_cnt = self.count_tokens(m.content)
            excluded_schemas.append(message_to_schema(m))
            inspection_items.append(
                MemoryInspectionItem(
                    sequence_number=m.sequence_number,
                    role=m.role,
                    content_snippet=m.content[:80] + ("..." if len(m.content) > 80 else ""),
                    token_count=t_cnt,
                    is_included=False,
                    inclusion_reason="Excluded by No Memory strategy policy.",
                )
            )

        return MemorySelectionResult(
            strategy=self.strategy_id,
            total_history_messages=len(messages),
            included_messages=[],
            excluded_messages=excluded_schemas,
            summary_used=None,
            summary_covered_sequence=None,
            allocated_memory_tokens=0,
            token_budget=0,
            inspection_items=inspection_items,
        )
