from typing import List, Optional
from app.models.conversation import ConversationMessage, ConversationSummary
from app.rag.conversation.schemas import (
    MemoryConfig,
    MemorySelectionResult,
    ConversationMessageResponse,
    MemoryInspectionItem,
)
from app.rag.conversation.memory.base import BaseMemoryStrategy, message_to_schema


class SummaryMemory(BaseMemoryStrategy):
    """
    Summary Memory Strategy:
    Maintains a high-density rolling summary of older conversation turns while
    keeping the most recent K turns verbatim.
    Prevents token bloat on long conversations while retaining critical conversational context.
    """
    strategy_id = "summary_memory"
    name = "Summary Memory (Hybrid Rolling Summary)"
    description = "Compresses older dialogue into an entity-preserving summary while keeping the last N turns verbatim."

    def select_memory(
        self,
        messages: List[ConversationMessage],
        summary: Optional[ConversationSummary],
        config: MemoryConfig,
        total_context_budget: int = 4096,
        prompt_and_retrieval_tokens: int = 0,
    ) -> MemorySelectionResult:
        keep_recent = config.keep_recent_turns
        max_tokens = config.max_memory_tokens

        # Separate recent messages from summarized messages
        # Count user messages from end to identify the last N turns
        recent_messages: List[ConversationMessage] = []
        user_turn_count = 0

        for m in reversed(messages):
            recent_messages.append(m)
            if m.role == "user":
                user_turn_count += 1
                if user_turn_count >= keep_recent:
                    break

        recent_messages.reverse()
        recent_ids = {m.id for m in recent_messages}

        # Calculate tokens for summary and recent messages
        summary_text = summary.summary_text if summary else None
        summary_seq = summary.covered_until_sequence if summary else None
        summary_tokens = self.count_tokens(summary_text) if summary_text else 0

        accumulated_tokens = summary_tokens
        included_messages: List[ConversationMessage] = []

        for m in recent_messages:
            m_tokens = self.count_tokens(m.content)
            if accumulated_tokens + m_tokens <= max_tokens:
                included_messages.append(m)
                accumulated_tokens += m_tokens

        included_ids = {m.id for m in included_messages}

        included_schemas: List[ConversationMessageResponse] = []
        excluded_schemas: List[ConversationMessageResponse] = []
        inspection_items: List[MemoryInspectionItem] = []

        for m in messages:
            t_cnt = self.count_tokens(m.content)
            is_inc = m.id in included_ids

            if is_inc:
                included_schemas.append(message_to_schema(m))
                reason = "Retained as verbatim recent turn."
            else:
                excluded_schemas.append(message_to_schema(m))
                if summary_seq and m.sequence_number <= summary_seq:
                    reason = f"Compressed into rolling summary (covered up to seq {summary_seq})."
                elif m.id in recent_ids:
                    reason = f"Excluded: Exceeds summary memory token ceiling ({max_tokens} tokens)."
                else:
                    reason = "Older turn awaiting background summarization."

            inspection_items.append(
                MemoryInspectionItem(
                    sequence_number=m.sequence_number,
                    role=m.role,
                    content_snippet=m.content[:80] + ("..." if len(m.content) > 80 else ""),
                    token_count=t_cnt,
                    is_included=is_inc,
                    inclusion_reason=reason,
                )
            )

        return MemorySelectionResult(
            strategy=self.strategy_id,
            total_history_messages=len(messages),
            included_messages=included_schemas,
            excluded_messages=excluded_schemas,
            summary_used=summary_text,
            summary_covered_sequence=summary_seq,
            allocated_memory_tokens=accumulated_tokens,
            token_budget=max_tokens,
            inspection_items=inspection_items,
        )
