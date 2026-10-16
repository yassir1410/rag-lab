from typing import List, Optional
from app.models.conversation import ConversationMessage, ConversationSummary
from app.rag.conversation.schemas import (
    MemoryConfig,
    MemorySelectionResult,
    ConversationMessageResponse,
    MemoryInspectionItem,
)
from app.rag.conversation.memory.base import BaseMemoryStrategy, message_to_schema


class TokenBoundedMemory(BaseMemoryStrategy):
    """
    Token-Bounded Memory Strategy:
    Dynamically computes the available memory budget based on:
    Total Window - System Instructions - Retrieved Context - User Query - Output Reservation.
    Packs as many complete recent turns as fit in the remaining token envelope.
    """
    strategy_id = "token_bounded"
    name = "Token-Bounded Memory"
    description = "Allocates history dynamically based on remaining model context after retrieval and system prompts."

    def select_memory(
        self,
        messages: List[ConversationMessage],
        summary: Optional[ConversationSummary],
        config: MemoryConfig,
        total_context_budget: int = 4096,
        prompt_and_retrieval_tokens: int = 0,
    ) -> MemorySelectionResult:
        # Calculate available token envelope
        # If prompt_and_retrieval_tokens is provided, use remaining envelope
        # Otherwise fallback to configured ratio of total budget
        if prompt_and_retrieval_tokens > 0:
            available_memory_tokens = max(0, total_context_budget - prompt_and_retrieval_tokens)
        else:
            available_memory_tokens = int(total_context_budget * config.token_budget_ratio)

        # Apply hard upper ceiling from config if lower
        effective_budget = min(available_memory_tokens, config.max_memory_tokens)

        # Select messages in reverse chronological order
        selected_subset: List[ConversationMessage] = []
        accumulated_tokens = 0

        for m in reversed(messages):
            msg_tokens = self.count_tokens(m.content)
            if accumulated_tokens + msg_tokens > effective_budget:
                break

            selected_subset.append(m)
            accumulated_tokens += msg_tokens

        # Restore chronological order
        selected_subset.reverse()
        selected_ids = {m.id for m in selected_subset}

        included_schemas: List[ConversationMessageResponse] = []
        excluded_schemas: List[ConversationMessageResponse] = []
        inspection_items: List[MemoryInspectionItem] = []

        for m in messages:
            t_cnt = self.count_tokens(m.content)
            is_inc = m.id in selected_ids

            if is_inc:
                included_schemas.append(message_to_schema(m))
                reason = f"Fits within dynamic token envelope ({accumulated_tokens}/{effective_budget} tokens)."
            else:
                excluded_schemas.append(message_to_schema(m))
                reason = f"Excluded: Would exceed dynamic token budget of {effective_budget} tokens."

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
            summary_used=None,
            summary_covered_sequence=None,
            allocated_memory_tokens=accumulated_tokens,
            token_budget=effective_budget,
            inspection_items=inspection_items,
        )
