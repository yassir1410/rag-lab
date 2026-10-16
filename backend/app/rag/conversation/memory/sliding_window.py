from typing import List, Optional
from app.models.conversation import ConversationMessage, ConversationSummary
from app.rag.conversation.schemas import (
    MemoryConfig,
    MemorySelectionResult,
    ConversationMessageResponse,
    MemoryInspectionItem,
)
from app.rag.conversation.memory.base import BaseMemoryStrategy, message_to_schema


class SlidingWindowMemory(BaseMemoryStrategy):
    """
    Sliding Window Memory Strategy:
    Maintains a fixed-size window of the most recent conversation turns.
    A 'turn' represents an exchange between user and assistant.
    Enforces a strict memory token ceiling without silently truncating individual messages.
    """
    strategy_id = "sliding_window"
    name = "Sliding Window Memory"
    description = "Retains the last K conversation turns (user + assistant) up to a max token threshold."

    def select_memory(
        self,
        messages: List[ConversationMessage],
        summary: Optional[ConversationSummary],
        config: MemoryConfig,
        total_context_budget: int = 4096,
        prompt_and_retrieval_tokens: int = 0,
    ) -> MemorySelectionResult:
        max_turns = config.sliding_window_turns
        include_assistant = config.include_assistant_replies
        max_tokens = config.max_memory_tokens

        # Filter out assistant messages if excluded
        filtered_candidates: List[ConversationMessage] = []
        for m in messages:
            if m.role == "assistant" and not include_assistant:
                continue
            filtered_candidates.append(m)

        # Count turns in reverse chronological order
        # A turn boundary is recognized when moving past a user message
        selected_subset: List[ConversationMessage] = []
        accumulated_tokens = 0
        turn_count = 0
        current_turn_seen_user = False

        for m in reversed(filtered_candidates):
            msg_tokens = self.count_tokens(m.content)

            # Check if adding this message exceeds the memory token limit
            if accumulated_tokens + msg_tokens > max_tokens:
                break

            # Check turn limit before appending message from an older turn
            if turn_count >= max_turns:
                break

            selected_subset.append(m)
            accumulated_tokens += msg_tokens

            if m.role == "user":
                turn_count += 1

        # Restore chronological ordering
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
                reason = "Within sliding window turns and token budget."
            else:
                excluded_schemas.append(message_to_schema(m))
                if m.role == "assistant" and not include_assistant:
                    reason = "Excluded by 'include_assistant_replies=False' setting."
                elif accumulated_tokens + t_cnt > max_tokens:
                    reason = f"Exceeds sliding window token ceiling ({max_tokens} tokens)."
                else:
                    reason = f"Older than the last {max_turns} turns window."

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
            token_budget=max_tokens,
            inspection_items=inspection_items,
        )
