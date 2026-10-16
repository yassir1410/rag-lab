from typing import Dict, List, Any
from app.rag.conversation.memory.base import BaseMemoryStrategy, NoMemoryStrategy
from app.rag.conversation.memory.sliding_window import SlidingWindowMemory
from app.rag.conversation.memory.token_bounded import TokenBoundedMemory
from app.rag.conversation.memory.summary_memory import SummaryMemory


class MemoryStrategyRegistry:
    """Registry providing singleton access to conversational memory strategies."""

    def __init__(self):
        self._strategies: Dict[str, BaseMemoryStrategy] = {
            "no_memory": NoMemoryStrategy(),
            "none": NoMemoryStrategy(),
            "sliding_window": SlidingWindowMemory(),
            "token_bounded": TokenBoundedMemory(),
            "summary_memory": SummaryMemory(),
        }

    def get_strategy(self, name: str = "sliding_window") -> BaseMemoryStrategy:
        key = (name or "sliding_window").lower().strip()
        return self._strategies.get(key, self._strategies["sliding_window"])

    def list_strategies(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": "no_memory",
                "name": "No Memory (Stateless Baseline)",
                "description": "Treats each user query as completely independent without prior conversation context.",
            },
            {
                "id": "sliding_window",
                "name": "Sliding Window Memory",
                "description": "Retains the most recent K dialogue turns up to an explicit memory token ceiling.",
            },
            {
                "id": "token_bounded",
                "name": "Token-Bounded Memory",
                "description": "Dynamically allocates context capacity based on system prompt, retrieved chunks, and generation reservation.",
            },
            {
                "id": "summary_memory",
                "name": "Summary Memory (Hybrid Rolling Summary)",
                "description": "Compresses older messages into a rolling entity-preserving summary while keeping the last N turns verbatim.",
            },
        ]


memory_registry = MemoryStrategyRegistry()

__all__ = [
    "BaseMemoryStrategy",
    "NoMemoryStrategy",
    "SlidingWindowMemory",
    "TokenBoundedMemory",
    "SummaryMemory",
    "MemoryStrategyRegistry",
    "memory_registry",
]
