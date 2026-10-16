from typing import Dict, List, Optional
from app.rag.chunking.base import ChunkingStrategy
from app.rag.chunking.strategies.fixed_size import FixedSizeChunking
from app.rag.chunking.strategies.recursive import RecursiveChunking
from app.rag.chunking.strategies.sentence import SentenceChunking
from app.rag.chunking.strategies.semantic import SemanticChunking
from app.rag.chunking.strategies.parent_child import ParentChildChunking
from app.rag.chunking.strategies.structure_aware import StructureAwareChunking


class ChunkingStrategyRegistry:
    """Registry managing available chunking algorithms."""

    def __init__(self):
        self._strategies: Dict[str, ChunkingStrategy] = {}
        self._register_defaults()

    def _register_defaults(self):
        self.register(FixedSizeChunking())
        self.register(RecursiveChunking())
        self.register(SentenceChunking())
        self.register(SemanticChunking())
        self.register(ParentChildChunking())
        self.register(StructureAwareChunking())

    def register(self, strategy: ChunkingStrategy):
        self._strategies[strategy.strategy_name] = strategy

    def get(self, strategy_name: str) -> ChunkingStrategy:
        if strategy_name not in self._strategies:
            valid = ", ".join(self._strategies.keys())
            raise ValueError(
                f"Unknown chunking strategy '{strategy_name}'. Valid strategies: {valid}"
            )
        return self._strategies[strategy_name]

    def list_strategies(self) -> List[Dict[str, any]]:
        return [
            {
                "strategy_name": s.strategy_name,
                "display_name": s.display_name,
                "description": s.description,
                "default_config": s.default_config,
            }
            for s in self._strategies.values()
        ]


# Singleton default registry
default_chunking_registry = ChunkingStrategyRegistry()
