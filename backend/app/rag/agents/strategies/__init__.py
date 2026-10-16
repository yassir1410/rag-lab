from typing import Dict, List, Optional
from app.rag.agents.schemas import AgentStrategyType
from app.rag.agents.strategies.base import BaseAgentStrategy
from app.rag.agents.strategies.single_step import SingleStepAgent
from app.rag.agents.strategies.iterative import IterativeRetrievalAgent
from app.rag.agents.strategies.plan_execute import PlanAndExecuteAgent
from app.rag.agents.strategies.adaptive import AdaptiveRoutingAgent


class AgentStrategyRegistry:
    """
    Registry for Phase 10 Agent Retrieval Strategies.
    """

    def __init__(self):
        self._strategies: Dict[AgentStrategyType, BaseAgentStrategy] = {
            AgentStrategyType.SINGLE_STEP: SingleStepAgent(),
            AgentStrategyType.ITERATIVE: IterativeRetrievalAgent(),
            AgentStrategyType.PLAN_EXECUTE: PlanAndExecuteAgent(),
            AgentStrategyType.ADAPTIVE: AdaptiveRoutingAgent(),
        }

    def get(self, strategy_type: AgentStrategyType) -> Optional[BaseAgentStrategy]:
        return self._strategies.get(strategy_type)

    def list_strategies(self) -> List[Dict[str, str]]:
        return [
            {
                "id": s.strategy_type.value,
                "name": s.display_name,
                "description": s.description,
            }
            for s in self._strategies.values()
        ]


# Singleton instance
agent_strategy_registry = AgentStrategyRegistry()

__all__ = [
    "BaseAgentStrategy",
    "SingleStepAgent",
    "IterativeRetrievalAgent",
    "PlanAndExecuteAgent",
    "AdaptiveRoutingAgent",
    "AgentStrategyRegistry",
    "agent_strategy_registry",
]
