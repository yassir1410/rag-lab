import time
from abc import ABC, abstractmethod
from typing import Optional, Callable
from sqlalchemy.orm import Session

from app.rag.agents.schemas import AgentQueryResponse, AgentStrategyType
from app.rag.agents.state import AgentState
from app.rag.agents.controller import AgentController
from app.rag.agents.tool_executor import SafeToolExecutor
from app.rag.agents.evidence import EvidenceCollector


class BaseAgentStrategy(ABC):
    """
    Abstract Base Class for all Phase 10 Agent Retrieval Strategies:
    - Single-Step Agent (Tool-calling baseline)
    - Iterative Retrieval Agent (Multi-turn Plan-Act-Observe loop)
    - Plan-and-Execute Agent (Query decomposition & multi-hop synthesis)
    - Adaptive Routing Agent (Dynamic fast-path vs agentic retrieval)
    """

    @property
    @abstractmethod
    def strategy_type(self) -> AgentStrategyType:
        pass

    @property
    @abstractmethod
    def display_name(self) -> str:
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        pass

    @abstractmethod
    def execute(
        self,
        session: Session,
        state: AgentState,
        controller: AgentController,
        executor: SafeToolExecutor,
        evidence_collector: EvidenceCollector,
        on_step_callback: Optional[Callable[[dict], None]] = None,
    ) -> AgentQueryResponse:
        """
        Executes the specific agentic strategy and returns a fully-traceable AgentQueryResponse.
        """
        pass
