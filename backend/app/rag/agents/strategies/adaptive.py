import time
import uuid
import re
from typing import Optional, Callable
from sqlalchemy.orm import Session

from app.rag.agents.strategies.base import BaseAgentStrategy
from app.rag.agents.strategies.single_step import SingleStepAgent
from app.rag.agents.strategies.iterative import IterativeRetrievalAgent
from app.rag.agents.schemas import (
    AgentStrategyType,
    AgentActionType,
    AgentStepTrace,
    AgentQueryResponse,
)
from app.rag.agents.state import AgentState
from app.rag.agents.controller import AgentController
from app.rag.agents.tool_executor import SafeToolExecutor
from app.rag.agents.evidence import EvidenceCollector


class AdaptiveRoutingAgent(BaseAgentStrategy):
    """
    Strategy D: Adaptive Routing Agent.
    Dynamically routes queries: selects a lightweight single-step RAG path for straightforward
    questions to minimize latency, and escalates to iterative multi-step retrieval for
    complex, multi-hop, ambiguous, or multi-topic questions.
    """

    def __init__(self):
        self._single_step_agent = SingleStepAgent()
        self._iterative_agent = IterativeRetrievalAgent()

    @property
    def strategy_type(self) -> AgentStrategyType:
        return AgentStrategyType.ADAPTIVE

    @property
    def display_name(self) -> str:
        return "Adaptive Routing Agent (Dynamic Fast-Path)"

    @property
    def description(self) -> str:
        return "Evaluates query complexity: routes simple queries to fast single-step retrieval, and complex multi-hop queries to iterative retrieval."

    def _classify_complexity(self, question: str) -> bool:
        """
        Returns True if complex / multi-hop, False if simple / direct.
        """
        q_lower = question.lower().strip()

        # Multi-clause or conjunction indicators
        has_conjunction = bool(re.search(r"\b(and|while|also|both|compare|difference|between|after|before)\b", q_lower))
        # Question length
        is_long = len(question.split()) > 10
        # Multi-question indicators
        has_multiple_qs = question.count("?") > 1

        # Multi-hop phrasing
        is_multihop = any(phrase in q_lower for phrase in [
            "which company and", "what technologies and", "where did he work and",
            "experience in both", "skills related to", "how many and what"
        ])

        return has_conjunction or is_long or has_multiple_qs or is_multihop

    def execute(
        self,
        session: Session,
        state: AgentState,
        controller: AgentController,
        executor: SafeToolExecutor,
        evidence_collector: EvidenceCollector,
        on_step_callback: Optional[Callable[[dict], None]] = None,
    ) -> AgentQueryResponse:
        t0 = time.perf_counter()
        is_complex = self._classify_complexity(state.question)

        selected_sub_strategy = "iterative" if is_complex else "single_step"
        routing_thought = (
            f"Adaptive Router Decision: Classified query as {'COMPLEX/MULTI-HOP' if is_complex else 'SIMPLE/DIRECT'}. "
            f"Routing execution to {selected_sub_strategy.upper()} strategy."
        )

        route_trace = AgentStepTrace(
            step_number=1,
            action_type=AgentActionType.PLAN,
            thought=routing_thought,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            timestamp=time.strftime("%H:%M:%S"),
        )
        state.record_step(route_trace)

        if on_step_callback:
            on_step_callback(route_trace.model_dump(mode="json"))

        # Execute chosen sub-strategy
        if is_complex:
            resp = self._iterative_agent.execute(
                session, state, controller, executor, evidence_collector, on_step_callback
            )
        else:
            resp = self._single_step_agent.execute(
                session, state, controller, executor, evidence_collector, on_step_callback
            )

        # Update response strategy to ADAPTIVE while preserving traces
        resp.strategy = self.strategy_type
        # Prepend route trace to response steps if not already present
        if resp.steps and resp.steps[0].action_type != AgentActionType.PLAN:
            resp.steps.insert(0, route_trace)
            resp.total_steps += 1

        return resp
