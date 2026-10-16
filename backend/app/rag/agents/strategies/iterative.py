import time
import uuid
from typing import Optional, Callable
from sqlalchemy.orm import Session

from app.rag.agents.strategies.base import BaseAgentStrategy
from app.rag.agents.schemas import (
    AgentStrategyType,
    AgentStopReason,
    AgentActionType,
    AgentStepTrace,
    AgentQueryResponse,
)
from app.rag.agents.state import AgentState
from app.rag.agents.controller import AgentController
from app.rag.agents.tool_executor import SafeToolExecutor
from app.rag.agents.evidence import EvidenceCollector
from app.rag.agents.stopping import StoppingConditionEvaluator


class IterativeRetrievalAgent(BaseAgentStrategy):
    """
    Strategy B: Iterative Retrieval Agent (Multi-turn Plan-Act-Observe loop).
    Loops continuously: inspects current question & accumulated evidence -> selects an approved tool ->
    executes tool -> inspects evidence sufficiency -> repeats until evidence is sufficient or budget halts.
    """

    def __init__(self, stopping_evaluator: Optional[StoppingConditionEvaluator] = None):
        self.stopping = stopping_evaluator or StoppingConditionEvaluator()

    @property
    def strategy_type(self) -> AgentStrategyType:
        return AgentStrategyType.ITERATIVE

    @property
    def display_name(self) -> str:
        return "Iterative Retrieval Agent (Plan-Act-Observe Loop)"

    @property
    def description(self) -> str:
        return "Multi-step retrieval loop gathering evidence incrementally until sufficient or resource budgets halt."

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
        trace_id = str(uuid.uuid4())
        step_traces = []
        stop_reason = AgentStopReason.MAX_ITERATIONS

        available_tools = executor.registry.list_tools()

        for iteration in range(1, state.max_iterations + 1):
            state.iteration_count = iteration

            # 1. Pre-step limits check
            limit_reason = self.stopping.check_pre_step_limits(state)
            if limit_reason:
                stop_reason = limit_reason
                break

            # 2. Decide next action
            t_plan_start = time.perf_counter()
            action, thought, tool_call = controller.decide_next_action(
                state, evidence_collector, available_tools
            )

            # 3. Action = STOP or SYNTHESIZE
            if action in (AgentActionType.STOP, AgentActionType.SYNTHESIZE) or not tool_call:
                stop_reason = (
                    AgentStopReason.SUFFICIENT_EVIDENCE
                    if evidence_collector.total_evidence_count > 0
                    else AgentStopReason.NO_NEW_EVIDENCE
                )
                break

            # 4. Action = TOOL_CALL
            state.record_tool_call(tool_call)
            t_tool_start = time.perf_counter()

            # Execute tool safely
            tool_result = executor.execute_tool(session, tool_call, evidence_collector)
            t_tool_lat = (time.perf_counter() - t_tool_start) * 1000.0

            # Collect evidence
            added_sources = evidence_collector.add_from_tool_result(
                tool_result, step_number=len(step_traces) + 1
            )

            step_trace = AgentStepTrace(
                step_number=len(step_traces) + 1,
                action_type=AgentActionType.TOOL_CALL,
                thought=thought,
                tool_call=tool_call,
                tool_result=tool_result,
                evidence_added=added_sources,
                latency_ms=t_tool_lat,
                timestamp=time.strftime("%H:%M:%S"),
            )
            state.record_step(step_trace)
            step_traces.append(step_trace)

            if on_step_callback:
                on_step_callback(step_trace.model_dump(mode="json"))

            # 5. Post-step check
            post_stop = self.stopping.check_post_step_progress(
                state, evidence_collector, added_sources, tool_result.success
            )
            if post_stop:
                stop_reason = post_stop
                break

        # 6. Final synthesis
        t_synth_start = time.perf_counter()
        final_answer = controller.synthesize_answer(
            state.question, evidence_collector, strategy_label=self.display_name
        )
        valid_citations, _ = evidence_collector.validate_citations(final_answer)

        synth_trace = AgentStepTrace(
            step_number=len(step_traces) + 1,
            action_type=AgentActionType.SYNTHESIZE,
            thought=f"Completed {state.iteration_count} iterations. Synthesized final answer with {len(valid_citations)} citations.",
            latency_ms=(time.perf_counter() - t_synth_start) * 1000.0,
            timestamp=time.strftime("%H:%M:%S"),
        )
        state.record_step(synth_trace)
        step_traces.append(synth_trace)

        if on_step_callback:
            on_step_callback(synth_trace.model_dump(mode="json"))

        total_lat = (time.perf_counter() - t0) * 1000.0
        state.halt(stop_reason, final_answer)

        return AgentQueryResponse(
            trace_id=trace_id,
            question=state.question,
            strategy=self.strategy_type,
            final_answer=final_answer,
            citations=valid_citations,
            stop_reason=stop_reason,
            total_steps=len(step_traces),
            tool_calls_count=state.tool_call_count,
            evidence_count=evidence_collector.total_evidence_count,
            latency_ms=total_lat,
            tokens_used=state.tokens_used,
            steps=step_traces,
            evidence=evidence_collector.items,
        )
