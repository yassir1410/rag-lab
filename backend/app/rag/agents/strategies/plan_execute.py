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
    ToolCallRequest,
)
from app.rag.agents.state import AgentState
from app.rag.agents.controller import AgentController
from app.rag.agents.tool_executor import SafeToolExecutor
from app.rag.agents.evidence import EvidenceCollector


class PlanAndExecuteAgent(BaseAgentStrategy):
    """
    Strategy C: Plan-and-Execute Agent.
    Produces a small, bounded decomposition plan (2-3 targeted sub-questions),
    executes approved retrieval tools for each sub-question sequentially,
    aggregates evidence across sub-queries, and synthesizes a unified cited response.
    """

    @property
    def strategy_type(self) -> AgentStrategyType:
        return AgentStrategyType.PLAN_EXECUTE

    @property
    def display_name(self) -> str:
        return "Plan-and-Execute Agent (Query Decomposition)"

    @property
    def description(self) -> str:
        return "Decomposes complex multi-hop questions into sub-questions, executes focused retrieval per step, and synthesizes a cited answer."

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

        # 1. Plan sub-questions
        t_plan_start = time.perf_counter()
        sub_questions = controller.plan_sub_questions(state.question)
        state.sub_questions = [{"sub_q": q, "status": "pending"} for q in sub_questions]
        t_plan_lat = (time.perf_counter() - t_plan_start) * 1000.0

        plan_trace = AgentStepTrace(
            step_number=1,
            action_type=AgentActionType.PLAN,
            thought=f"Decomposed question into {len(sub_questions)} sub-questions: {', '.join(sub_questions)}",
            latency_ms=t_plan_lat,
            timestamp=time.strftime("%H:%M:%S"),
        )
        state.record_step(plan_trace)
        step_traces.append(plan_trace)

        if on_step_callback:
            on_step_callback(plan_trace.model_dump(mode="json"))

        # 2. Execute retrieval per sub-question
        for idx, sub_q in enumerate(sub_questions, start=1):
            if state.tool_call_count >= state.max_tool_calls:
                break
            if state.elapsed_seconds >= state.time_budget_seconds:
                break

            # Choose tool: if sub-q has specific tech keywords -> hybrid_search, else dense_search
            q_lower = sub_q.lower()
            if any(term in q_lower for term in ["react", "python", "fastapi", "postgresql", "docker", "redis"]):
                tool_name = "hybrid_search"
                args = {"query": sub_q, "strategy": "hybrid_rrf", "top_k": 3}
            else:
                tool_name = "dense_search"
                args = {"query": sub_q, "top_k": 3}

            tool_call = ToolCallRequest(tool_name=tool_name, arguments=args)
            state.record_tool_call(tool_call)

            t_step_start = time.perf_counter()
            tool_result = executor.execute_tool(session, tool_call, evidence_collector)
            t_step_lat = (time.perf_counter() - t_step_start) * 1000.0

            added_sources = evidence_collector.add_from_tool_result(
                tool_result, step_number=len(step_traces) + 1
            )

            step_trace = AgentStepTrace(
                step_number=len(step_traces) + 1,
                action_type=AgentActionType.TOOL_CALL,
                thought=f"Executing plan item {idx}/{len(sub_questions)}: '{sub_q}'",
                tool_call=tool_call,
                tool_result=tool_result,
                evidence_added=added_sources,
                latency_ms=t_step_lat,
                timestamp=time.strftime("%H:%M:%S"),
            )
            state.record_step(step_trace)
            step_traces.append(step_trace)

            if on_step_callback:
                on_step_callback(step_trace.model_dump(mode="json"))

        # 3. Synthesize final multi-hop response
        t_synth_start = time.perf_counter()
        final_answer = controller.synthesize_answer(
            state.question, evidence_collector, strategy_label=self.display_name
        )
        valid_citations, _ = evidence_collector.validate_citations(final_answer)

        synth_trace = AgentStepTrace(
            step_number=len(step_traces) + 1,
            action_type=AgentActionType.SYNTHESIZE,
            thought=f"Synthesized unified response covering all sub-questions with {len(valid_citations)} citations.",
            latency_ms=(time.perf_counter() - t_synth_start) * 1000.0,
            timestamp=time.strftime("%H:%M:%S"),
        )
        state.record_step(synth_trace)
        step_traces.append(synth_trace)

        if on_step_callback:
            on_step_callback(synth_trace.model_dump(mode="json"))

        stop_reason = (
            AgentStopReason.SUFFICIENT_EVIDENCE
            if evidence_collector.total_evidence_count > 0
            else AgentStopReason.NO_NEW_EVIDENCE
        )

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
