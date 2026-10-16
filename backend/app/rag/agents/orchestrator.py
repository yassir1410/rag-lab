import asyncio
import json
import time
import uuid
from typing import AsyncIterator, Optional, List, Dict, Any
from sqlalchemy.orm import Session

from app.rag.agents.schemas import (
    AgentQueryRequest,
    AgentQueryResponse,
    AgentStrategyType,
    StrategyComparisonRequest,
    StrategyComparisonResponse,
    StrategyComparisonItem,
)
from app.rag.agents.state import AgentState
from app.rag.agents.evidence import EvidenceCollector
from app.rag.agents.controller import AgentController
from app.rag.agents.tool_executor import SafeToolExecutor
from app.rag.agents.registry import agent_tool_registry
from app.rag.agents.strategies import agent_strategy_registry
from app.rag.agents.tracing import agent_trace_manager


class AgentOrchestrator:
    """
    Central Orchestrator for Phase 10 Agentic RAG.
    Coordinates execution of allowlisted tools, agent strategies, Plan-Act-Observe loops,
    real-time SSE streaming, and persistent trace logging.
    """

    def __init__(self):
        self.tool_registry = agent_tool_registry
        self.strategy_registry = agent_strategy_registry
        self.trace_manager = agent_trace_manager

    def run_query(
        self,
        session: Session,
        request: AgentQueryRequest,
    ) -> AgentQueryResponse:
        """
        Executes an agent query synchronously and returns the complete response with trace.
        """
        strategy = self.strategy_registry.get(request.strategy)
        if not strategy:
            raise ValueError(f"Unknown agent strategy: '{request.strategy}'")

        state = AgentState(
            question=request.question,
            strategy=request.strategy,
            conversation_id=request.conversation_id,
            max_iterations=request.max_iterations,
            max_tool_calls=request.max_tool_calls,
            time_budget_seconds=request.time_budget_seconds,
            max_context_tokens=request.max_context_tokens,
        )

        controller = AgentController(
            provider_id=request.provider_id,
            model_id=request.model_id,
            tool_registry=self.tool_registry,
        )
        executor = SafeToolExecutor(registry=self.tool_registry)
        evidence_collector = EvidenceCollector()

        # Execute strategy
        response = strategy.execute(
            session=session,
            state=state,
            controller=controller,
            executor=executor,
            evidence_collector=evidence_collector,
        )

        # Persist trace
        self.trace_manager.save_trace(
            session=session,
            response=response,
            conversation_id=request.conversation_id,
        )

        return response

    async def stream_query(
        self,
        session: Session,
        request: AgentQueryRequest,
    ) -> AsyncIterator[str]:
        """
        Asynchronously streams step-by-step Plan-Act-Observe telemetry and the final answer
        as Server-Sent Events (SSE).
        """
        queue: asyncio.Queue[str] = asyncio.Queue()

        strategy = self.strategy_registry.get(request.strategy)
        if not strategy:
            err_payload = json.dumps({"error": f"Unknown agent strategy: '{request.strategy}'"})
            yield f"event: error\ndata: {err_payload}\n\n"
            return

        state = AgentState(
            question=request.question,
            strategy=request.strategy,
            conversation_id=request.conversation_id,
            max_iterations=request.max_iterations,
            max_tool_calls=request.max_tool_calls,
            time_budget_seconds=request.time_budget_seconds,
            max_context_tokens=request.max_context_tokens,
        )

        controller = AgentController(
            provider_id=request.provider_id,
            model_id=request.model_id,
            tool_registry=self.tool_registry,
        )
        executor = SafeToolExecutor(registry=self.tool_registry)
        evidence_collector = EvidenceCollector()

        # Send initial metadata
        init_meta = {
            "question": request.question,
            "strategy": request.strategy.value,
            "max_iterations": request.max_iterations,
            "max_tool_calls": request.max_tool_calls,
            "time_budget_seconds": request.time_budget_seconds,
        }
        yield f"event: meta\ndata: {json.dumps(init_meta)}\n\n"

        def on_step_callback(step_dict: dict):
            # Put step event onto the async stream
            msg = f"event: step\ndata: {json.dumps(step_dict)}\n\n"
            queue.put_nowait(msg)

        # Run strategy in thread pool to avoid blocking async event loop
        loop = asyncio.get_running_loop()

        def sync_worker():
            try:
                resp = strategy.execute(
                    session=session,
                    state=state,
                    controller=controller,
                    executor=executor,
                    evidence_collector=evidence_collector,
                    on_step_callback=on_step_callback,
                )
                self.trace_manager.save_trace(
                    session=session,
                    response=resp,
                    conversation_id=request.conversation_id,
                )
                return resp
            except Exception as e:
                return e

        worker_future = loop.run_in_executor(None, sync_worker)

        # Yield step events as they arrive
        while not worker_future.done():
            while not queue.empty():
                yield queue.get_nowait()
            await asyncio.sleep(0.05)

        # Drain remaining step events
        while not queue.empty():
            yield queue.get_nowait()

        result = await worker_future
        if isinstance(result, Exception):
            err_data = json.dumps({"error": str(result)})
            yield f"event: error\ndata: {err_data}\n\n"
            return

        # Stream final answer in tokens
        words = result.final_answer.split(" ")
        for i, word in enumerate(words):
            token_chunk = word + (" " if i < len(words) - 1 else "")
            payload = json.dumps({"token": token_chunk})
            yield f"event: token\ndata: {payload}\n\n"
            await asyncio.sleep(0.01)

        # Stream evidence and citations
        citations_payload = json.dumps({
            "citations": result.citations,
            "evidence_count": result.evidence_count,
            "stop_reason": result.stop_reason.value if hasattr(result.stop_reason, "value") else str(result.stop_reason),
        })
        yield f"event: citations\ndata: {citations_payload}\n\n"

        # Final done event
        done_payload = json.dumps({
            "trace_id": result.trace_id,
            "total_steps": result.total_steps,
            "tool_calls_count": result.tool_calls_count,
            "latency_ms": result.latency_ms,
            "tokens_used": result.tokens_used,
        })
        yield f"event: done\ndata: {done_payload}\n\n"

    def compare_strategies(
        self,
        session: Session,
        request: StrategyComparisonRequest,
    ) -> StrategyComparisonResponse:
        """
        Executes a question across multiple agent strategies sequentially
        and returns comparative metrics.
        """
        strategies_to_test = request.strategies or [
            AgentStrategyType.SINGLE_STEP,
            AgentStrategyType.ITERATIVE,
            AgentStrategyType.PLAN_EXECUTE,
            AgentStrategyType.ADAPTIVE,
        ]

        results: Dict[str, StrategyComparisonItem] = {}

        for strat_type in strategies_to_test:
            q_req = AgentQueryRequest(
                question=request.question,
                strategy=strat_type,
                provider_id=request.provider_id,
                model_id=request.model_id,
                max_iterations=request.max_iterations,
                max_tool_calls=request.max_tool_calls,
            )
            try:
                resp = self.run_query(session, q_req)
                snippet = resp.final_answer[:160] + "..." if len(resp.final_answer) > 160 else resp.final_answer
                results[strat_type.value] = StrategyComparisonItem(
                    strategy=strat_type.value,
                    latency_ms=round(resp.latency_ms, 2),
                    total_steps=resp.total_steps,
                    tool_calls_count=resp.tool_calls_count,
                    evidence_count=resp.evidence_count,
                    stop_reason=resp.stop_reason.value if hasattr(resp.stop_reason, "value") else str(resp.stop_reason),
                    answer_snippet=snippet,
                    citation_count=len(resp.citations),
                    success=True,
                )
            except Exception as e:
                results[strat_type.value] = StrategyComparisonItem(
                    strategy=strat_type.value,
                    latency_ms=0.0,
                    total_steps=0,
                    tool_calls_count=0,
                    evidence_count=0,
                    stop_reason="ERROR",
                    answer_snippet="",
                    citation_count=0,
                    success=False,
                    error=str(e),
                )

        return StrategyComparisonResponse(
            question=request.question,
            results=results,
        )


# Singleton instance
agent_orchestrator = AgentOrchestrator()
