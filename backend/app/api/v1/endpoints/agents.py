from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.rag.agents.schemas import (
    AgentQueryRequest,
    AgentQueryResponse,
    StrategyComparisonRequest,
    StrategyComparisonResponse,
    ToolDefinitionSchema,
    AgentStrategyType,
)
from app.rag.agents.orchestrator import agent_orchestrator
from app.rag.agents.registry import agent_tool_registry
from app.rag.agents.strategies import agent_strategy_registry
from app.rag.agents.tracing import agent_trace_manager
from app.rag.agents.evaluation import agent_benchmark_runner, AgentBenchmarkReport

router = APIRouter()


@router.get("/strategies", response_model=List[Dict[str, str]])
def list_agent_strategies():
    """
    Returns the list of supported Agentic RAG retrieval strategies.
    """
    return agent_strategy_registry.list_strategies()


@router.get("/tools", response_model=List[ToolDefinitionSchema])
def list_agent_tools():
    """
    Returns all approved read-only tools with their typed JSON schemas and descriptions.
    """
    return agent_tool_registry.list_tools()


@router.post("/query", response_model=AgentQueryResponse)
def execute_agent_query(
    request: AgentQueryRequest,
    db: Session = Depends(get_db),
):
    """
    Executes a bounded, multi-step Agentic RAG query synchronously.
    """
    try:
        return agent_orchestrator.run_query(db, request)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Agent execution failed: {str(exc)}")


@router.post("/query/stream")
async def stream_agent_query(
    request: AgentQueryRequest,
    db: Session = Depends(get_db),
):
    """
    Streams step-by-step Plan-Act-Observe telemetry and the final answer via Server-Sent Events (SSE).
    """
    return StreamingResponse(
        agent_orchestrator.stream_query(db, request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/traces/{trace_id}", response_model=Dict[str, Any])
def get_agent_trace(
    trace_id: str,
    db: Session = Depends(get_db),
):
    """
    Retrieves a persistent Agent Execution Trace by ID.
    """
    trace = agent_trace_manager.get_trace(db, trace_id)
    if not trace:
        raise HTTPException(status_code=404, detail=f"Trace with ID '{trace_id}' not found.")
    return trace


@router.post("/compare", response_model=StrategyComparisonResponse)
def compare_agent_strategies(
    request: StrategyComparisonRequest,
    db: Session = Depends(get_db),
):
    """
    Executes a question across multiple agent strategies sequentially to compare latency,
    tool calls, evidence count, and citation grounding.
    """
    try:
        return agent_orchestrator.compare_strategies(db, request)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Strategy comparison failed: {str(exc)}")


@router.post("/benchmark", response_model=AgentBenchmarkReport)
def run_agent_benchmark(
    strategy: AgentStrategyType = Query(default=AgentStrategyType.ITERATIVE),
    provider_id: str = Query(default="mock"),
    model_id: str = Query(default="mock-instruct-v1"),
    db: Session = Depends(get_db),
):
    """
    Runs the 15-scenario synthetic evaluation benchmark for the selected strategy.
    """
    try:
        return agent_benchmark_runner.run_benchmark(
            session=db,
            strategy=strategy,
            provider_id=provider_id,
            model_id=model_id,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Benchmark run failed: {str(exc)}")
