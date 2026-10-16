import pytest
import uuid
from fastapi.testclient import TestClient

from app.main import app
from app.db.session import SessionLocal
from app.rag.agents.registry import agent_tool_registry
from app.rag.agents.tool_executor import SafeToolExecutor
from app.rag.agents.schemas import (
    ToolCallRequest,
    AgentQueryRequest,
    AgentStrategyType,
    AgentStopReason,
)
from app.rag.agents.state import AgentState
from app.rag.agents.evidence import EvidenceCollector
from app.rag.agents.stopping import StoppingConditionEvaluator
from app.rag.agents.controller import AgentController
from app.rag.agents.strategies import (
    SingleStepAgent,
    IterativeRetrievalAgent,
    PlanAndExecuteAgent,
    AdaptiveRoutingAgent,
)
from app.rag.agents.evaluation import agent_benchmark_runner

client = TestClient(app)


@pytest.fixture
def db_session():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def test_tool_registry_and_schema_validation():
    """Verify tool allowlist and parameter schema exports."""
    tool_names = agent_tool_registry.list_tool_names()
    expected_tools = [
        "dense_search",
        "hybrid_search",
        "read_document_section",
        "inspect_document_metadata",
        "inspect_evidence",
    ]
    for t in expected_tools:
        assert t in tool_names, f"Tool '{t}' missing from allowlisted registry"

    tools_defs = agent_tool_registry.list_tools()
    assert len(tools_defs) == 5
    for td in tools_defs:
        assert td.is_read_only is True
        assert "properties" in td.parameters_schema


def test_safe_tool_executor_sandboxing(db_session):
    """Verify safe sandboxing: unapproved tools are rejected; approved tools execute cleanly."""
    executor = SafeToolExecutor(registry=agent_tool_registry)

    # 1. Unapproved tool rejection
    unapproved_call = ToolCallRequest(
        tool_name="drop_database",
        arguments={"table": "documents"},
    )
    res = executor.execute_tool(db_session, unapproved_call)
    assert res.success is False
    assert "Security/Validation Violation" in res.error

    # 2. Approved metadata tool execution
    meta_call = ToolCallRequest(
        tool_name="inspect_document_metadata",
        arguments={},
    )
    res_meta = executor.execute_tool(db_session, meta_call)
    assert res_meta.success is True
    assert isinstance(res_meta.data, list)


def test_evidence_collector_deduplication_and_citations():
    """Verify deduplication, stable source IDs [S1], [S2], and citation validation."""
    collector = EvidenceCollector()

    # Create dummy tool results
    from app.rag.agents.schemas import ToolResult
    res1 = ToolResult(
        call_id="c1",
        tool_name="dense_search",
        success=True,
        data=[
            {"chunk_id": "chunk-1", "content": "The candidate has 5 years of Python experience.", "score": 0.88},
            {"chunk_id": "chunk-2", "content": "Deploys microservices with FastAPI and Docker.", "score": 0.82},
        ],
    )
    added1 = collector.add_from_tool_result(res1, step_number=1)
    assert added1 == ["[S1]", "[S2]"]
    assert collector.total_evidence_count == 2

    # Add duplicate chunk
    res2 = ToolResult(
        call_id="c2",
        tool_name="hybrid_search",
        success=True,
        data=[
            {"chunk_id": "chunk-1", "content": "The candidate has 5 years of Python experience.", "score": 0.95},
            {"chunk_id": "chunk-3", "content": "Expert in PostgreSQL and pgvector indexing.", "score": 0.79},
        ],
    )
    added2 = collector.add_from_tool_result(res2, step_number=2)
    assert added2 == ["[S3]"]  # chunk-1 was deduplicated
    assert collector.total_evidence_count == 3

    # Validate citations
    answer_text = "The candidate writes Python [S1] and designs PostgreSQL databases [S3] using LangChain [S99]."
    valid, hallucinated = collector.validate_citations(answer_text)
    assert len(valid) == 2
    assert [v["source_id"] for v in valid] == ["[S1]", "[S3]"]
    assert hallucinated == ["[S99]"]


def test_stopping_conditions_evaluator():
    """Verify resource bounding halts loops deterministically."""
    evaluator = StoppingConditionEvaluator()
    state = AgentState(
        question="Test query",
        strategy=AgentStrategyType.ITERATIVE,
        max_iterations=2,
        max_tool_calls=3,
        time_budget_seconds=10.0,
    )

    # Initial state should not halt
    assert evaluator.check_pre_step_limits(state) is None

    # Iteration limit reached
    state.iteration_count = 2
    assert evaluator.check_pre_step_limits(state) == AgentStopReason.MAX_ITERATIONS

    # Tool calls limit reached
    state.iteration_count = 1
    state.tool_call_count = 3
    assert evaluator.check_pre_step_limits(state) == AgentStopReason.MAX_TOOL_CALLS


def test_single_step_agent_execution(db_session):
    """Verify Strategy A: SingleStepAgent."""
    agent = SingleStepAgent()
    state = AgentState(question="What skills does the candidate have in Python?", strategy=AgentStrategyType.SINGLE_STEP)
    controller = AgentController(provider_id="mock", tool_registry=agent_tool_registry)
    executor = SafeToolExecutor(registry=agent_tool_registry)
    collector = EvidenceCollector()

    resp = agent.execute(db_session, state, controller, executor, collector)
    assert resp.strategy == AgentStrategyType.SINGLE_STEP
    assert resp.total_steps >= 1
    assert resp.final_answer is not None
    assert resp.latency_ms > 0


def test_iterative_agent_execution(db_session):
    """Verify Strategy B: IterativeRetrievalAgent multi-hop loop."""
    agent = IterativeRetrievalAgent()
    state = AgentState(
        question="Which company did the candidate work at, and what messaging tools were used?",
        strategy=AgentStrategyType.ITERATIVE,
        max_iterations=3,
    )
    controller = AgentController(provider_id="mock", tool_registry=agent_tool_registry)
    executor = SafeToolExecutor(registry=agent_tool_registry)
    collector = EvidenceCollector()

    resp = agent.execute(db_session, state, controller, executor, collector)
    assert resp.strategy == AgentStrategyType.ITERATIVE
    assert resp.total_steps >= 2
    assert resp.stop_reason in (AgentStopReason.SUFFICIENT_EVIDENCE, AgentStopReason.MAX_ITERATIONS)


def test_plan_and_execute_agent_decomposition(db_session):
    """Verify Strategy C: PlanAndExecuteAgent query decomposition."""
    agent = PlanAndExecuteAgent()
    state = AgentState(
        question="Compare the candidate's university education and backend experience.",
        strategy=AgentStrategyType.PLAN_EXECUTE,
    )
    controller = AgentController(provider_id="mock", tool_registry=agent_tool_registry)
    executor = SafeToolExecutor(registry=agent_tool_registry)
    collector = EvidenceCollector()

    resp = agent.execute(db_session, state, controller, executor, collector)
    assert resp.strategy == AgentStrategyType.PLAN_EXECUTE
    assert any(step.action_type.value == "PLAN" for step in resp.steps)
    assert resp.final_answer is not None


def test_adaptive_routing_agent(db_session):
    """Verify Strategy D: AdaptiveRoutingAgent routing logic."""
    agent = AdaptiveRoutingAgent()

    # Simple question -> single step
    state_simple = AgentState(question="What is Python?", strategy=AgentStrategyType.ADAPTIVE)
    controller = AgentController(provider_id="mock", tool_registry=agent_tool_registry)
    executor = SafeToolExecutor(registry=agent_tool_registry)
    collector = EvidenceCollector()

    resp_simple = agent.execute(db_session, state_simple, controller, executor, collector)
    assert resp_simple.strategy == AgentStrategyType.ADAPTIVE
    assert resp_simple.steps[0].action_type.value == "PLAN"
    assert "SIMPLE/DIRECT" in resp_simple.steps[0].thought


def test_agent_api_endpoints():
    """Verify REST API endpoints under /api/v1/agents."""
    # 1. Strategies
    res_strat = client.get("/api/v1/agents/strategies")
    assert res_strat.status_code == 200
    strat_ids = [s["id"] for s in res_strat.json()]
    assert "iterative" in strat_ids
    assert "plan_execute" in strat_ids

    # 2. Tools
    res_tools = client.get("/api/v1/agents/tools")
    assert res_tools.status_code == 200
    tool_names = [t["name"] for t in res_tools.json()]
    assert "dense_search" in tool_names
    assert "hybrid_search" in tool_names

    # 3. Synchronous Query
    payload = {
        "question": "What is the candidate's backend stack?",
        "strategy": "single_step",
        "provider_id": "mock",
        "model_id": "mock-instruct-v1",
        "max_iterations": 2,
    }
    res_query = client.post("/api/v1/agents/query", json=payload)
    assert res_query.status_code == 200
    data = res_query.json()
    assert "trace_id" in data
    assert data["strategy"] == "single_step"
    trace_id = data["trace_id"]

    # 4. Get Trace
    res_trace = client.get(f"/api/v1/agents/traces/{trace_id}")
    assert res_trace.status_code == 200
    assert res_trace.json()["trace_id"] == trace_id

    # 5. Compare Strategies
    cmp_payload = {
        "question": "What technologies are used?",
        "strategies": ["single_step", "iterative"],
        "provider_id": "mock",
        "model_id": "mock-instruct-v1",
    }
    res_cmp = client.post("/api/v1/agents/compare", json=cmp_payload)
    assert res_cmp.status_code == 200
    assert "single_step" in res_cmp.json()["results"]


def test_agent_benchmark_runner(db_session):
    """Verify 15-scenario synthetic evaluation benchmark."""
    report = agent_benchmark_runner.run_benchmark(
        session=db_session,
        strategy=AgentStrategyType.SINGLE_STEP,
        provider_id="mock",
        scenario_ids=["SC-01", "SC-05", "SC-15"],
    )
    assert report.total_scenarios == 3
    assert report.passed_scenarios >= 2
    assert len(report.scenario_results) == 3
