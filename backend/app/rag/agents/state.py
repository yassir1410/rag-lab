import time
import uuid
from typing import List, Dict, Any, Optional
from app.rag.agents.schemas import (
    AgentStrategyType,
    AgentStopReason,
    ToolCallRequest,
    AgentStepTrace,
)


class AgentState:
    """
    Explicit, observable agent state tracking the Plan-Act-Observe cycle.
    Enforces iteration and tool-call budgets and prevents unbounded execution.
    """

    def __init__(
        self,
        question: str,
        strategy: AgentStrategyType,
        conversation_id: Optional[uuid.UUID] = None,
        max_iterations: int = 4,
        max_tool_calls: int = 8,
        time_budget_seconds: float = 30.0,
        max_context_tokens: int = 4096,
    ):
        self.question = question
        self.strategy = strategy
        self.conversation_id = conversation_id
        self.max_iterations = max_iterations
        self.max_tool_calls = max_tool_calls
        self.time_budget_seconds = time_budget_seconds
        self.max_context_tokens = max_context_tokens

        self.iteration_count: int = 0
        self.tool_call_count: int = 0
        self.queries_executed: List[str] = []
        self.tool_calls_history: List[ToolCallRequest] = []
        self.step_traces: List[AgentStepTrace] = []
        self.sub_questions: List[Dict[str, Any]] = []

        self.start_time: float = time.perf_counter()
        self.status: str = "running"
        self.stop_reason: Optional[AgentStopReason] = None
        self.final_answer: Optional[str] = None
        self.tokens_used: int = 0
        self.metadata: Dict[str, Any] = {}

    @property
    def elapsed_seconds(self) -> float:
        return time.perf_counter() - self.start_time

    @property
    def latency_ms(self) -> float:
        return self.elapsed_seconds * 1000.0

    @property
    def remaining_iterations(self) -> int:
        return max(0, self.max_iterations - self.iteration_count)

    @property
    def remaining_tool_calls(self) -> int:
        return max(0, self.max_tool_calls - self.tool_call_count)

    def record_step(self, step: AgentStepTrace):
        self.step_traces.append(step)

    def record_tool_call(self, tool_call: ToolCallRequest):
        self.tool_call_count += 1
        self.tool_calls_history.append(tool_call)

    def record_query(self, query: str):
        if query not in self.queries_executed:
            self.queries_executed.append(query)

    def is_query_duplicate(self, query: str) -> bool:
        normalized = query.lower().strip()
        return any(q.lower().strip() == normalized for q in self.queries_executed)

    def halt(self, reason: AgentStopReason, answer: Optional[str] = None):
        self.status = "halted"
        self.stop_reason = reason
        if answer:
            self.final_answer = answer
