from typing import Optional, List
from app.rag.agents.schemas import AgentStopReason
from app.rag.agents.state import AgentState
from app.rag.agents.evidence import EvidenceCollector


class StoppingConditionEvaluator:
    """
    Evaluates stopping criteria for Agentic RAG loops.
    Guarantees that the agent terminates safely and reports an explicit stop reason.
    """

    def check_pre_step_limits(self, state: AgentState) -> Optional[AgentStopReason]:
        """
        Evaluates whether execution must halt before starting another iteration or tool call.
        """
        # 1. Time budget exceeded
        if state.elapsed_seconds >= state.time_budget_seconds:
            return AgentStopReason.TIME_BUDGET_EXCEEDED

        # 2. Maximum iterations reached
        if state.iteration_count >= state.max_iterations:
            return AgentStopReason.MAX_ITERATIONS

        # 3. Maximum tool calls reached
        if state.tool_call_count >= state.max_tool_calls:
            return AgentStopReason.MAX_TOOL_CALLS

        return None

    def check_post_step_progress(
        self,
        state: AgentState,
        evidence_collector: EvidenceCollector,
        recent_evidence_added: List[str],
        tool_success: bool,
    ) -> Optional[AgentStopReason]:
        """
        Evaluates progress after a tool execution.
        """
        # 1. Unrecoverable tool failure
        if not tool_success and state.tool_call_count >= state.max_tool_calls:
            return AgentStopReason.TOOL_FAILURE

        # 2. Resource budgets
        pre_check = self.check_pre_step_limits(state)
        if pre_check:
            return pre_check

        return None
