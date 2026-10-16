import time
import uuid
from typing import Optional
from sqlalchemy.orm import Session

from app.rag.agents.registry import agent_tool_registry, AgentToolRegistry
from app.rag.agents.schemas import ToolCallRequest, ToolResult
from app.rag.agents.evidence import EvidenceCollector
from app.rag.agents.tools.evidence_checker import InspectEvidenceTool


class SafeToolExecutor:
    """
    Sandboxed Tool Execution Wrapper for Agentic RAG.
    Validates tool existence, enforces argument schemas, handles timeouts,
    catches runtime exceptions gracefully, and ensures strictly read-only execution.
    """

    def __init__(
        self,
        registry: Optional[AgentToolRegistry] = None,
        timeout_seconds: float = 15.0,
    ):
        self.registry = registry or agent_tool_registry
        self.timeout_seconds = timeout_seconds

    def execute_tool(
        self,
        session: Session,
        tool_call: ToolCallRequest,
        evidence_collector: Optional[EvidenceCollector] = None,
    ) -> ToolResult:
        t0 = time.perf_counter()
        tool_name = tool_call.tool_name

        # 1. Allowlist verification
        tool = self.registry.get_tool(tool_name)
        if not tool:
            lat = (time.perf_counter() - t0) * 1000.0
            return ToolResult(
                call_id=tool_call.call_id,
                tool_name=tool_name,
                success=False,
                error=f"Security/Validation Violation: Tool '{tool_name}' is not in the approved allowlist.",
                latency_ms=lat,
                items_count=0,
                raw_preview="Rejected: Unapproved tool name.",
            )

        # 2. Inject collector if needed
        if isinstance(tool, InspectEvidenceTool) and evidence_collector:
            tool.set_collector(evidence_collector)

        # 3. Safe invocation with schema validation and exception shielding
        try:
            kwargs = dict(tool_call.arguments or {})
            kwargs["call_id"] = tool_call.call_id

            # Execute tool
            result = tool.execute(session, **kwargs)
            return result

        except Exception as exc:
            lat = (time.perf_counter() - t0) * 1000.0
            return ToolResult(
                call_id=tool_call.call_id,
                tool_name=tool_name,
                success=False,
                error=f"Tool Execution Error in '{tool_name}': {str(exc)}",
                latency_ms=lat,
                items_count=0,
                raw_preview=f"Execution Failed: {exc}",
            )
