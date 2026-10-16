import uuid
from enum import Enum
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, ConfigDict


class AgentStrategyType(str, Enum):
    SINGLE_STEP = "single_step"
    ITERATIVE = "iterative"
    PLAN_EXECUTE = "plan_execute"
    ADAPTIVE = "adaptive"


class AgentStopReason(str, Enum):
    SUFFICIENT_EVIDENCE = "SUFFICIENT_EVIDENCE"
    MAX_ITERATIONS = "MAX_ITERATIONS"
    MAX_TOOL_CALLS = "MAX_TOOL_CALLS"
    TIME_BUDGET_EXCEEDED = "TIME_BUDGET_EXCEEDED"
    NO_NEW_EVIDENCE = "NO_NEW_EVIDENCE"
    TOOL_FAILURE = "TOOL_FAILURE"
    USER_CANCELLED = "USER_CANCELLED"
    DIRECT_ANSWER = "DIRECT_ANSWER"


class AgentActionType(str, Enum):
    PLAN = "PLAN"
    TOOL_CALL = "TOOL_CALL"
    OBSERVE = "OBSERVE"
    SYNTHESIZE = "SYNTHESIZE"
    STOP = "STOP"
    ERROR = "ERROR"


class ToolDefinitionSchema(BaseModel):
    name: str
    description: str
    parameters_schema: Dict[str, Any]
    is_read_only: bool = True
    category: str = "retrieval"
    example_call: Optional[Dict[str, Any]] = None


class ToolCallRequest(BaseModel):
    call_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    tool_name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    call_id: str
    tool_name: str
    success: bool
    data: Any = None
    error: Optional[str] = None
    latency_ms: float = 0.0
    items_count: int = 0
    raw_preview: Optional[str] = None


class EvidenceItem(BaseModel):
    source_id: str  # Stable label, e.g. "[S1]"
    document_id: str
    document_name: str
    chunk_id: str
    section_id: Optional[str] = None
    page_number: Optional[int] = None
    content: str
    score: float = 0.0
    retrieved_by_tool: str
    retrieved_at_step: int
    token_count: int = 0


class AgentStepTrace(BaseModel):
    step_number: int
    action_type: AgentActionType
    thought: Optional[str] = None
    tool_call: Optional[ToolCallRequest] = None
    tool_result: Optional[ToolResult] = None
    evidence_added: List[str] = Field(default_factory=list)  # list of source_ids
    latency_ms: float = 0.0
    timestamp: str = ""


class AgentQueryRequest(BaseModel):
    question: str = Field(..., min_length=2, max_length=2000)
    conversation_id: Optional[uuid.UUID] = None
    strategy: AgentStrategyType = AgentStrategyType.ITERATIVE
    max_iterations: int = Field(default=4, ge=1, le=10)
    max_tool_calls: int = Field(default=8, ge=1, le=20)
    time_budget_seconds: float = Field(default=30.0, ge=1.0, le=120.0)
    max_context_tokens: int = Field(default=4096, ge=512, le=16384)
    provider_id: str = "mock"
    model_id: str = "mock-instruct-v1"
    temperature: float = Field(default=0.0, ge=0.0, le=1.0)
    document_filters: Optional[Dict[str, Any]] = None


class AgentQueryResponse(BaseModel):
    trace_id: str
    question: str
    strategy: AgentStrategyType
    final_answer: str
    citations: List[Dict[str, Any]] = Field(default_factory=list)
    stop_reason: AgentStopReason
    total_steps: int
    tool_calls_count: int
    evidence_count: int
    latency_ms: float
    tokens_used: Optional[int] = None
    steps: List[AgentStepTrace] = Field(default_factory=list)
    evidence: List[EvidenceItem] = Field(default_factory=list)

    model_config = ConfigDict(arbitrary_types_allowed=True)


class StrategyComparisonRequest(BaseModel):
    question: str = Field(..., min_length=2, max_length=2000)
    strategies: Optional[List[AgentStrategyType]] = None
    provider_id: str = "mock"
    model_id: str = "mock-instruct-v1"
    max_iterations: int = 4
    max_tool_calls: int = 8


class StrategyComparisonItem(BaseModel):
    strategy: str
    latency_ms: float
    total_steps: int
    tool_calls_count: int
    evidence_count: int
    stop_reason: str
    answer_snippet: str
    citation_count: int
    success: bool
    error: Optional[str] = None


class StrategyComparisonResponse(BaseModel):
    question: str
    results: Dict[str, StrategyComparisonItem]
