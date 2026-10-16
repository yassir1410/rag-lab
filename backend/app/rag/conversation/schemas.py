import uuid
from enum import Enum
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class MemoryStrategyType(str, Enum):
    NO_MEMORY = "no_memory"
    SLIDING_WINDOW = "sliding_window"
    TOKEN_BOUNDED = "token_bounded"
    SUMMARY_MEMORY = "summary_memory"


class MemoryConfig(BaseModel):
    sliding_window_turns: int = Field(default=3, ge=1, le=20, description="Number of complete turns (user + assistant) to retain.")
    include_assistant_replies: bool = Field(default=True, description="Whether to include previous assistant replies in memory.")
    max_memory_tokens: int = Field(default=1200, ge=10, le=8000, description="Hard token ceiling for conversational memory.")
    token_budget_ratio: float = Field(default=0.35, ge=0.05, le=0.80, description="Proportion of available context window allocated to memory.")
    summary_trigger_turns: int = Field(default=4, ge=2, le=20, description="Trigger summarization when unsummarized turns reach this count.")
    keep_recent_turns: int = Field(default=2, ge=1, le=10, description="Recent turns kept verbatim alongside the rolling summary.")


class ConversationSettings(BaseModel):
    memory_strategy: str = Field(default="sliding_window", description="no_memory, sliding_window, token_bounded, summary_memory")
    memory_config: MemoryConfig = Field(default_factory=MemoryConfig)
    retrieval_strategy: str = Field(default="hybrid_rrf", description="Phase 08 strategy (dense_only, lexical_only, hybrid_rrf, etc.)")
    embedding_model_id: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    search_algorithm: str = Field(default="exact", description="exact, hnsw, ivfflat")
    distance_metric: str = Field(default="cosine")
    top_k: int = Field(default=5, ge=1, le=20)
    reranker_model: str = Field(default="cross-encoder/ms-marco-TinyBERT-L-2-v2")
    hybrid_alpha: float = Field(default=0.5, ge=0.0, le=1.0)
    rrf_k: int = Field(default=60, ge=1, le=100)
    enable_query_rewriting: bool = Field(default=True, description="Resolve pronouns and contextual references into standalone retrieval queries.")
    llm_provider: str = Field(default="mock", description="ollama, openai, gemini, mock")
    llm_model: Optional[str] = Field(default="mock-instruct-v1")
    prompt_template: str = Field(default="grounded", description="basic, grounded, concise, detailed")
    temperature: float = Field(default=0.0, ge=0.0, le=1.0)
    max_context_tokens: int = Field(default=2048, ge=256, le=16384)
    max_output_tokens: int = Field(default=512, ge=64, le=4096)
    document_filter: Optional[uuid.UUID] = Field(default=None, description="Optional document ID constraint")


class ConversationCreateRequest(BaseModel):
    title: Optional[str] = Field(default=None, max_length=255)
    settings: Optional[Dict[str, Any]] = Field(default_factory=dict)


class ConversationUpdateRequest(BaseModel):
    title: Optional[str] = Field(default=None, max_length=255)
    settings: Optional[Dict[str, Any]] = None
    status: Optional[str] = Field(default=None, description="'active' or 'archived'")


class ConversationMessageResponse(BaseModel):
    id: uuid.UUID
    conversation_id: uuid.UUID
    role: str
    content: str
    sequence_number: int
    metadata_json: Dict[str, Any] = Field(default_factory=dict)
    trace_id: Optional[str] = None
    created_at: str


class ConversationSummaryResponse(BaseModel):
    id: uuid.UUID
    conversation_id: uuid.UUID
    summary_text: str
    covered_until_sequence: int
    summarizer_model: Optional[str] = None
    metadata_json: Dict[str, Any] = Field(default_factory=dict)
    created_at: str


class ConversationResponse(BaseModel):
    id: uuid.UUID
    title: str
    settings: Dict[str, Any]
    status: str
    created_at: str
    updated_at: str
    message_count: int = 0
    latest_message: Optional[str] = None


class TurnExecutionRequest(BaseModel):
    message: str = Field(..., min_length=1, description="User input message for this turn")
    override_settings: Optional[Dict[str, Any]] = Field(default=None, description="Temporary setting overrides for this turn only")


class MemoryInspectionItem(BaseModel):
    sequence_number: int
    role: str
    content_snippet: str
    token_count: int
    is_included: bool
    inclusion_reason: str


class MemorySelectionResult(BaseModel):
    strategy: str
    total_history_messages: int
    included_messages: List[ConversationMessageResponse]
    excluded_messages: List[ConversationMessageResponse]
    summary_used: Optional[str] = None
    summary_covered_sequence: Optional[int] = None
    allocated_memory_tokens: int
    token_budget: int
    inspection_items: List[MemoryInspectionItem]


class QueryRewriteResult(BaseModel):
    original_query: str
    rewritten_query: str
    is_rewritten: bool
    is_standalone: bool
    detected_references: List[str] = Field(default_factory=list)
    ambiguity_detected: bool = False
    uncertainty_warning: Optional[str] = None
    rewrite_latency_ms: float = 0.0


class TurnExecutionResponse(BaseModel):
    user_message: ConversationMessageResponse
    assistant_message: ConversationMessageResponse
    rewritten_query_info: QueryRewriteResult
    memory_selection: MemorySelectionResult
    sources: List[Dict[str, Any]]
    citations: List[Dict[str, Any]]
    invalid_citations: List[str] = Field(default_factory=list)
    uncited_sources: List[str] = Field(default_factory=list)
    timing_breakdown: Dict[str, float]
    token_metrics: Dict[str, Any]
    warnings: List[str] = Field(default_factory=list)


class StrategyComparisonResult(BaseModel):
    strategy: str
    rewritten_query: str
    memory_tokens: int
    retrieved_chunk_ids: List[str]
    retrieved_chunk_sequences: List[int]
    latency_ms: float
    answer_preview: str


class MultiTurnBenchmarkScenario(BaseModel):
    scenario_id: str
    name: str
    category: str
    description: str
    history_turns: List[Dict[str, str]]
    current_question: str
    expected_standalone_query: str
    relevant_chunk_ids: List[str]
    is_answerable: bool = True
    expected_refusal: bool = False
    notes: Optional[str] = None


class MultiTurnScenarioEvaluationResult(BaseModel):
    scenario_id: str
    name: str
    category: str
    original_query: str
    rewritten_query: str
    expected_query: str
    rewrite_keyword_overlap: float
    retrieved_chunk_ids: List[str]
    target_chunk_ids: List[str]
    recall_at_5: float
    mrr: float
    hit_rate: float
    memory_tokens_used: int
    total_latency_ms: float
    passed: bool
    diagnostics: str
