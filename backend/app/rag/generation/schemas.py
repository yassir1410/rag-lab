import uuid
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from app.rag.retrieval.schemas import RetrievalFilter


class ContextSourceItem(BaseModel):
    citation_id: str = Field(..., description="Stable citation identifier, e.g. [S1]")
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    document_name: str
    mime_type: str = "text/plain"
    sequence_index: int = 0
    token_count: int = 0
    char_count: int = 0
    content: str
    page_number: Optional[int] = None
    similarity_score: float = 0.0
    raw_distance: float = 0.0
    distance_metric: str = "cosine"
    was_cited: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CitationDetail(BaseModel):
    citation_id: str = Field(..., description="e.g. [S1]")
    is_valid: bool = Field(..., description="Whether citation ID exists in retrieved context")
    document_name: Optional[str] = None
    page_number: Optional[int] = None
    chunk_id: Optional[uuid.UUID] = None
    supporting_passage: Optional[str] = None
    similarity_score: Optional[float] = None
    caveat: str = "Citation match indicates identifier reference, not proven semantic entailment."


class PipelineTimingMetrics(BaseModel):
    embedding_latency_ms: float = 0.0
    retrieval_latency_ms: float = 0.0
    context_assembly_ms: float = 0.0
    prompt_building_ms: float = 0.0
    llm_inference_ms: float = 0.0
    model_load_latency_ms: Optional[float] = None
    citation_validation_ms: float = 0.0
    total_pipeline_latency_ms: float = 0.0


class TokenAllocationMetrics(BaseModel):
    system_prompt_tokens: int = 0
    user_question_tokens: int = 0
    context_tokens: int = 0
    reserved_output_tokens: int = 0
    total_prompt_tokens: int = 0
    model_context_window: int = 4096
    remaining_capacity: int = 0
    tokenizer_name: str = "gpt_subword_regex"
    is_approximate: bool = True


class RAGQueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Natural language question")
    embedding_model_id: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Target embedding model for vector retrieval",
    )
    search_algorithm: str = Field(
        default="exact",
        description="Vector search algorithm: 'exact', 'hnsw', or 'ivfflat'",
    )
    distance_metric: str = Field(
        default="cosine",
        description="Distance metric: 'cosine', 'l2', or 'inner_product'",
    )
    retrieval_strategy: str = Field(
        default="dense_only",
        description="Strategy: 'dense_only', 'lexical_only', 'hybrid_linear', 'hybrid_rrf', 'dense_rerank', 'hybrid_rrf_rerank', 'multi_query_hybrid_rerank', 'adaptive_context'",
    )
    hybrid_alpha: float = Field(default=0.5, ge=0.0, le=1.0)
    rrf_k: int = Field(default=60, ge=1, le=500)
    reranker_model: str = Field(default="none")
    enable_query_expansion: bool = Field(default=False)
    enable_neighbor_expansion: bool = Field(default=False)
    top_k: int = Field(default=5, ge=1, le=20, description="Number of retrieved chunks")
    filters: Optional[RetrievalFilter] = None
    llm_provider: str = Field(
        default="ollama",
        description="LLM provider: 'ollama', 'openai', 'gemini', or 'mock'",
    )
    llm_model: Optional[str] = Field(
        default=None,
        description="Target LLM model ID. If omitted, provider default is used.",
    )
    prompt_template: str = Field(
        default="grounded",
        description="Prompt template: 'basic', 'grounded', 'concise', or 'detailed'",
    )
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_output_tokens: int = Field(default=512, ge=16, le=4096)
    max_context_tokens: int = Field(default=2048, ge=128, le=16384)
    ef_search: Optional[int] = Field(default=40, ge=1, le=500)
    probes: Optional[int] = Field(default=1, ge=1, le=100)
    include_assembled_prompt: bool = Field(
        default=True, description="Whether to include full prompt in trace for inspection"
    )


class RAGQueryResponse(BaseModel):
    trace_id: str
    question: str
    answer: str
    llm_provider: str
    llm_model: str
    prompt_template: str
    sources: List[ContextSourceItem]
    citations: List[CitationDetail]
    invalid_citations: List[str] = Field(default_factory=list)
    uncited_sources: List[str] = Field(default_factory=list)
    timing_metrics: PipelineTimingMetrics
    token_metrics: TokenAllocationMetrics
    assembled_prompt: Optional[str] = None
    warnings: List[str] = Field(default_factory=list)


class ProviderInfoResponse(BaseModel):
    provider_id: str
    display_name: str
    is_available: bool
    status_message: str
    default_model: str
    supported_models: List[str]


class ModelInfoResponse(BaseModel):
    model_id: str
    display_name: str
    provider_id: str
    context_window: int
    parameter_count: str
    quantization: str
    supports_streaming: bool
    is_available: bool
    description: str


class PromptTemplateResponse(BaseModel):
    id: str
    name: str
    description: str
    system_prompt_preview: str
    user_prompt_preview: str
