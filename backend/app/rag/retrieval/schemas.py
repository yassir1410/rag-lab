import uuid
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class RetrievalFilter(BaseModel):
    document_id: Optional[uuid.UUID] = None
    mime_type: Optional[str] = None
    chunking_experiment_id: Optional[uuid.UUID] = None
    model_id: Optional[str] = None
    page_number: Optional[int] = None


class CandidateAuditItem(BaseModel):
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    document_name: str
    sequence_index: int
    content_snippet: str
    dense_rank: Optional[int] = None
    dense_score: Optional[float] = None
    lexical_rank: Optional[int] = None
    lexical_score: Optional[float] = None
    rrf_score: Optional[float] = None
    linear_score: Optional[float] = None
    rerank_score: Optional[float] = None
    final_rank: int = 1


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Natural language search query")
    strategy: str = Field(
        default="dense_only",
        description="Retrieval strategy: 'dense_only', 'lexical_only', 'hybrid_linear', 'hybrid_rrf', 'dense_rerank', 'hybrid_rrf_rerank', 'multi_query_hybrid_rerank', 'adaptive_context'",
    )
    model_id: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Target embedding model ID",
    )
    search_algorithm: str = Field(
        default="exact",
        description="Dense vector search algorithm: 'exact', 'hnsw', or 'ivfflat'",
    )
    distance_metric: str = Field(
        default="cosine",
        description="Distance metric: 'cosine', 'l2', or 'inner_product'",
    )
    top_k: int = Field(default=5, ge=1, le=100, description="Number of results to retrieve")
    filters: Optional[RetrievalFilter] = None
    hybrid_alpha: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Dense vs lexical weight for linear blend (1.0 = pure dense)"
    )
    rrf_k: int = Field(default=60, ge=1, le=500, description="Reciprocal Rank Fusion smoothing constant")
    reranker_model: str = Field(
        default="none", description="Reranker: 'none', 'tinybert', or 'mock'"
    )
    enable_query_expansion: bool = Field(default=False, description="Enable multi-query decomposition")
    enable_neighbor_expansion: bool = Field(default=False, description="Enable +-1 contiguous chunk expansion")
    language: str = Field(default="english", description="FTS dictionary language ('english' or 'french')")
    ef_search: Optional[int] = Field(
        default=40, ge=1, le=500, description="HNSW search depth (higher = higher recall, slower)"
    )
    probes: Optional[int] = Field(
        default=1, ge=1, le=200, description="IVFFlat clusters to probe (higher = higher recall, slower)"
    )
    explain_plan: bool = Field(
        default=True, description="Whether to include EXPLAIN ANALYZE execution plan"
    )


class SearchResultItem(BaseModel):
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    document_name: str
    mime_type: str
    sequence_index: int
    token_count: int
    char_count: int
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    page_number: Optional[int] = None
    raw_distance: float
    similarity_score: float
    distance_metric: str
    distance_label: str
    similarity_label: str


class ExecutionPlanSummary(BaseModel):
    node_type: str
    planning_time_ms: float
    execution_time_ms: float
    actual_rows: int
    total_cost: float
    plan_tree_summary: str
    raw_plan: Optional[Dict[str, Any]] = None


class SearchResponse(BaseModel):
    query: str
    strategy: str = "dense_only"
    model_id: str
    search_algorithm: str
    distance_metric: str
    top_k: int
    total_candidates: int
    results: List[SearchResultItem]
    candidate_progression: List[CandidateAuditItem] = Field(default_factory=list)
    query_latency_ms: float
    embedding_latency_ms: float
    rerank_latency_ms: float = 0.0
    total_latency_ms: float
    timing_breakdown: Dict[str, float] = Field(default_factory=dict)
    expanded_queries: List[str] = Field(default_factory=list)
    execution_plan: Optional[ExecutionPlanSummary] = None
    warnings: List[str] = Field(default_factory=list)


class RetrievalStrategySpec(BaseModel):
    id: str
    name: str
    description: str
    uses_dense: bool
    uses_lexical: bool
    uses_fusion: bool
    uses_reranker: bool
    uses_query_expansion: bool
    uses_context_optimization: bool


class LexicalSearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=100)
    filters: Optional[RetrievalFilter] = None
    language: str = Field(default="english")


class HybridSearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    model_id: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    search_algorithm: str = Field(default="exact")
    distance_metric: str = Field(default="cosine")
    top_k: int = Field(default=5, ge=1, le=100)
    fusion_method: str = Field(default="rrf", description="'rrf' or 'linear'")
    hybrid_alpha: float = Field(default=0.5, ge=0.0, le=1.0)
    rrf_k: int = Field(default=60, ge=1, le=500)
    reranker_model: str = Field(default="none")
    filters: Optional[RetrievalFilter] = None
    language: str = Field(default="english")


class RerankRequest(BaseModel):
    query: str
    reranker_model: str = Field(default="tinybert")
    candidates: List[SearchResultItem]
    top_k: int = Field(default=5, ge=1, le=50)


class RerankResponse(BaseModel):
    query: str
    reranker_model: str
    results: List[SearchResultItem]
    latency_ms: float


class IndexCreateRequest(BaseModel):
    index_type: str = Field(..., description="'hnsw' or 'ivfflat'")
    distance_metric: str = Field(default="cosine", description="'cosine', 'l2', or 'inner_product'")
    model_id: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    embedding_experiment_id: Optional[uuid.UUID] = None
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="HNSW: {'m': 16, 'ef_construction': 64}; IVFFlat: {'lists': 10}",
    )


class IndexResponse(BaseModel):
    id: uuid.UUID
    index_name: str
    index_type: str
    distance_metric: str
    opclass: str
    vector_dimension: int
    model_id: str
    embedding_experiment_id: Optional[uuid.UUID] = None
    parameters: Dict[str, Any]
    status: str
    build_duration_ms: float
    index_size_bytes: int
    index_size_pretty: str
    error_message: Optional[str] = None
    created_at: str


class BenchmarkRequest(BaseModel):
    query: str = Field(default="What are the key technical skills and architecture patterns?")
    model_id: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    distance_metric: str = Field(default="cosine")
    top_k: int = Field(default=5, ge=1, le=50)
    filters: Optional[RetrievalFilter] = None
    hnsw_ef_search: int = Field(default=40, ge=1, le=500)
    ivfflat_probes: int = Field(default=1, ge=1, le=100)
    runs: int = Field(default=5, ge=1, le=20, description="Repeated executions for percentile metrics")


class AlgorithmBenchmarkResult(BaseModel):
    algorithm: str
    latency_p50_ms: float
    latency_p95_ms: float
    latency_mean_ms: float
    throughput_qps: float
    recall_at_k: float
    jaccard_overlap: float
    plan_node_type: str
    retrieved_chunk_ids: List[str]
    index_build_ms: Optional[float] = None
    explain_plan_summary: Optional[str] = None


class BenchmarkResponse(BaseModel):
    query: str
    model_id: str
    distance_metric: str
    top_k: int
    total_vectors_searched: int
    algorithms: Dict[str, AlgorithmBenchmarkResult]
    embedding_latency_ms: Optional[float] = None
    warnings: List[str] = Field(default_factory=list)


class RetrievalExperimentCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=150)
    query_text: str
    model_id: str
    distance_metric: str
    top_k: int = 5
    filters: Dict[str, Any] = Field(default_factory=dict)
    results: List[Dict[str, Any]] = Field(default_factory=list)
    metrics: Dict[str, Any] = Field(default_factory=dict)


class RetrievalExperimentResponse(BaseModel):
    id: uuid.UUID
    name: str
    query_text: str
    model_id: str
    distance_metric: str
    top_k: int
    filters: Dict[str, Any]
    results: List[Dict[str, Any]]
    metrics: Dict[str, Any]
    created_at: str
