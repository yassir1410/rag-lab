import uuid
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field


class UnifiedEvidenceItem(BaseModel):
    """
    Standardized documentary evidence model across all retrieval engines.
    Captures origin provenance, heterogeneous score semantics, structured data, and public asset URLs.
    """
    document_id: Optional[uuid.UUID] = None
    document_name: str = "Unknown"
    evidence_id: str = Field(..., description="Globally unique identifier for this evidence item")
    evidence_type: str = Field(..., description="text, table, image, graph, page")
    page_number: Optional[int] = None
    section_id: Optional[str] = None
    engine: str = Field(..., description="text_dense, text_hybrid, multimodal_table, multimodal_visual, graph, agentic")
    raw_score: float = Field(..., description="Score directly from the underlying retrieval engine")
    score_semantics: str = Field(
        ...,
        description="Origin score scale: cosine_similarity, rrf_score, keyword_match, clip_similarity, graph_relevance",
    )
    calibrated_score: float = Field(..., ge=0.0, le=1.0, description="Normalized score calibrated across engines")
    content: str = Field(..., description="Text content, Markdown table representation, or entity relation summary")
    structured_data: Optional[Dict[str, Any]] = Field(default=None, description="Structured table/graph payload")
    image_url: Optional[str] = Field(default=None, description="Public authorized asset URL")
    citation_id: str = Field(..., description="Assigned stable citation token, e.g. [S1]")
    origin_tag: str = Field(..., description="Original domain tag, e.g. [TABLE_P1], [PAGE_1], [CHUNK_12]")
    provenance: Dict[str, Any] = Field(default_factory=dict, description="Detailed engine metadata and index specs")
    was_cited: bool = False

    # Exclude internal container disk path from client-facing serialization
    _local_image_path: Optional[str] = None


class DocumentCapabilityReport(BaseModel):
    """
    Inspects whether a document has been parsed, chunked, embedded, table-extracted, and graph-indexed.
    Prevents running retrieval routes on non-existent indexes.
    """
    document_id: Optional[uuid.UUID] = None
    filename: str = ""
    mime_type: str = ""
    has_text_chunks: bool = False
    has_embeddings: bool = False
    has_tables: bool = False
    has_page_images: bool = False
    has_clip_embeddings: bool = False
    has_knowledge_graph: bool = False
    chunk_count: int = 0
    embedding_count: int = 0
    table_count: int = 0
    page_count: int = 0
    graph_evidence_count: int = 0
    supported_routes: List[str] = Field(default_factory=list)
    recommended_actions: List[str] = Field(default_factory=list)


class RoutingDecision(BaseModel):
    """
    Explainable routing determination made by the Intelligent Query Router.
    """
    mode: str = Field("auto", description="auto, manual, comparative")
    selected_routes: List[str] = Field(default_factory=list)
    routing_reasons: List[str] = Field(default_factory=list)
    intent_detected: List[str] = Field(default_factory=list)
    capabilities_checked: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    fallback_decisions: List[str] = Field(default_factory=list)
    effective_page_number: Optional[int] = Field(None, description="Resolved page filter from request or explicit query text")


class UnifiedCitationDetail(BaseModel):
    citation_id: str = Field(..., description="Citation marker, e.g. [S1]")
    origin_tag: str = Field(..., description="Original evidence tag, e.g. [TABLE_P1]")
    is_valid: bool = Field(..., description="Whether citation identifier exists in retrieved context")
    document_name: Optional[str] = None
    page_number: Optional[int] = None
    evidence_type: Optional[str] = None
    supporting_passage: Optional[str] = None
    calibrated_score: Optional[float] = None
    image_url: Optional[str] = None
    structured_table: Optional[Dict[str, Any]] = None
    caveat: str = "Citation match indicates identifier reference, not proven semantic entailment."


class UnifiedTimingMetrics(BaseModel):
    routing_latency_ms: float = 0.0
    retrieval_latency_ms: float = 0.0
    fusion_latency_ms: float = 0.0
    generation_latency_ms: float = 0.0
    citation_validation_ms: float = 0.0
    total_latency_ms: float = 0.0
    per_route_latency_ms: Dict[str, float] = Field(default_factory=dict)


class UnifiedRAGQueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Natural language user question")
    document_id: Optional[uuid.UUID] = Field(default=None, description="Primary document filter")
    document_ids: Optional[List[uuid.UUID]] = Field(default=None, description="Multi-document filtering")
    page_number: Optional[int] = Field(default=None, ge=1, description="Specific page constraint")
    mode: str = Field(default="auto", description="Routing mode: 'auto', 'manual', or 'comparative'")
    manual_routes: Optional[List[str]] = Field(
        default=None,
        description="Explicit route selection when mode='manual' (text_dense, text_hybrid, multimodal_table, multimodal_visual, graph, agentic)",
    )
    top_k: int = Field(default=5, ge=1, le=20)
    llm_provider: str = Field(default="gemini", description="LLM Provider: 'gemini', 'ollama', 'openai', 'mock'")
    llm_model: Optional[str] = Field(default=None, description="Target model ID. If omitted, default is used.")
    prompt_template: str = Field(default="grounded", description="basic, grounded, concise, detailed")
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_output_tokens: int = Field(default=512, ge=16, le=4096)
    max_context_tokens: int = Field(default=2048, ge=128, le=16384)
    include_assembled_prompt: bool = Field(default=False)


class UnifiedRAGQueryResponse(BaseModel):
    trace_id: str
    question: str
    answer: str
    routing: RoutingDecision
    evidence_items: List[UnifiedEvidenceItem]
    citations: List[UnifiedCitationDetail]
    invalid_citations: List[str] = Field(default_factory=list)
    uncited_sources: List[str] = Field(default_factory=list)
    model_used: str
    is_vision_enabled: bool = False
    timing_metrics: UnifiedTimingMetrics
    assembled_prompt: Optional[str] = None
    warnings: List[str] = Field(default_factory=list)


class RouteComparisonItem(BaseModel):
    route: str
    latency_ms: float
    evidence_count: int
    top_score: float
    top_citation: str
    snippet: str
    answer_preview: Optional[str] = None


class RouteComparisonResponse(BaseModel):
    question: str
    document_id: Optional[uuid.UUID] = None
    comparisons: List[RouteComparisonItem]
    routing_recommendation: str
    total_comparison_latency_ms: float
