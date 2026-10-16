import uuid
from enum import Enum
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, ConfigDict


class EntityType(str, Enum):
    PERSON = "PERSON"
    ORGANIZATION = "ORGANIZATION"
    PROJECT = "PROJECT"
    TECHNOLOGY = "TECHNOLOGY"
    JOB_ROLE = "JOB_ROLE"
    LOCATION = "LOCATION"
    DATE = "DATE"
    CERTIFICATION = "CERTIFICATION"


class RelationType(str, Enum):
    WORKED_AT = "WORKED_AT"
    DEVELOPED = "DEVELOPED"
    USES = "USES"
    HAS_SKILL = "HAS_SKILL"
    STUDIED_AT = "STUDIED_AT"
    BELONGS_TO = "BELONGS_TO"
    LOCATED_IN = "LOCATED_IN"
    HOLDS_ROLE = "HOLDS_ROLE"
    CERTIFIED_IN = "CERTIFIED_IN"


class GraphRetrievalStrategy(str, Enum):
    VECTOR_ONLY = "vector_only"
    ENTITY_GRAPH = "entity_graph"
    GRAPH_TRAVERSAL = "graph_traversal"
    HYBRID_GRAPH_VECTOR = "hybrid_graph_vector"
    COMMUNITY_BASED = "community_based"


# Entity Schemas
class EntityBase(BaseModel):
    canonical_name: str = Field(..., min_length=1, max_length=255)
    entity_type: str = Field(...)
    aliases: List[str] = Field(default_factory=list)
    properties: Dict[str, Any] = Field(default_factory=dict)
    extraction_method: str = "rule"


class EntityCreate(EntityBase):
    pass


class EntityResponse(EntityBase):
    id: uuid.UUID
    created_at: str
    updated_at: str
    evidence_count: int = 0
    degree: int = 0

    model_config = ConfigDict(from_attributes=True)


# Relation Schemas
class RelationBase(BaseModel):
    source_entity_id: uuid.UUID
    target_entity_id: uuid.UUID
    relation_type: str
    properties: Dict[str, Any] = Field(default_factory=dict)
    extraction_method: str = "rule"
    confidence: Optional[float] = None


class RelationCreate(RelationBase):
    pass


class RelationResponse(RelationBase):
    id: uuid.UUID
    source_name: Optional[str] = None
    target_name: Optional[str] = None
    created_at: str
    evidence_count: int = 0

    model_config = ConfigDict(from_attributes=True)


# Evidence Schemas
class EvidenceResponse(BaseModel):
    id: uuid.UUID
    entity_id: Optional[uuid.UUID] = None
    relation_id: Optional[uuid.UUID] = None
    document_id: uuid.UUID
    chunk_id: Optional[uuid.UUID] = None
    section_id: Optional[uuid.UUID] = None
    evidence_text: str
    page_reference: Optional[int] = None
    extraction_run_id: Optional[uuid.UUID] = None

    model_config = ConfigDict(from_attributes=True)


# Extraction Run Schemas
class ExtractionRunRequest(BaseModel):
    document_id: uuid.UUID
    strategy: str = "rule"  # "rule", "llm", "hybrid"
    model_id: str = "mock-instruct-v1"
    provider_id: str = "mock"
    extraction_config: Dict[str, Any] = Field(default_factory=dict)


class ExtractionRunResponse(BaseModel):
    id: uuid.UUID
    document_id: uuid.UUID
    model_id: str
    status: str
    metrics: Dict[str, Any]
    created_at: str

    model_config = ConfigDict(from_attributes=True)


# Traversal & Path Schemas
class GraphTraversalRequest(BaseModel):
    start_entity_id: uuid.UUID
    max_depth: int = Field(default=2, ge=1, le=4)
    allowed_relations: Optional[List[str]] = None
    max_nodes: int = Field(default=50, ge=1, le=200)


class GraphTraversalResponse(BaseModel):
    start_entity_id: uuid.UUID
    nodes: List[EntityResponse]
    edges: List[RelationResponse]
    depth_reached: int
    total_nodes: int
    total_edges: int


class GraphPathsRequest(BaseModel):
    source_entity_id: uuid.UUID
    target_entity_id: uuid.UUID
    max_depth: int = Field(default=3, ge=1, le=4)


class GraphPathItem(BaseModel):
    length: int
    nodes: List[EntityResponse]
    relations: List[RelationResponse]


class GraphPathsResponse(BaseModel):
    source_entity_id: uuid.UUID
    target_entity_id: uuid.UUID
    paths: List[GraphPathItem]
    found: bool


# Query & Grounded Generation Schemas
class GraphQueryRequest(BaseModel):
    question: str = Field(..., min_length=2, max_length=2000)
    strategy: GraphRetrievalStrategy = GraphRetrievalStrategy.HYBRID_GRAPH_VECTOR
    max_hops: int = Field(default=2, ge=1, le=4)
    top_k: int = Field(default=5, ge=1, le=20)
    provider_id: str = "mock"
    model_id: str = "mock-instruct-v1"
    temperature: float = 0.0


class GraphQueryResponse(BaseModel):
    question: str
    strategy: GraphRetrievalStrategy
    final_answer: str
    citations: List[Dict[str, Any]] = Field(default_factory=list)
    retrieved_entities: List[EntityResponse] = Field(default_factory=list)
    retrieved_relations: List[RelationResponse] = Field(default_factory=list)
    retrieved_evidence: List[EvidenceResponse] = Field(default_factory=list)
    traversal_paths: List[List[str]] = Field(default_factory=list)
    latency_ms: float
    total_tokens: Optional[int] = None


# Strategy Comparison Schemas
class GraphCompareRequest(BaseModel):
    question: str = Field(..., min_length=2, max_length=2000)
    strategies: Optional[List[GraphRetrievalStrategy]] = None
    provider_id: str = "mock"
    model_id: str = "mock-instruct-v1"
    top_k: int = 5


class GraphCompareItem(BaseModel):
    strategy: str
    latency_ms: float
    entities_count: int
    relations_count: int
    evidence_count: int
    citation_count: int
    answer_snippet: str
    success: bool
    error: Optional[str] = None


class GraphCompareResponse(BaseModel):
    question: str
    results: Dict[str, GraphCompareItem]


# Graph Stats & Analytics
class GraphStatsResponse(BaseModel):
    node_count: int
    edge_count: int
    evidence_count: int
    entity_types_count: Dict[str, int]
    relation_types_count: Dict[str, int]
    density: float
    avg_degree: float
    connected_components: int


# Community Schemas
class CommunitySummary(BaseModel):
    community_id: str
    name: str
    member_entities: List[str]
    key_relations: List[str]
    summary_text: str
    size: int


# Entity Resolution Schemas
class EntityMergeCandidate(BaseModel):
    source_entity: EntityResponse
    target_entity: EntityResponse
    similarity_score: float
    match_reason: str
    requires_human_review: bool


class EntityMergeRequest(BaseModel):
    surviving_entity_id: uuid.UUID
    merged_entity_id: uuid.UUID
