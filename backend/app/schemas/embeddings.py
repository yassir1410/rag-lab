from datetime import datetime
from typing import List, Optional, Dict, Any
from uuid import UUID
from pydantic import BaseModel, Field, ConfigDict


class ModelInfoResponse(BaseModel):
    model_id: str
    display_name: str
    provider: str
    dimension: int
    max_seq_length: int
    language: str
    description: str
    query_prefix: str = ""
    passage_prefix: str = ""
    approx_size_mb: float = 0.0
    normalized_by_default: bool = True
    is_loaded: bool = False


class TokenizeRequest(BaseModel):
    text: str
    model_id: str = "sentence-transformers/all-MiniLM-L6-v2"


class TokenizeResponse(BaseModel):
    model_id: str
    text: str
    char_count: int
    word_count: int
    token_count: int
    max_seq_length: int
    tokens: List[str]
    token_ids: List[int]
    is_truncated: bool
    truncated_tokens_count: int
    warnings: List[str]
    ratio_char_to_token: float
    ratio_word_to_token: float


class TokenizeCompareRequest(BaseModel):
    text: str
    model_ids: List[str] = Field(
        default=[
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        ]
    )


class GenerateEmbeddingsRequest(BaseModel):
    texts: Optional[List[str]] = None
    document_id: Optional[UUID] = None
    chunking_experiment_id: Optional[UUID] = None
    chunk_ids: Optional[List[UUID]] = None
    model_id: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 32
    normalize: bool = True
    is_query: bool = False


class EmbeddingItemResponse(BaseModel):
    index: int
    text: str
    token_count: int
    dimension: int
    norm: float
    vector_preview: List[float]  # first 8 dimensions for quick UI preview
    full_vector: Optional[List[float]] = None  # full vector when explicitly requested


class GenerateEmbeddingsResponse(BaseModel):
    model_id: str
    dimension: int
    is_normalized: bool
    total_items: int
    total_tokens: int
    latency_ms: float
    items: List[EmbeddingItemResponse]


class SimilarityCalculateRequest(BaseModel):
    text_a: str
    text_b: str
    model_id: str = "sentence-transformers/all-MiniLM-L6-v2"
    normalize: bool = True


class SimilarityCalculateResponse(BaseModel):
    text_a: str
    text_b: str
    model_id: str
    cosine_similarity: float
    dot_product: float
    euclidean_distance: float
    vector_a_norm: float
    vector_b_norm: float
    explanation: str


class SimilarityMatrixRequest(BaseModel):
    texts: List[str]
    model_id: str = "sentence-transformers/all-MiniLM-L6-v2"
    metric: str = "cosine"
    normalize: bool = True


class SimilarityMatrixResponse(BaseModel):
    texts: List[str]
    model_id: str
    metric: str
    matrix: List[List[float]]
    is_normalized: bool


class ProjectEmbeddingsRequest(BaseModel):
    texts: Optional[List[str]] = None
    chunking_experiment_id: Optional[UUID] = None
    embedding_experiment_id: Optional[UUID] = None
    model_id: str = "sentence-transformers/all-MiniLM-L6-v2"
    algorithm: str = "pca"  # "pca" | "tsne" | "umap"
    n_components: int = 2
    perplexity: float = 30.0
    n_neighbors: int = 15
    min_dist: float = 0.1


class ProjectedPoint(BaseModel):
    index: int
    coordinates: List[float]
    label: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ProjectEmbeddingsResponse(BaseModel):
    algorithm: str
    n_components: int
    model_id: str
    points: List[ProjectedPoint]
    explained_variance_ratio: Optional[List[float]] = None
    warnings: List[str] = Field(default_factory=list)


class BenchmarkRunRequest(BaseModel):
    model_id: str = "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingExperimentCreate(BaseModel):
    name: Optional[str] = None
    document_id: Optional[UUID] = None
    chunking_experiment_id: Optional[UUID] = None
    model_id: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 32
    normalize: bool = True


class EmbeddingExperimentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    name: Optional[str] = None
    document_id: Optional[UUID] = None
    chunking_experiment_id: Optional[UUID] = None
    model_id: str
    provider: str
    vector_dimension: int
    is_normalized: bool
    parameters: Dict[str, Any] = Field(default_factory=dict)
    metrics: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    embeddings_count: int = 0


class EmbeddingExperimentListResponse(BaseModel):
    total: int
    items: List[EmbeddingExperimentResponse]
    limit: int
    offset: int
