from datetime import datetime
from typing import List, Optional, Dict, Any
from uuid import UUID
from pydantic import BaseModel, Field, ConfigDict


class ChunkMetadataResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    start_char: Optional[int] = None
    end_char: Optional[int] = None
    page_numbers: List[int] = Field(default_factory=list)
    headings: List[str] = Field(default_factory=list)
    source_section_ids: List[str] = Field(default_factory=list)
    content_type: str = "text"
    parent_chunk_id: Optional[str] = None
    child_chunk_ids: List[str] = Field(default_factory=list)
    breakpoint_distance: Optional[float] = None
    custom: Dict[str, Any] = Field(default_factory=dict)


class ChunkResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    document_id: UUID
    sequence_index: int
    content: str
    char_count: int
    token_count: int
    parent_chunk_id: Optional[UUID] = None
    metadata: Dict[str, Any] = Field(default_factory=dict, alias="metadata_json")
    created_at: datetime


class StrategyInfoResponse(BaseModel):
    strategy_name: str
    display_name: str
    description: str
    default_config: Dict[str, Any]


class ChunkingPreviewRequest(BaseModel):
    document_id: UUID
    strategy: str
    config: Dict[str, Any] = Field(default_factory=dict)


class ChunkingPreviewResponse(BaseModel):
    strategy: str
    chunks: List[ChunkResponse]
    metrics: Dict[str, Any]
    config: Dict[str, Any]
    processing_duration_ms: float


class ChunkingExperimentCreate(BaseModel):
    document_id: UUID
    strategy: str
    config: Dict[str, Any] = Field(default_factory=dict)


class ChunkingExperimentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    document_id: UUID
    strategy: str
    config: Dict[str, Any]
    metrics: Dict[str, Any]
    created_at: datetime
    chunks_count: int = 0
    chunks: Optional[List[ChunkResponse]] = None


class ChunkingExperimentListResponse(BaseModel):
    total: int
    items: List[ChunkingExperimentResponse]
    limit: int
    offset: int


class StrategyConfigItem(BaseModel):
    strategy: str
    config: Dict[str, Any] = Field(default_factory=dict)


class ChunkingCompareRequest(BaseModel):
    document_id: UUID
    strategies: List[StrategyConfigItem]


class ChunkingCompareResponse(BaseModel):
    document_id: UUID
    results: List[ChunkingPreviewResponse]
