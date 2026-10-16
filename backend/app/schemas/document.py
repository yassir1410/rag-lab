from datetime import datetime
from typing import List, Optional, Dict, Any
from uuid import UUID
from pydantic import BaseModel, Field, ConfigDict


class DocumentSectionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    id: UUID
    document_id: UUID
    section_index: int
    page_number: Optional[int] = None
    heading: Optional[str] = None
    content: str
    normalized_content: Optional[str] = None
    content_type: str
    metadata: Dict[str, Any] = Field(default_factory=dict, alias="metadata_json")
    created_at: datetime


class DocumentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    original_filename: str
    mime_type: str
    file_size: int
    sha256_hash: str
    status: str
    detected_language: Optional[str] = None
    page_count: Optional[int] = None
    extraction_method: str
    warnings: List[str] = Field(default_factory=list)
    sections_count: int = 0
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class DocumentListResponse(BaseModel):
    total: int
    items: List[DocumentResponse]
    limit: int
    offset: int


class DocumentContentResponse(BaseModel):
    id: UUID
    original_filename: str
    raw_text: str
    normalized_text: str
    comparison_stats: Dict[str, Any]
    extraction_method: str
    warnings: List[str] = Field(default_factory=list)
    page_count: Optional[int] = None
    sections_count: int
    processing_duration_ms: Optional[float] = None


class DuplicateErrorResponse(BaseModel):
    detail: str
    existing_document_id: str
    existing_filename: str
