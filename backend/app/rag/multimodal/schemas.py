import uuid
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    x0: float = Field(..., description="Left coordinate in points or normalized")
    y0: float = Field(..., description="Top coordinate in points or normalized")
    x1: float = Field(..., description="Right coordinate in points or normalized")
    y1: float = Field(..., description="Bottom coordinate in points or normalized")
    width: float = Field(..., description="Width")
    height: float = Field(..., description="Height")


class DocumentElementSchema(BaseModel):
    id: uuid.UUID
    document_id: uuid.UUID
    page_number: int
    element_type: str  # title, paragraph, table, figure, header, footer, caption
    reading_order: int
    bounding_box: BoundingBox
    content_text: str = ""
    structured_data: Dict[str, Any] = Field(default_factory=dict)
    asset_path: Optional[str] = None
    confidence: float = 1.0


class DocumentPageAssetSchema(BaseModel):
    id: uuid.UUID
    document_id: uuid.UUID
    page_number: int
    width: int
    height: int
    has_native_text: bool
    native_text_char_count: int
    is_scanned: bool
    image_path: str
    ocr_text: Optional[str] = None
    ocr_confidence: Optional[float] = None
    ocr_language: str = "eng"


class TableDataSchema(BaseModel):
    table_id: str
    page_number: int
    caption: Optional[str] = None
    num_rows: int
    num_cols: int
    headers: List[str]
    rows: List[List[str]]
    markdown_table: str
    raw_flattened_text: str
    bounding_box: Optional[BoundingBox] = None


class AnalyzeDocumentRequest(BaseModel):
    document_id: Optional[uuid.UUID] = None
    dpi: int = Field(150, ge=72, le=300)
    force_ocr: bool = False
    ocr_language: str = "eng"


class AnalyzeDocumentResponse(BaseModel):
    document_id: uuid.UUID
    total_pages: int
    scanned_pages_count: int
    total_elements: int
    elements_by_type: Dict[str, int]
    processing_time_ms: float
    message: str


class OCRPreviewRequest(BaseModel):
    document_id: uuid.UUID
    page_number: int = Field(1, ge=1)
    language: str = Field("eng", pattern="^(eng|fra|fra\\+eng|eng\\+fra)$")
    run_engine: str = Field("tesseract", description="tesseract or fallback")


class OCRPreviewResponse(BaseModel):
    document_id: uuid.UUID
    page_number: int
    language: str
    is_available: bool
    native_text: str
    ocr_text: str
    cer: float = Field(..., description="Character Error Rate between native and OCR (0.0 to 1.0+)")
    wer: float = Field(..., description="Word Error Rate between native and OCR (0.0 to 1.0+)")
    ocr_confidence: float = Field(..., description="Average word/block confidence percentage 0-100")
    words_detected: int
    execution_time_ms: float
    warning: Optional[str] = None


class TableExtractRequest(BaseModel):
    document_id: uuid.UUID
    page_number: Optional[int] = None


class TableExtractResponse(BaseModel):
    document_id: uuid.UUID
    tables: List[TableDataSchema]
    count: int


class MultimodalModelInfo(BaseModel):
    model_id: str
    modality: str  # text_and_image, image_only, vision_language
    dimension: int
    description: str
    is_loaded: bool
    provider: str


class MultimodalSearchRequest(BaseModel):
    document_id: Optional[uuid.UUID] = None
    query: str = Field(..., min_length=1)
    strategy: str = Field(
        "hybrid",
        description="ocr_text, image_to_image, text_to_image, page_level, hybrid",
    )
    page_number: Optional[int] = None
    element_ids: Optional[List[uuid.UUID]] = None
    top_k: int = Field(5, ge=1, le=20)
    ocr_weight: float = Field(0.5, ge=0.0, le=1.0)
    visual_weight: float = Field(0.5, ge=0.0, le=1.0)


class MultimodalSearchResultItem(BaseModel):
    id: str
    document_id: str
    page_number: int
    element_type: str
    score: float
    content_text: str
    structured_data: Dict[str, Any] = Field(default_factory=dict)
    image_path: Optional[str] = None
    bounding_box: Optional[Dict[str, float]] = None
    citation_tag: str
    strategy: str


class MultimodalSearchResponse(BaseModel):
    query: str
    strategy: str
    top_k: int
    results: List[MultimodalSearchResultItem]
    total_found: int
    latency_ms: float


class VisionQARequest(BaseModel):
    document_id: uuid.UUID
    question: str = Field(..., min_length=2)
    page_number: Optional[int] = None
    element_ids: Optional[List[uuid.UUID]] = None
    provider: str = Field("mock", description="mock, gemini, or ollama")


class VisionQAResponse(BaseModel):
    question: str
    answer: str
    citations: List[Union[str, Dict[str, Any]]]
    evidence_items: List[Dict[str, Any]]
    model_used: str
    is_vision_enabled: bool
    warnings: List[str]
    latency_ms: float


class MultimodalBenchmarkScenarioResult(BaseModel):
    scenario_id: str
    name: str
    category: str
    passed: bool
    score: float
    retrieval_recall: float
    citation_valid: bool
    answer_grounding: float
    latency_ms: float
    notes: str


class MultimodalBenchmarkReport(BaseModel):
    total_scenarios: int
    passed_count: int
    pass_rate: float
    avg_latency_ms: float
    avg_cer: float
    avg_table_accuracy: float
    results: List[MultimodalBenchmarkScenarioResult]
