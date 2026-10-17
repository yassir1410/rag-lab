import os
import uuid
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Query, Response
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.rag.multimodal.schemas import (
    AnalyzeDocumentRequest,
    AnalyzeDocumentResponse,
    OCRPreviewRequest,
    OCRPreviewResponse,
    TableExtractRequest,
    TableExtractResponse,
    MultimodalSearchRequest,
    MultimodalSearchResponse,
    VisionQARequest,
    VisionQAResponse,
    MultimodalModelInfo,
    MultimodalBenchmarkReport,
)
from app.rag.multimodal.service import MultimodalService
from app.rag.multimodal.generation.vision_provider import VisionProviderError
from app.rag.multimodal.embeddings.multimodal_registry import MultimodalModelRegistry

router = APIRouter()
multimodal_service = MultimodalService()


@router.post("/documents/{document_id}/analyze", response_model=AnalyzeDocumentResponse)
def analyze_document_endpoint(
    document_id: uuid.UUID,
    req: Optional[AnalyzeDocumentRequest] = None,
    db: Session = Depends(get_db),
):
    dpi = req.dpi if req else 150
    force_ocr = req.force_ocr if req else False
    ocr_lang = req.ocr_language if req else "eng"

    try:
        res = multimodal_service.analyze_document(
            db=db,
            document_id=document_id,
            dpi=dpi,
            force_ocr=force_ocr,
            ocr_language=ocr_lang,
        )
        return res
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Document analysis failed: {str(e)}")


@router.get("/documents/{document_id}/elements")
def get_document_elements_endpoint(
    document_id: uuid.UUID,
    element_type: Optional[str] = Query(None, description="title, paragraph, table, figure, header, footer, caption"),
    page_number: Optional[int] = Query(None, ge=1),
    db: Session = Depends(get_db),
):
    return multimodal_service.get_document_elements(
        db=db,
        document_id=document_id,
        element_type=element_type,
        page_number=page_number,
    )


@router.get("/documents/{document_id}/pages")
def get_document_pages_endpoint(
    document_id: uuid.UUID,
    db: Session = Depends(get_db),
):
    return multimodal_service.get_document_pages(db=db, document_id=document_id)


@router.get("/documents/{document_id}/pages/{page_num}/image")
def get_page_image_endpoint(
    document_id: uuid.UUID,
    page_num: int,
    db: Session = Depends(get_db),
):
    img_path = multimodal_service.get_page_image_path(db=db, document_id=document_id, page_number=page_num)
    if not img_path or not os.path.exists(img_path):
        raise HTTPException(status_code=404, detail=f"Image for document {document_id} page {page_num} not found.")

    # Prevent path traversal
    real_path = os.path.realpath(img_path)
    allowed_dir = os.path.realpath("/app/data/uploads/multimodal")
    if not real_path.startswith(allowed_dir):
        raise HTTPException(status_code=403, detail="Unauthorized asset access.")

    return FileResponse(real_path, media_type="image/png")


@router.post("/ocr/preview", response_model=OCRPreviewResponse)
def preview_ocr_endpoint(
    req: OCRPreviewRequest,
    db: Session = Depends(get_db),
):
    try:
        return multimodal_service.preview_ocr(
            db=db,
            document_id=req.document_id,
            page_number=req.page_number,
            language=req.language,
            run_engine=req.run_engine,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tables/extract", response_model=TableExtractResponse)
def extract_tables_endpoint(
    req: TableExtractRequest,
    db: Session = Depends(get_db),
):
    try:
        tables = multimodal_service.extract_tables(
            db=db,
            document_id=req.document_id,
            page_number=req.page_number,
        )
        return {
            "document_id": req.document_id,
            "tables": tables,
            "count": len(tables),
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models", response_model=List[MultimodalModelInfo])
def get_multimodal_models_endpoint():
    return MultimodalModelRegistry.list_models()


@router.post("/embeddings/generate")
def generate_embeddings_endpoint(
    text_prompt: Optional[str] = None,
):
    provider = MultimodalModelRegistry.get_provider()
    vec = provider.encode_text(text_prompt or "sample text query")
    return {
        "model_id": "clip-ViT-B-32",
        "dimension": len(vec),
        "vector_sample": vec[:10],
        "is_real_clip": provider.is_real_clip_loaded(),
    }


@router.post("/search", response_model=MultimodalSearchResponse)
def search_multimodal_endpoint(
    req: MultimodalSearchRequest,
    db: Session = Depends(get_db),
):
    return multimodal_service.search(
        db=db,
        query=req.query,
        strategy=req.strategy,
        document_id=req.document_id,
        page_number=req.page_number,
        element_ids=req.element_ids,
        top_k=req.top_k,
        ocr_weight=req.ocr_weight,
        visual_weight=req.visual_weight,
    )


@router.post("/query", response_model=VisionQAResponse)
def query_vision_endpoint(
    req: VisionQARequest,
    db: Session = Depends(get_db),
):
    try:
        return multimodal_service.query_vision(
            db=db,
            document_id=req.document_id,
            question=req.question,
            page_number=req.page_number,
            element_ids=req.element_ids,
            provider=req.provider,
        )
    except VisionProviderError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/compare")
def compare_strategies_endpoint(
    document_id: uuid.UUID,
    query: str = Query(..., min_length=2),
    db: Session = Depends(get_db),
):
    return multimodal_service.compare_strategies(
        db=db,
        document_id=document_id,
        query=query,
    )


@router.post("/benchmark", response_model=MultimodalBenchmarkReport)
def run_benchmark_endpoint():
    return multimodal_service.run_benchmark()
