import uuid
from typing import List, Optional
from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    UploadFile,
    File,
    Query,
    status,
    Response,
)
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.logging import logger
from app.db.session import get_db
from app.models.document import Document
from app.rag.ingestion.exceptions import (
    DuplicateDocumentError,
    FileTooLargeError,
    FileValidationError,
    UnsupportedFileTypeError,
    CorruptedFileError,
    EncryptedFileError,
)
from app.schemas.document import (
    DocumentResponse,
    DocumentSectionResponse,
    DocumentListResponse,
    DocumentContentResponse,
    DuplicateErrorResponse,
)
from app.services.document_service import DocumentService

router = APIRouter()
doc_service = DocumentService()


def _to_document_response(doc: Document) -> DocumentResponse:
    sections_count = len(doc.sections) if doc.sections else 0
    return DocumentResponse(
        id=doc.id,
        original_filename=doc.original_filename,
        mime_type=doc.mime_type,
        file_size=doc.file_size,
        sha256_hash=doc.sha256_hash,
        status=doc.status,
        detected_language=doc.detected_language,
        page_count=doc.page_count,
        extraction_method=doc.extraction_method,
        warnings=doc.warnings if isinstance(doc.warnings, list) else [],
        sections_count=sections_count,
        error_message=doc.error_message,
        created_at=doc.created_at,
        updated_at=doc.updated_at,
    )


@router.post(
    "",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload & Parse Document",
    description="Upload a document (PDF, DOCX, TXT, MD), validate, parse sections, normalize text, and store.",
    responses={
        409: {"model": DuplicateErrorResponse, "description": "Duplicate document detected"},
        413: {"description": "File exceeds maximum size limit"},
        400: {"description": "Invalid, corrupted, or unsupported file format"},
    },
)
async def upload_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> DocumentResponse:
    filename = file.filename or "uploaded_file"
    reported_mime = file.content_type or ""

    # Stream read with size limit enforcement
    max_bytes = settings.MAX_UPLOAD_SIZE_BYTES
    chunks = []
    total_bytes = 0

    try:
        try:
            while True:
                chunk = await file.read(1024 * 64)  # 64 KB chunks
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > max_bytes:
                    raise FileTooLargeError(
                        f"File exceeds maximum allowed limit of {settings.MAX_UPLOAD_SIZE_MB} MB"
                    )
                chunks.append(chunk)

            file_bytes = b"".join(chunks)
            if not file_bytes:
                raise FileValidationError("Uploaded file is empty (0 bytes)")

        except FileTooLargeError as e:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=str(e),
            )
        except FileValidationError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e),
            )
        finally:
            await file.close()

        doc = doc_service.ingest_document(
            db=db,
            file_bytes=file_bytes,
            original_filename=filename,
            reported_mime=reported_mime,
        )
        return _to_document_response(doc)

    except DuplicateDocumentError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "message": str(e),
                "existing_document_id": e.existing_document_id,
                "existing_filename": e.existing_filename,
            },
        )
    except UnsupportedFileTypeError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported format: {str(e)}",
        )
    except (FileValidationError, CorruptedFileError, EncryptedFileError) as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Unexpected error processing document upload")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process document: {str(e)}",
        )


@router.get(
    "",
    response_model=DocumentListResponse,
    summary="List Documents",
    description="Retrieve a paginated list of ingested documents.",
)
def list_documents(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
) -> DocumentListResponse:
    total, docs = doc_service.list_documents(db=db, limit=limit, offset=offset)
    items = [_to_document_response(doc) for doc in docs]
    return DocumentListResponse(total=total, items=items, limit=limit, offset=offset)


@router.get(
    "/{document_id}",
    response_model=DocumentResponse,
    summary="Get Document Details",
)
def get_document(
    document_id: uuid.UUID,
    db: Session = Depends(get_db),
) -> DocumentResponse:
    doc = doc_service.get_document(db=db, document_id=document_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with id {document_id} not found",
        )
    return _to_document_response(doc)


@router.get(
    "/{document_id}/sections",
    response_model=List[DocumentSectionResponse],
    summary="Get Document Sections",
)
def get_sections(
    document_id: uuid.UUID,
    db: Session = Depends(get_db),
) -> List[DocumentSectionResponse]:
    doc = doc_service.get_document(db=db, document_id=document_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with id {document_id} not found",
        )
    sections = doc_service.get_sections(db=db, document_id=document_id)
    return [
        DocumentSectionResponse(
            id=s.id,
            document_id=s.document_id,
            section_index=s.section_index,
            page_number=s.page_number,
            heading=s.heading,
            content=s.content,
            normalized_content=s.normalized_content,
            content_type=s.content_type,
            metadata_json=s.metadata_json or {},
            created_at=s.created_at,
        )
        for s in sections
    ]


@router.get(
    "/{document_id}/content",
    response_model=DocumentContentResponse,
    summary="Get Raw & Normalized Content Comparison",
)
def get_document_content(
    document_id: uuid.UUID,
    db: Session = Depends(get_db),
) -> DocumentContentResponse:
    content_data = doc_service.get_document_content(db=db, document_id=document_id)
    if not content_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with id {document_id} not found",
        )
    return DocumentContentResponse(
        id=content_data["id"],
        original_filename=content_data["original_filename"],
        raw_text=content_data["raw_text"],
        normalized_text=content_data["normalized_text"],
        comparison_stats=content_data["comparison_stats"],
        extraction_method=content_data["extraction_method"],
        warnings=content_data["warnings"] or [],
        page_count=content_data["page_count"],
        sections_count=content_data["sections_count"],
        processing_duration_ms=content_data["processing_duration_ms"],
    )


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete Document",
)
def delete_document(
    document_id: uuid.UUID,
    db: Session = Depends(get_db),
):
    deleted = doc_service.delete_document(db=db, document_id=document_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with id {document_id} not found",
        )
    return Response(status_code=status.HTTP_204_NO_CONTENT)
