import time
import uuid
from typing import Optional, Type
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.rag.agents.tools.base import BaseAgentTool
from app.rag.agents.schemas import ToolResult
from app.models.document import Document
from app.models.chunking import DocumentChunk


class InspectDocumentMetadataArgs(BaseModel):
    document_id: Optional[str] = Field(default=None, description="UUID of the document to inspect")
    filename: Optional[str] = Field(default=None, description="Filename or partial filename to search for")


class InspectDocumentMetadataTool(BaseAgentTool):
    """
    Approved Tool: Inspect Document Metadata.
    Returns technical and provenance metadata for one or all indexed documents
    (page count, file size, extraction method, chunk count, creation date).
    """

    @property
    def name(self) -> str:
        return "inspect_document_metadata"

    @property
    def description(self) -> str:
        return (
            "Inspects document metadata such as original filename, page count, file format, "
            "detected language, chunk count, and ingestion timestamps. If arguments are empty, lists all documents."
        )

    @property
    def args_schema(self) -> Type[BaseModel]:
        return InspectDocumentMetadataArgs

    def execute(self, session: Session, **kwargs) -> ToolResult:
        t0 = time.perf_counter()
        call_id = kwargs.pop("call_id", str(uuid.uuid4())[:8])

        try:
            validated = self.args_schema(**kwargs)
            query = session.query(Document)

            if validated.document_id:
                try:
                    d_uuid = uuid.UUID(validated.document_id)
                    query = query.filter(Document.id == d_uuid)
                except ValueError:
                    raise ValueError(f"Invalid UUID: '{validated.document_id}'")
            elif validated.filename:
                query = query.filter(Document.original_filename.ilike(f"%{validated.filename}%"))

            docs = query.order_by(Document.created_at.desc()).limit(10).all()

            results = []
            for doc in docs:
                chunk_count = (
                    session.query(func.count(DocumentChunk.id))
                    .filter(DocumentChunk.document_id == doc.id)
                    .scalar()
                    or 0
                )
                results.append({
                    "document_id": str(doc.id),
                    "filename": doc.original_filename,
                    "mime_type": doc.mime_type,
                    "file_size_bytes": doc.file_size,
                    "page_count": doc.page_count,
                    "detected_language": doc.detected_language or "unknown",
                    "extraction_method": doc.extraction_method,
                    "status": doc.status,
                    "chunk_count": chunk_count,
                    "created_at": doc.created_at.isoformat() if doc.created_at else None,
                })

            lat = (time.perf_counter() - t0) * 1000.0
            preview = f"Found metadata for {len(results)} document(s)"
            return ToolResult(
                call_id=call_id,
                tool_name=self.name,
                success=True,
                data=results,
                latency_ms=lat,
                items_count=len(results),
                raw_preview=preview,
            )

        except Exception as e:
            lat = (time.perf_counter() - t0) * 1000.0
            return ToolResult(
                call_id=call_id,
                tool_name=self.name,
                success=False,
                error=str(e),
                latency_ms=lat,
                items_count=0,
                raw_preview=f"Error: {e}",
            )
