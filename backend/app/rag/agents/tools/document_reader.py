import time
import uuid
from typing import Optional, Type
from pydantic import BaseModel, Field, model_validator
from sqlalchemy.orm import Session

from app.rag.agents.tools.base import BaseAgentTool
from app.rag.agents.schemas import ToolResult
from app.models.document import Document, DocumentSection
from app.models.chunking import DocumentChunk


class ReadDocumentSectionArgs(BaseModel):
    document_id: Optional[str] = Field(default=None, description="UUID of the parent document")
    section_id: Optional[str] = Field(default=None, description="UUID of the specific section to read")
    chunk_id: Optional[str] = Field(default=None, description="UUID of a specific chunk to read")
    sequence_index: Optional[int] = Field(default=None, description="Sequence index of the chunk if chunk_id is unknown")

    @model_validator(mode="after")
    def validate_at_least_one_id(self):
        if not self.section_id and not self.chunk_id and not (self.document_id and self.sequence_index is not None):
            raise ValueError("Must provide either section_id, chunk_id, or (document_id + sequence_index)")
        return self


class ReadDocumentSectionTool(BaseAgentTool):
    """
    Approved Tool: Direct Document Section/Chunk Reader.
    Reads a specific text segment or adjacent passage from the database without executing a vector query.
    Useful for multi-hop verification or expanding context around a known evidence snippet.
    """

    @property
    def name(self) -> str:
        return "read_document_section"

    @property
    def description(self) -> str:
        return (
            "Reads the full content and metadata of a specific document section or chunk by ID. "
            "Use this when you have an evidence reference and need to inspect adjacent context or full section details."
        )

    @property
    def args_schema(self) -> Type[BaseModel]:
        return ReadDocumentSectionArgs

    def execute(self, session: Session, **kwargs) -> ToolResult:
        t0 = time.perf_counter()
        call_id = kwargs.pop("call_id", str(uuid.uuid4())[:8])

        try:
            validated = self.args_schema(**kwargs)
            results = []

            # 1. Look up by chunk_id
            if validated.chunk_id:
                try:
                    c_uuid = uuid.UUID(validated.chunk_id)
                except ValueError:
                    raise ValueError(f"Invalid UUID for chunk_id: '{validated.chunk_id}'")

                chunk = session.query(DocumentChunk).filter(DocumentChunk.id == c_uuid).first()
                if chunk:
                    doc = session.query(Document).filter(Document.id == chunk.document_id).first()
                    results.append({
                        "type": "chunk",
                        "chunk_id": str(chunk.id),
                        "document_id": str(chunk.document_id),
                        "document_name": doc.original_filename if doc else "Unknown",
                        "sequence_index": chunk.sequence_index,
                        "token_count": chunk.token_count,
                        "content": chunk.content,
                        "metadata": chunk.metadata_json or {},
                    })

            # 2. Look up by section_id
            elif validated.section_id:
                try:
                    s_uuid = uuid.UUID(validated.section_id)
                except ValueError:
                    raise ValueError(f"Invalid UUID for section_id: '{validated.section_id}'")

                sec = session.query(DocumentSection).filter(DocumentSection.id == s_uuid).first()
                if sec:
                    doc = session.query(Document).filter(Document.id == sec.document_id).first()
                    results.append({
                        "type": "section",
                        "section_id": str(sec.id),
                        "document_id": str(sec.document_id),
                        "document_name": doc.original_filename if doc else "Unknown",
                        "section_index": sec.section_index,
                        "page_number": sec.page_number,
                        "heading": sec.heading,
                        "content": sec.normalized_content or sec.content,
                        "metadata": sec.metadata_json or {},
                    })

            # 3. Look up by document_id and sequence_index
            elif validated.document_id and validated.sequence_index is not None:
                try:
                    d_uuid = uuid.UUID(validated.document_id)
                except ValueError:
                    raise ValueError(f"Invalid UUID for document_id: '{validated.document_id}'")

                chunk = (
                    session.query(DocumentChunk)
                    .filter(DocumentChunk.document_id == d_uuid, DocumentChunk.sequence_index == validated.sequence_index)
                    .first()
                )
                if chunk:
                    doc = session.query(Document).filter(Document.id == d_uuid).first()
                    results.append({
                        "type": "chunk",
                        "chunk_id": str(chunk.id),
                        "document_id": str(chunk.document_id),
                        "document_name": doc.original_filename if doc else "Unknown",
                        "sequence_index": chunk.sequence_index,
                        "token_count": chunk.token_count,
                        "content": chunk.content,
                        "metadata": chunk.metadata_json or {},
                    })

            lat = (time.perf_counter() - t0) * 1000.0
            if not results:
                return ToolResult(
                    call_id=call_id,
                    tool_name=self.name,
                    success=True,
                    data=[],
                    latency_ms=lat,
                    items_count=0,
                    raw_preview="No matching document section or chunk found for given identifier.",
                )

            preview = f"Read {results[0].get('type')} content ({len(results[0].get('content', ''))} chars)"
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
