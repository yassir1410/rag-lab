import os
import uuid
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
from sqlalchemy import select, func, desc
from sqlalchemy.orm import Session
from app.core.config import settings
from app.core.logging import logger
from app.models.document import Document, DocumentSection
from app.rag.ingestion.pipeline import IngestionPipeline, IngestionResult
from app.rag.ingestion.exceptions import DuplicateDocumentError, IngestionError


class DocumentService:
    def __init__(self, pipeline: Optional[IngestionPipeline] = None):
        self.pipeline = pipeline or IngestionPipeline()
        self.storage_dir = settings.ABSOLUTE_STORAGE_DIR

    def ingest_document(
        self,
        db: Session,
        file_bytes: bytes,
        original_filename: str,
        reported_mime: str = "",
    ) -> Document:
        """
        Validate, parse, normalize, store, and persist an uploaded document.
        Rolls back database and deletes saved disk file on any error.
        """
        # 1. Quick duplicate pre-check using hash before full parse
        content_hash = self.pipeline.validator.compute_sha256(file_bytes)
        existing = db.execute(
            select(Document).where(Document.sha256_hash == content_hash)
        ).scalar_one_or_none()

        if existing:
            logger.warning(
                f"Duplicate upload rejected: SHA-256 hash match with document id={existing.id}"
            )
            raise DuplicateDocumentError(
                message=f"A document with identical content already exists: '{existing.original_filename}'",
                existing_document_id=str(existing.id),
                existing_filename=existing.original_filename,
            )

        # 2. Run ingestion pipeline (validation, parsing, normalization)
        result: IngestionResult = self.pipeline.process(
            file_bytes=file_bytes,
            filename=original_filename,
            reported_mime=reported_mime,
        )

        saved_file_path: Optional[Path] = None

        try:
            # 3. Store file to disk safely under UUID storage name
            saved_file_path = self.storage_dir / result.storage_filename
            with open(saved_file_path, "wb") as f:
                f.write(file_bytes)

            # 4. Prepare Document ORM entity
            doc_id = uuid.uuid4()
            document = Document(
                id=doc_id,
                original_filename=result.sanitized_filename,
                mime_type=result.mime_type,
                file_size=result.file_size,
                sha256_hash=result.sha256_hash,
                storage_path=result.storage_filename,  # relative filename, not exposing absolute path
                status="processed",
                detected_language=result.parsed_document.detected_language,
                page_count=result.parsed_document.page_count,
                extraction_method=result.parsed_document.extraction_method,
                error_message=None,
                warnings=result.parsed_document.warnings,
            )
            db.add(document)

            # 5. Prepare DocumentSection ORM entities
            for section in result.parsed_document.sections:
                sec_entity = DocumentSection(
                    id=uuid.uuid4(),
                    document_id=doc_id,
                    section_index=section.section_index,
                    page_number=section.page_number,
                    heading=section.heading,
                    content=section.content,
                    normalized_content=section.metadata.get("normalized_content"),
                    content_type=section.content_type,
                    metadata_json=section.metadata,
                )
                db.add(sec_entity)

            db.commit()
            db.refresh(document)
            logger.info(
                f"Successfully ingested document '{result.sanitized_filename}' "
                f"({len(result.parsed_document.sections)} sections, duration {result.processing_duration_ms}ms)"
            )
            return document

        except Exception as e:
            db.rollback()
            # Clean up stored file if written
            if saved_file_path and saved_file_path.exists():
                try:
                    saved_file_path.unlink()
                except OSError:
                    pass
            logger.error(f"Failed to ingest document '{original_filename}': {str(e)}")
            raise

    def list_documents(
        self, db: Session, limit: int = 50, offset: int = 0
    ) -> Tuple[int, List[Document]]:
        """List documents with pagination and section counts."""
        total = db.scalar(select(func.count(Document.id))) or 0
        docs = (
            db.execute(
                select(Document)
                .order_by(desc(Document.created_at))
                .limit(limit)
                .offset(offset)
            )
            .scalars()
            .all()
        )
        return total, list(docs)

    def get_document(self, db: Session, document_id: uuid.UUID) -> Optional[Document]:
        """Fetch document by primary key."""
        return db.get(Document, document_id)

    def get_sections(
        self, db: Session, document_id: uuid.UUID
    ) -> List[DocumentSection]:
        """Fetch ordered parsed sections for a document."""
        return list(
            db.execute(
                select(DocumentSection)
                .where(DocumentSection.document_id == document_id)
                .order_by(DocumentSection.section_index)
            )
            .scalars()
            .all()
        )

    def get_document_content(
        self, db: Session, document_id: uuid.UUID
    ) -> Optional[Dict[str, Any]]:
        """Retrieve full raw text, normalized text, comparison metrics, and sections."""
        document = self.get_document(db, document_id)
        if not document:
            return None

        sections = self.get_sections(db, document_id)
        raw_text_parts = [s.content for s in sections]
        norm_text_parts = [s.normalized_content or s.content for s in sections]

        raw_full = "\n\n".join(raw_text_parts)
        norm_full = "\n\n".join(norm_text_parts)

        normalizer = self.pipeline.normalizer
        _, stats = normalizer.compare(raw_full)

        return {
            "id": document.id,
            "original_filename": document.original_filename,
            "raw_text": raw_full,
            "normalized_text": norm_full,
            "comparison_stats": stats,
            "extraction_method": document.extraction_method,
            "warnings": document.warnings,
            "page_count": document.page_count,
            "sections_count": len(sections),
            "processing_duration_ms": None,
        }

    def delete_document(self, db: Session, document_id: uuid.UUID) -> bool:
        """Safely delete document database record and disk file."""
        doc = self.get_document(db, document_id)
        if not doc:
            return False

        # Physical file removal
        file_path = self.storage_dir / doc.storage_path
        if file_path.exists():
            try:
                file_path.unlink()
            except OSError as e:
                logger.warning(f"Could not delete physical file '{file_path}': {e}")

        db.delete(doc)
        db.commit()
        logger.info(f"Deleted document id={document_id} and associated sections")
        return True
