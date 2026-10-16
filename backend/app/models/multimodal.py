import uuid
from datetime import datetime, timezone
from typing import List, Optional
from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    String,
    Integer,
    Float,
    Boolean,
    Text,
    DateTime,
    ForeignKey,
    Index,
    Uuid,
    JSON,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.db.base import Base

# Portable JSON type: JSONB on PostgreSQL, JSON on SQLite
PortableJSON = JSON().with_variant(JSONB, "postgresql")


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class DocumentPageAsset(Base):
    __tablename__ = "document_page_assets"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    page_number: Mapped[int] = mapped_column(Integer, nullable=False)
    width: Mapped[int] = mapped_column(Integer, default=595, nullable=False)
    height: Mapped[int] = mapped_column(Integer, default=842, nullable=False)
    has_native_text: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    native_text_char_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    is_scanned: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    image_path: Mapped[str] = mapped_column(String(500), nullable=False)
    ocr_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    ocr_confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    ocr_language: Mapped[str] = mapped_column(String(20), default="eng", nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )

    # Relationships
    document = relationship("Document")
    elements: Mapped[List["DocumentElement"]] = relationship(
        "DocumentElement",
        back_populates="page_asset",
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        Index("ix_doc_page_assets_doc_page", "document_id", "page_number"),
    )


class DocumentElement(Base):
    __tablename__ = "document_elements"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    page_asset_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("document_page_assets.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    page_number: Mapped[int] = mapped_column(Integer, nullable=False)
    element_type: Mapped[str] = mapped_column(
        String(50), nullable=False
    )  # title, paragraph, table, figure, header, footer, caption
    reading_order: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    bounding_box: Mapped[dict] = mapped_column(
        PortableJSON, default=dict, nullable=False
    )  # {x0, y0, x1, y1, width, height}
    content_text: Mapped[str] = mapped_column(Text, default="", nullable=False)
    structured_data: Mapped[dict] = mapped_column(
        PortableJSON, default=dict, nullable=False
    )  # for tables: {headers: [], rows: [[]], num_rows, num_cols}
    asset_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    confidence: Mapped[float] = mapped_column(Float, default=1.0, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )

    # Relationships
    document = relationship("Document")
    page_asset = relationship("DocumentPageAsset", back_populates="elements")

    __table_args__ = (
        Index("ix_doc_elements_doc_type", "document_id", "element_type"),
        Index("ix_doc_elements_doc_page_order", "document_id", "page_number", "reading_order"),
    )


class MultimodalEmbedding(Base):
    __tablename__ = "multimodal_embeddings"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    element_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("document_elements.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    page_asset_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("document_page_assets.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    model_id: Mapped[str] = mapped_column(String(100), default="clip-ViT-B-32", nullable=False)
    embedding_type: Mapped[str] = mapped_column(
        String(50), default="image", nullable=False
    )  # image, text_prompt, page
    vector_dimension: Mapped[int] = mapped_column(Integer, default=512, nullable=False)
    vector = mapped_column(Vector(), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )

    # Relationships
    document = relationship("Document")
    element = relationship("DocumentElement")
    page_asset = relationship("DocumentPageAsset")

    __table_args__ = (
        Index("ix_mm_embeddings_doc_type", "document_id", "embedding_type"),
        Index("ix_mm_embeddings_model", "model_id"),
    )
