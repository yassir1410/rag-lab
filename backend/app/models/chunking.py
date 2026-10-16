import uuid
from datetime import datetime, timezone
from typing import List, Optional
from sqlalchemy import (
    String,
    Integer,
    Text,
    DateTime,
    ForeignKey,
    Index,
    Uuid,
    JSON,
)
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.db.base import Base

# Portable JSON type: JSONB on PostgreSQL, JSON on SQLite
PortableJSON = JSON().with_variant(JSONB, "postgresql")
# Portable TSVector type: TSVECTOR on PostgreSQL, Text on SQLite
PortableTSVector = Text().with_variant(TSVECTOR, "postgresql")


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ChunkingExperiment(Base):
    __tablename__ = "chunking_experiments"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    strategy: Mapped[str] = mapped_column(String(50), nullable=False)
    config: Mapped[dict] = mapped_column(PortableJSON, default=dict, nullable=False)
    metrics: Mapped[dict] = mapped_column(PortableJSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )

    # Relationships
    document = relationship("Document")
    chunks: Mapped[List["DocumentChunk"]] = relationship(
        "DocumentChunk",
        back_populates="experiment",
        cascade="all, delete-orphan",
        order_by="DocumentChunk.sequence_index",
    )

    __table_args__ = (
        Index("ix_chunking_experiments_doc_created", "document_id", "created_at"),
    )


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    experiment_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("chunking_experiments.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    sequence_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False)
    char_count: Mapped[int] = mapped_column(Integer, nullable=False)
    parent_chunk_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("document_chunks.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    metadata_json: Mapped[dict] = mapped_column(
        "metadata", PortableJSON, default=dict, nullable=False
    )
    tsv_content: Mapped[Optional[str]] = mapped_column(
        PortableTSVector, nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )

    # Relationships
    experiment = relationship("ChunkingExperiment", back_populates="chunks")
    document = relationship("Document")
    parent_chunk = relationship("DocumentChunk", remote_side=[id])

    __table_args__ = (
        Index("ix_document_chunks_exp_seq", "experiment_id", "sequence_index"),
        Index("ix_document_chunks_doc_seq", "document_id", "sequence_index"),
    )
