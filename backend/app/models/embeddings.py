import uuid
from datetime import datetime, timezone
from typing import List, Optional
from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    String,
    Integer,
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


class EmbeddingExperiment(Base):
    __tablename__ = "embedding_experiments"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    name: Mapped[Optional[str]] = mapped_column(String(150), nullable=True)
    document_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    chunking_experiment_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("chunking_experiments.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    model_id: Mapped[str] = mapped_column(String(150), nullable=False, index=True)
    provider: Mapped[str] = mapped_column(String(50), nullable=False)
    vector_dimension: Mapped[int] = mapped_column(Integer, nullable=False)
    is_normalized: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    parameters: Mapped[dict] = mapped_column(PortableJSON, default=dict, nullable=False)
    metrics: Mapped[dict] = mapped_column(PortableJSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )

    # Relationships
    document = relationship("Document")
    chunking_experiment = relationship("ChunkingExperiment")
    embeddings: Mapped[List["ChunkEmbedding"]] = relationship(
        "ChunkEmbedding",
        back_populates="experiment",
        cascade="all, delete-orphan",
        order_by="ChunkEmbedding.sequence_index",
    )

    __table_args__ = (
        Index("ix_embedding_experiments_model_created", "model_id", "created_at"),
    )


class ChunkEmbedding(Base):
    __tablename__ = "chunk_embeddings"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    experiment_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("embedding_experiments.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    chunk_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("document_chunks.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    sequence_index: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    text_snippet: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    vector_dimension: Mapped[int] = mapped_column(Integer, default=384, nullable=False)

    # Vector column without fixed dimension constraint allows multi-model embeddings (384, 768, etc.)
    # In SQLite unit tests, Vector creates successfully as a generic type.
    vector = mapped_column(Vector(), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )

    # Relationships
    experiment = relationship("EmbeddingExperiment", back_populates="embeddings")
    chunk = relationship("DocumentChunk")

    __table_args__ = (
        Index("ix_chunk_embeddings_exp_seq", "experiment_id", "sequence_index"),
    )
