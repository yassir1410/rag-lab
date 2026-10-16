import uuid
from datetime import datetime, timezone
from typing import Optional
from sqlalchemy import (
    String,
    Integer,
    BigInteger,
    Float,
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


class RetrievalIndex(Base):
    """
    Tracks dynamic pgvector indexes (HNSW, IVFFlat) created in PostgreSQL.
    """
    __tablename__ = "retrieval_indexes"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    index_name: Mapped[str] = mapped_column(
        String(100), unique=True, nullable=False, index=True
    )
    index_type: Mapped[str] = mapped_column(
        String(20), nullable=False
    )  # 'hnsw' | 'ivfflat'
    distance_metric: Mapped[str] = mapped_column(
        String(20), nullable=False
    )  # 'cosine' | 'l2' | 'inner_product'
    opclass: Mapped[str] = mapped_column(
        String(30), nullable=False
    )  # 'vector_cosine_ops' | 'vector_l2_ops' | 'vector_ip_ops'
    vector_dimension: Mapped[int] = mapped_column(Integer, nullable=False)
    model_id: Mapped[str] = mapped_column(String(150), nullable=False, index=True)
    embedding_experiment_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("embedding_experiments.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    parameters: Mapped[dict] = mapped_column(PortableJSON, default=dict, nullable=False)
    status: Mapped[str] = mapped_column(
        String(20), default="ready", nullable=False
    )  # 'building' | 'ready' | 'failed'
    build_duration_ms: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    index_size_bytes: Mapped[int] = mapped_column(BigInteger, default=0, nullable=False)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )

    # Relationships
    embedding_experiment = relationship("EmbeddingExperiment")

    __table_args__ = (
        Index("ix_retrieval_indexes_type_metric", "index_type", "distance_metric"),
    )


class RetrievalExperiment(Base):
    """
    Stores historical retrieval test queries and comparative benchmarks.
    """
    __tablename__ = "retrieval_experiments"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(String(150), nullable=False)
    query_text: Mapped[str] = mapped_column(Text, nullable=False)
    model_id: Mapped[str] = mapped_column(String(150), nullable=False)
    distance_metric: Mapped[str] = mapped_column(String(20), nullable=False)
    top_k: Mapped[int] = mapped_column(Integer, default=5, nullable=False)
    filters: Mapped[dict] = mapped_column(PortableJSON, default=dict, nullable=False)
    results: Mapped[list] = mapped_column(PortableJSON, default=list, nullable=False)
    metrics: Mapped[dict] = mapped_column(PortableJSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )

    __table_args__ = (
        Index("ix_retrieval_experiments_created", "created_at"),
    )
