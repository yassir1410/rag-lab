import uuid
from datetime import datetime, timezone
from typing import Optional, List
from sqlalchemy import (
    String,
    Integer,
    Float,
    Text,
    DateTime,
    ForeignKey,
    Index,
    Uuid,
    JSON,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.db.base import Base

# Portable JSON type: JSONB on PostgreSQL, JSON on SQLite
PortableJSON = JSON().with_variant(JSONB, "postgresql")


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class GraphEntity(Base):
    """
    Node in the Knowledge Graph.
    Represents an extracted entity (Person, Organization, Technology, Project, etc.)
    with canonical name, entity type, aliases, and structured properties.
    """
    __tablename__ = "graph_entities"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    canonical_name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    entity_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    aliases: Mapped[dict] = mapped_column(PortableJSON, default=list, nullable=False)
    properties: Mapped[dict] = mapped_column(PortableJSON, default=dict, nullable=False)
    extraction_method: Mapped[str] = mapped_column(String(50), default="rule", nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False
    )

    # Relationships
    outgoing_relations = relationship(
        "GraphRelation",
        foreign_keys="GraphRelation.source_entity_id",
        back_populates="source_entity",
        cascade="all, delete-orphan",
    )
    incoming_relations = relationship(
        "GraphRelation",
        foreign_keys="GraphRelation.target_entity_id",
        back_populates="target_entity",
        cascade="all, delete-orphan",
    )
    evidence_items = relationship(
        "GraphEvidence",
        foreign_keys="GraphEvidence.entity_id",
        back_populates="entity",
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        UniqueConstraint("canonical_name", "entity_type", name="uq_graph_entities_name_type"),
        Index("ix_graph_entities_type_name", "entity_type", "canonical_name"),
    )


class GraphRelation(Base):
    """
    Edge in the Knowledge Graph.
    Represents a directed relationship between two entities (e.g. WORKED_AT, USES, HAS_SKILL).
    """
    __tablename__ = "graph_relations"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    source_entity_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("graph_entities.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    target_entity_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("graph_entities.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    relation_type: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    properties: Mapped[dict] = mapped_column(PortableJSON, default=dict, nullable=False)
    extraction_method: Mapped[str] = mapped_column(String(50), default="rule", nullable=False)
    confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )

    # Relationships
    source_entity = relationship("GraphEntity", foreign_keys=[source_entity_id], back_populates="outgoing_relations")
    target_entity = relationship("GraphEntity", foreign_keys=[target_entity_id], back_populates="incoming_relations")
    evidence_items = relationship(
        "GraphEvidence",
        foreign_keys="GraphEvidence.relation_id",
        back_populates="relation",
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        Index("ix_graph_relations_source_rel", "source_entity_id", "relation_type"),
        Index("ix_graph_relations_target_rel", "target_entity_id", "relation_type"),
        Index("ix_graph_relations_triple", "source_entity_id", "relation_type", "target_entity_id"),
    )


class GraphExtractionRun(Base):
    """
    Audit log of Knowledge Graph extraction runs.
    Records model, configuration, document target, execution metrics, and status.
    """
    __tablename__ = "graph_extraction_runs"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    model_id: Mapped[str] = mapped_column(String(100), nullable=False)
    extraction_config: Mapped[dict] = mapped_column(PortableJSON, default=dict, nullable=False)
    status: Mapped[str] = mapped_column(String(50), default="completed", nullable=False)
    metrics: Mapped[dict] = mapped_column(PortableJSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )

    # Relationships
    document = relationship("Document")
    evidence_items = relationship("GraphEvidence", back_populates="extraction_run", cascade="all, delete-orphan")


class GraphEvidence(Base):
    """
    Provenance binding linking graph nodes and edges to original source chunks and document pages.
    Guarantees that no graph relation or entity is ungrounded.
    """
    __tablename__ = "graph_evidence"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    entity_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("graph_entities.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    relation_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("graph_relations.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    chunk_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("document_chunks.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    section_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("document_sections.id", ondelete="SET NULL"),
        nullable=True,
    )
    evidence_text: Mapped[str] = mapped_column(Text, nullable=False)
    page_reference: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    extraction_run_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("graph_extraction_runs.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )

    # Relationships
    entity = relationship("GraphEntity", foreign_keys=[entity_id], back_populates="evidence_items")
    relation = relationship("GraphRelation", foreign_keys=[relation_id], back_populates="evidence_items")
    document = relationship("Document")
    chunk = relationship("DocumentChunk")
    section = relationship("DocumentSection")
    extraction_run = relationship("GraphExtractionRun", back_populates="evidence_items")

    __table_args__ = (
        Index("ix_graph_evidence_doc_chunk", "document_id", "chunk_id"),
    )
