import uuid
from datetime import datetime, timezone
from typing import Optional, List
from sqlalchemy import (
    String,
    Integer,
    Float,
    Text,
    Boolean,
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


class EvaluationDataset(Base):
    """
    Persistent Golden Evaluation Dataset for benchmarking RAG pipelines.
    """
    __tablename__ = "evaluation_datasets"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(String(150), nullable=False, index=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    document_collection_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        Uuid(as_uuid=True), nullable=True
    )
    metadata_json: Mapped[dict] = mapped_column(PortableJSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False
    )

    # Relationships
    questions = relationship(
        "EvaluationQuestion",
        back_populates="dataset",
        cascade="all, delete-orphan",
        order_by="EvaluationQuestion.created_at",
    )
    runs = relationship("EvaluationRun", back_populates="dataset")


class EvaluationQuestion(Base):
    """
    Individual test question within a Golden Dataset with ground truth evidence and review status.
    """
    __tablename__ = "evaluation_questions"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    dataset_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("evaluation_datasets.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    question: Mapped[str] = mapped_column(Text, nullable=False)
    reference_answer: Mapped[str] = mapped_column(Text, nullable=False)
    relevant_document_ids: Mapped[list] = mapped_column(PortableJSON, default=list, nullable=False)
    relevant_chunk_ids: Mapped[list] = mapped_column(PortableJSON, default=list, nullable=False)
    expected_evidence: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    category: Mapped[str] = mapped_column(
        String(50), default="factual", nullable=False, index=True
    )  # factual, multi_chunk, multi_document, paraphrasing, ambiguous, unanswerable, conflicting, prompt_injection
    difficulty: Mapped[str] = mapped_column(
        String(20), default="medium", nullable=False
    )  # easy, medium, hard
    is_answerable: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    is_reviewed: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False, index=True)
    reviewer_notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    relevance_labels: Mapped[dict] = mapped_column(PortableJSON, default=dict, nullable=False)
    audit_history: Mapped[list] = mapped_column(PortableJSON, default=list, nullable=False)
    revision: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    key_evidence_spans: Mapped[list] = mapped_column(PortableJSON, default=list, nullable=False)
    metadata_json: Mapped[dict] = mapped_column(PortableJSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )

    # Relationships
    dataset = relationship("EvaluationDataset", back_populates="questions")

    __table_args__ = (
        Index("ix_evaluation_questions_category_reviewed", "category", "is_reviewed"),
    )


class EvaluationRun(Base):
    """
    Historical execution record of an evaluation benchmark across a dataset.
    """
    __tablename__ = "evaluation_runs"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    dataset_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("evaluation_datasets.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    name: Mapped[str] = mapped_column(String(150), nullable=False)
    status: Mapped[str] = mapped_column(
        String(30), default="pending", nullable=False, index=True
    )  # pending, running, completed, cancelled, failed
    run_type: Mapped[str] = mapped_column(
        String(30), default="end_to_end", nullable=False
    )  # retrieval_only, generation_only, end_to_end
    config: Mapped[dict] = mapped_column(PortableJSON, default=dict, nullable=False)
    aggregate_metrics: Mapped[dict] = mapped_column(PortableJSON, default=dict, nullable=False)
    latency_summary: Mapped[dict] = mapped_column(PortableJSON, default=dict, nullable=False)
    token_summary: Mapped[dict] = mapped_column(PortableJSON, default=dict, nullable=False)
    cost_summary: Mapped[dict] = mapped_column(PortableJSON, default=dict, nullable=False)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False, index=True
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Relationships
    dataset = relationship("EvaluationDataset", back_populates="runs")
    item_results = relationship(
        "EvaluationItemResult",
        back_populates="run",
        cascade="all, delete-orphan",
        order_by="EvaluationItemResult.created_at",
    )


class EvaluationItemResult(Base):
    """
    Detailed granular evaluation outcome for an individual question in a run.
    """
    __tablename__ = "evaluation_item_results"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    run_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("evaluation_runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    question_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        Uuid(as_uuid=True), nullable=True, index=True
    )
    question_text: Mapped[str] = mapped_column(Text, nullable=False)
    reference_answer: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    generated_answer: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    retrieved_chunk_ids: Mapped[list] = mapped_column(PortableJSON, default=list, nullable=False)
    relevant_chunk_ids: Mapped[list] = mapped_column(PortableJSON, default=list, nullable=False)
    retrieval_metrics: Mapped[dict] = mapped_column(PortableJSON, default=dict, nullable=False)
    faithfulness_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    answer_relevance_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    citation_metrics: Mapped[dict] = mapped_column(PortableJSON, default=dict, nullable=False)
    claim_judgments: Mapped[list] = mapped_column(PortableJSON, default=list, nullable=False)
    latency_breakdown: Mapped[dict] = mapped_column(PortableJSON, default=dict, nullable=False)
    token_breakdown: Mapped[dict] = mapped_column(PortableJSON, default=dict, nullable=False)
    estimated_cost_usd: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    status: Mapped[str] = mapped_column(
        String(20), default="success", nullable=False
    )  # success, failed, skipped
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )

    # Relationships
    run = relationship("EvaluationRun", back_populates="item_results")
