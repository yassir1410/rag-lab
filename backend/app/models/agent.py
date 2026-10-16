import uuid
from datetime import datetime, timezone
from typing import Optional
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
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.db.base import Base

# Portable JSON type: JSONB on PostgreSQL, JSON on SQLite
PortableJSON = JSON().with_variant(JSONB, "postgresql")


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class AgentExecutionTrace(Base):
    """
    Persistent execution trace for Phase 10 Agentic RAG runs.
    Stores strategy decisions, step-by-step Plan-Act-Observe logs, tool executions,
    evidence accumulation, latency profiles, and final cited answers.
    """
    __tablename__ = "agent_traces"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    conversation_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("conversations.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    strategy: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    final_answer: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    stop_reason: Mapped[str] = mapped_column(String(50), nullable=False)
    total_steps: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    tool_calls_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    evidence_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    latency_ms: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    tokens_used: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    steps_data: Mapped[dict] = mapped_column(
        PortableJSON, default=list, nullable=False
    )
    metadata_json: Mapped[dict] = mapped_column(
        "metadata", PortableJSON, default=dict, nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )

    __table_args__ = (
        Index("ix_agent_traces_created_at", "created_at"),
        Index("ix_agent_traces_strategy_created", "strategy", "created_at"),
    )
