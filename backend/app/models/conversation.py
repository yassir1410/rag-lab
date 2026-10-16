import uuid
from datetime import datetime, timezone
from typing import Optional, List
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
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.db.base import Base

# Portable JSON type: JSONB on PostgreSQL, JSON on SQLite
PortableJSON = JSON().with_variant(JSONB, "postgresql")


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Conversation(Base):
    """
    Persistent conversational session for multi-turn RAG.
    Tracks session-level configuration, active memory strategy, and status.
    """
    __tablename__ = "conversations"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    title: Mapped[str] = mapped_column(
        String(255), default="Nouvelle Conversation", nullable=False, index=True
    )
    settings: Mapped[dict] = mapped_column(
        PortableJSON, default=dict, nullable=False
    )
    status: Mapped[str] = mapped_column(
        String(50), default="active", nullable=False, index=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False
    )

    # Cascading relationships
    messages = relationship(
        "ConversationMessage",
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="ConversationMessage.sequence_number",
    )
    summaries = relationship(
        "ConversationSummary",
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="ConversationSummary.created_at",
    )


class ConversationMessage(Base):
    """
    Individual turn message in a multi-turn conversation.
    Enforces strict sequence ordering, role isolation, and complete telemetry.
    """
    __tablename__ = "conversation_messages"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    role: Mapped[str] = mapped_column(
        String(50), nullable=False
    )  # 'user' | 'assistant'
    content: Mapped[str] = mapped_column(
        Text, nullable=False
    )
    sequence_number: Mapped[int] = mapped_column(
        Integer, nullable=False
    )
    metadata_json: Mapped[dict] = mapped_column(
        PortableJSON, default=dict, nullable=False
    )
    trace_id: Mapped[Optional[str]] = mapped_column(
        String(100), nullable=True, index=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )

    conversation = relationship("Conversation", back_populates="messages")

    __table_args__ = (
        Index("ix_conv_msg_order", "conversation_id", "sequence_number"),
    )


class ConversationSummary(Base):
    """
    Rolling conversational summary covering older dialogue turns.
    Preserves factual state and constraints while capping token consumption.
    """
    __tablename__ = "conversation_summaries"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    summary_text: Mapped[str] = mapped_column(
        Text, nullable=False
    )
    covered_until_sequence: Mapped[int] = mapped_column(
        Integer, nullable=False
    )
    summarizer_model: Mapped[Optional[str]] = mapped_column(
        String(100), nullable=True
    )
    metadata_json: Mapped[dict] = mapped_column(
        PortableJSON, default=dict, nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )

    conversation = relationship("Conversation", back_populates="summaries")

    __table_args__ = (
        Index("ix_conv_summary_seq", "conversation_id", "covered_until_sequence"),
    )
