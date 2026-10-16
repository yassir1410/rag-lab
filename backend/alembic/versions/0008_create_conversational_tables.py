"""create conversations, conversation_messages, and conversation_summaries tables

Revision ID: 0008_create_conversational_tables
Revises: 0007_hybrid_and_audit_tables
Create Date: 2026-10-16 10:00:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

# revision identifiers, used by Alembic.
revision: str = "0008_conversational_tables"
down_revision: Union[str, None] = "0007_hybrid_and_audit_tables"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    is_postgres = bind.dialect.name == "postgresql"
    json_type = JSONB if is_postgres else sa.JSON

    # 1. conversations table
    op.create_table(
        "conversations",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("title", sa.String(length=255), nullable=False, server_default="Nouvelle Conversation"),
        sa.Column("settings", json_type, nullable=False, server_default=sa.text("'{}'::jsonb" if is_postgres else "'{}'")),
        sa.Column("status", sa.String(length=50), nullable=False, server_default="active"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_conversations_title", "conversations", ["title"], unique=False)
    op.create_index("ix_conversations_status", "conversations", ["status"], unique=False)
    op.create_index("ix_conversations_created_at", "conversations", ["created_at"], unique=False)

    # 2. conversation_messages table
    op.create_table(
        "conversation_messages",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("conversation_id", sa.Uuid(), nullable=False),
        sa.Column("role", sa.String(length=50), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("sequence_number", sa.Integer(), nullable=False),
        sa.Column("metadata_json", json_type, nullable=False, server_default=sa.text("'{}'::jsonb" if is_postgres else "'{}'")),
        sa.Column("trace_id", sa.String(length=100), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.ForeignKeyConstraint(
            ["conversation_id"],
            ["conversations.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_conversation_messages_conversation_id", "conversation_messages", ["conversation_id"], unique=False)
    op.create_index("ix_conversation_messages_trace_id", "conversation_messages", ["trace_id"], unique=False)
    op.create_index("ix_conv_msg_order", "conversation_messages", ["conversation_id", "sequence_number"], unique=False)

    # 3. conversation_summaries table
    op.create_table(
        "conversation_summaries",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("conversation_id", sa.Uuid(), nullable=False),
        sa.Column("summary_text", sa.Text(), nullable=False),
        sa.Column("covered_until_sequence", sa.Integer(), nullable=False),
        sa.Column("summarizer_model", sa.String(length=100), nullable=True),
        sa.Column("metadata_json", json_type, nullable=False, server_default=sa.text("'{}'::jsonb" if is_postgres else "'{}'")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.ForeignKeyConstraint(
            ["conversation_id"],
            ["conversations.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_conversation_summaries_conversation_id", "conversation_summaries", ["conversation_id"], unique=False)
    op.create_index("ix_conv_summary_seq", "conversation_summaries", ["conversation_id", "covered_until_sequence"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_conv_summary_seq", table_name="conversation_summaries")
    op.drop_index("ix_conversation_summaries_conversation_id", table_name="conversation_summaries")
    op.drop_table("conversation_summaries")

    op.drop_index("ix_conv_msg_order", table_name="conversation_messages")
    op.drop_index("ix_conversation_messages_trace_id", table_name="conversation_messages")
    op.drop_index("ix_conversation_messages_conversation_id", table_name="conversation_messages")
    op.drop_table("conversation_messages")

    op.drop_index("ix_conversations_created_at", table_name="conversations")
    op.drop_index("ix_conversations_status", table_name="conversations")
    op.drop_index("ix_conversations_title", table_name="conversations")
    op.drop_table("conversations")
