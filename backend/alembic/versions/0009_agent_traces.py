"""create agent_traces table for phase 10 agentic rag

Revision ID: 0009_agent_traces
Revises: 0008_conversational_tables
Create Date: 2026-10-16 10:10:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

# revision identifiers, used by Alembic.
revision: str = "0009_agent_traces"
down_revision: Union[str, None] = "0008_conversational_tables"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    is_postgres = bind.dialect.name == "postgresql"
    json_type = JSONB if is_postgres else sa.JSON

    op.create_table(
        "agent_traces",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("conversation_id", sa.Uuid(), nullable=True),
        sa.Column("strategy", sa.String(length=50), nullable=False),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("final_answer", sa.Text(), nullable=True),
        sa.Column("stop_reason", sa.String(length=50), nullable=False),
        sa.Column("total_steps", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("tool_calls_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("evidence_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("latency_ms", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("tokens_used", sa.Integer(), nullable=True),
        sa.Column("steps_data", json_type, nullable=False, server_default=sa.text("'[]'::jsonb" if is_postgres else "'[]'")),
        sa.Column("metadata", json_type, nullable=False, server_default=sa.text("'{}'::jsonb" if is_postgres else "'{}'")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.ForeignKeyConstraint(["conversation_id"], ["conversations.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_agent_traces_created_at", "agent_traces", ["created_at"], unique=False)
    op.create_index("ix_agent_traces_strategy_created", "agent_traces", ["strategy", "created_at"], unique=False)
    op.create_index(op.f("ix_agent_traces_conversation_id"), "agent_traces", ["conversation_id"], unique=False)
    op.create_index(op.f("ix_agent_traces_strategy"), "agent_traces", ["strategy"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_agent_traces_strategy"), table_name="agent_traces")
    op.drop_index(op.f("ix_agent_traces_conversation_id"), table_name="agent_traces")
    op.drop_index("ix_agent_traces_strategy_created", table_name="agent_traces")
    op.drop_index("ix_agent_traces_created_at", table_name="agent_traces")
    op.drop_table("agent_traces")
