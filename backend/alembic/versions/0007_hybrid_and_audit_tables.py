"""add tsv_content and GIN index to document_chunks, and audit fields to evaluation_questions

Revision ID: 0007_create_hybrid_and_audit_tables
Revises: 0006_create_evaluation_tables
Create Date: 2026-10-15 03:00:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

# revision identifiers, used by Alembic.
revision: str = "0007_hybrid_and_audit_tables"
down_revision: Union[str, None] = "0006_create_evaluation_tables"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    is_postgres = bind.dialect.name == "postgresql"

    # 1. document_chunks: add tsv_content and GIN index
    if is_postgres:
        # PostgreSQL generated tsvector column and GIN index
        op.execute(
            """
            ALTER TABLE document_chunks
            ADD COLUMN IF NOT EXISTS tsv_content tsvector
            GENERATED ALWAYS AS (to_tsvector('english', coalesce(content, ''))) STORED;
            """
        )
        op.execute(
            """
            CREATE INDEX IF NOT EXISTS ix_document_chunks_tsv_content
            ON document_chunks USING gin(tsv_content);
            """
        )
    else:
        # SQLite / other fallback
        with op.batch_alter_table("document_chunks") as batch_op:
            batch_op.add_column(sa.Column("tsv_content", sa.Text(), nullable=True))

    # 2. evaluation_questions: add graded relevance labels, audit history, revision, and key evidence spans
    if is_postgres:
        with op.batch_alter_table("evaluation_questions") as batch_op:
            batch_op.add_column(
                sa.Column(
                    "relevance_labels",
                    JSONB(),
                    nullable=False,
                    server_default=sa.text("'{}'::jsonb"),
                )
            )
            batch_op.add_column(
                sa.Column(
                    "audit_history",
                    JSONB(),
                    nullable=False,
                    server_default=sa.text("'[]'::jsonb"),
                )
            )
            batch_op.add_column(
                sa.Column(
                    "revision",
                    sa.Integer(),
                    nullable=False,
                    server_default=sa.text("1"),
                )
            )
            batch_op.add_column(
                sa.Column(
                    "key_evidence_spans",
                    JSONB(),
                    nullable=False,
                    server_default=sa.text("'[]'::jsonb"),
                )
            )
    else:
        with op.batch_alter_table("evaluation_questions") as batch_op:
            batch_op.add_column(
                sa.Column(
                    "relevance_labels",
                    sa.JSON(),
                    nullable=False,
                    server_default=sa.text("'{}'"),
                )
            )
            batch_op.add_column(
                sa.Column(
                    "audit_history",
                    sa.JSON(),
                    nullable=False,
                    server_default=sa.text("'[]'"),
                )
            )
            batch_op.add_column(
                sa.Column(
                    "revision",
                    sa.Integer(),
                    nullable=False,
                    server_default=sa.text("1"),
                )
            )
            batch_op.add_column(
                sa.Column(
                    "key_evidence_spans",
                    sa.JSON(),
                    nullable=False,
                    server_default=sa.text("'[]'"),
                )
            )


def downgrade() -> None:
    bind = op.get_bind()
    is_postgres = bind.dialect.name == "postgresql"

    # 2. evaluation_questions
    with op.batch_alter_table("evaluation_questions") as batch_op:
        batch_op.drop_column("key_evidence_spans")
        batch_op.drop_column("revision")
        batch_op.drop_column("audit_history")
        batch_op.drop_column("relevance_labels")

    # 1. document_chunks
    if is_postgres:
        op.execute("DROP INDEX IF EXISTS ix_document_chunks_tsv_content;")
        op.execute("ALTER TABLE document_chunks DROP COLUMN IF EXISTS tsv_content;")
    else:
        with op.batch_alter_table("document_chunks") as batch_op:
            batch_op.drop_column("tsv_content")
