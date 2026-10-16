"""create retrieval_indexes and retrieval_experiments tables

Revision ID: 0005_create_retrieval_tables
Revises: 0004_create_embeddings_tables
Create Date: 2026-10-15 00:10:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, JSONB

# revision identifiers, used by Alembic.
revision: str = "0005_create_retrieval_tables"
down_revision: Union[str, None] = "0004_create_embeddings_tables"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Add vector_dimension column to chunk_embeddings
    op.add_column(
        "chunk_embeddings",
        sa.Column(
            "vector_dimension",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("384"),
        ),
    )
    # Backfill vector_dimension from embedding_experiments
    op.execute(
        """
        UPDATE chunk_embeddings ce
        SET vector_dimension = ee.vector_dimension
        FROM embedding_experiments ee
        WHERE ce.experiment_id = ee.id
        """
    )
    op.create_index(
        "ix_chunk_embeddings_dimension",
        "chunk_embeddings",
        ["vector_dimension"],
    )

    # 2. Create retrieval_indexes table
    op.create_table(
        "retrieval_indexes",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("index_name", sa.String(length=100), nullable=False, unique=True),
        sa.Column("index_type", sa.String(length=20), nullable=False),
        sa.Column("distance_metric", sa.String(length=20), nullable=False),
        sa.Column("opclass", sa.String(length=30), nullable=False),
        sa.Column("vector_dimension", sa.Integer(), nullable=False),
        sa.Column("model_id", sa.String(length=150), nullable=False),
        sa.Column(
            "embedding_experiment_id",
            UUID(as_uuid=True),
            sa.ForeignKey("embedding_experiments.id", ondelete="CASCADE"),
            nullable=True,
        ),
        sa.Column(
            "parameters",
            JSONB(),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "status",
            sa.String(length=20),
            nullable=False,
            server_default=sa.text("'ready'"),
        ),
        sa.Column(
            "build_duration_ms",
            sa.Float(),
            nullable=False,
            server_default=sa.text("0.0"),
        ),
        sa.Column(
            "index_size_bytes",
            sa.BigInteger(),
            nullable=False,
            server_default=sa.text("0"),
        ),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    op.create_index(
        "ix_retrieval_indexes_name", "retrieval_indexes", ["index_name"]
    )
    op.create_index(
        "ix_retrieval_indexes_model_id", "retrieval_indexes", ["model_id"]
    )
    op.create_index(
        "ix_retrieval_indexes_type_metric",
        "retrieval_indexes",
        ["index_type", "distance_metric"],
    )

    # 3. Create retrieval_experiments table
    op.create_table(
        "retrieval_experiments",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("name", sa.String(length=150), nullable=False),
        sa.Column("query_text", sa.Text(), nullable=False),
        sa.Column("model_id", sa.String(length=150), nullable=False),
        sa.Column("distance_metric", sa.String(length=20), nullable=False),
        sa.Column("top_k", sa.Integer(), nullable=False, server_default=sa.text("5")),
        sa.Column(
            "filters",
            JSONB(),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "results",
            JSONB(),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column(
            "metrics",
            JSONB(),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    op.create_index(
        "ix_retrieval_experiments_created",
        "retrieval_experiments",
        ["created_at"],
    )


def downgrade() -> None:
    op.drop_table("retrieval_experiments")
    op.drop_table("retrieval_indexes")
    op.drop_index("ix_chunk_embeddings_dimension", "chunk_embeddings")
    op.drop_column("chunk_embeddings", "vector_dimension")
