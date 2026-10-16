"""create embedding_experiments and chunk_embeddings tables

Revision ID: 0004_create_embeddings_tables
Revises: 0003_create_chunking_tables
Create Date: 2026-10-14 22:30:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, JSONB
from pgvector.sqlalchemy import Vector

# revision identifiers, used by Alembic.
revision: str = '0004_create_embeddings_tables'
down_revision: Union[str, None] = '0003_create_chunking_tables'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Create embedding_experiments table
    op.create_table(
        'embedding_experiments',
        sa.Column('id', UUID(as_uuid=True), primary_key=True),
        sa.Column('name', sa.String(length=150), nullable=True),
        sa.Column('document_id', UUID(as_uuid=True), sa.ForeignKey('documents.id', ondelete='CASCADE'), nullable=True),
        sa.Column('chunking_experiment_id', UUID(as_uuid=True), sa.ForeignKey('chunking_experiments.id', ondelete='SET NULL'), nullable=True),
        sa.Column('model_id', sa.String(length=150), nullable=False),
        sa.Column('provider', sa.String(length=50), nullable=False),
        sa.Column('vector_dimension', sa.Integer(), nullable=False),
        sa.Column('is_normalized', sa.Boolean(), nullable=False, server_default=sa.text('true')),
        sa.Column('parameters', JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column('metrics', JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index('ix_embedding_experiments_model_created', 'embedding_experiments', ['model_id', 'created_at'])
    op.create_index('ix_embedding_experiments_doc_id', 'embedding_experiments', ['document_id'])

    # 2. Create chunk_embeddings table
    op.create_table(
        'chunk_embeddings',
        sa.Column('id', UUID(as_uuid=True), primary_key=True),
        sa.Column('experiment_id', UUID(as_uuid=True), sa.ForeignKey('embedding_experiments.id', ondelete='CASCADE'), nullable=False),
        sa.Column('chunk_id', UUID(as_uuid=True), sa.ForeignKey('document_chunks.id', ondelete='CASCADE'), nullable=True),
        sa.Column('sequence_index', sa.Integer(), nullable=False, server_default=sa.text('0')),
        sa.Column('text_snippet', sa.Text(), nullable=False),
        sa.Column('token_count', sa.Integer(), nullable=False, server_default=sa.text('0')),
        sa.Column('vector', Vector(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index('ix_chunk_embeddings_exp_seq', 'chunk_embeddings', ['experiment_id', 'sequence_index'])
    op.create_index('ix_chunk_embeddings_chunk_id', 'chunk_embeddings', ['chunk_id'])


def downgrade() -> None:
    op.drop_table('chunk_embeddings')
    op.drop_table('embedding_experiments')
