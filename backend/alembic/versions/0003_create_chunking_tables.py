"""create chunking_experiments and document_chunks tables

Revision ID: 0003_create_chunking_tables
Revises: 0002_create_documents_tables
Create Date: 2026-10-14 21:00:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, JSONB

# revision identifiers, used by Alembic.
revision: str = '0003_create_chunking_tables'
down_revision: Union[str, None] = '0002_create_documents_tables'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Create chunking_experiments table
    op.create_table(
        'chunking_experiments',
        sa.Column('id', UUID(as_uuid=True), primary_key=True),
        sa.Column('document_id', UUID(as_uuid=True), sa.ForeignKey('documents.id', ondelete='CASCADE'), nullable=False),
        sa.Column('strategy', sa.String(length=50), nullable=False),
        sa.Column('config', JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column('metrics', JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index('ix_chunking_experiments_doc_created', 'chunking_experiments', ['document_id', 'created_at'])

    # 2. Create document_chunks table
    op.create_table(
        'document_chunks',
        sa.Column('id', UUID(as_uuid=True), primary_key=True),
        sa.Column('experiment_id', UUID(as_uuid=True), sa.ForeignKey('chunking_experiments.id', ondelete='CASCADE'), nullable=True),
        sa.Column('document_id', UUID(as_uuid=True), sa.ForeignKey('documents.id', ondelete='CASCADE'), nullable=False),
        sa.Column('sequence_index', sa.Integer(), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('token_count', sa.Integer(), nullable=False),
        sa.Column('char_count', sa.Integer(), nullable=False),
        sa.Column('parent_chunk_id', UUID(as_uuid=True), sa.ForeignKey('document_chunks.id', ondelete='SET NULL'), nullable=True),
        sa.Column('metadata', JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index('ix_document_chunks_exp_seq', 'document_chunks', ['experiment_id', 'sequence_index'])
    op.create_index('ix_document_chunks_doc_seq', 'document_chunks', ['document_id', 'sequence_index'])


def downgrade() -> None:
    op.drop_table('document_chunks')
    op.drop_table('chunking_experiments')
