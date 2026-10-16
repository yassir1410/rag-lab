"""create document_page_assets, document_elements, and multimodal_embeddings tables

Revision ID: 0011_multimodal_rag_tables
Revises: 0010_graph_rag_tables
Create Date: 2026-10-16 11:15:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB
from pgvector.sqlalchemy import Vector

# revision identifiers, used by Alembic.
revision: str = "0011_multimodal_rag_tables"
down_revision: Union[str, None] = "0010_graph_rag_tables"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    is_postgres = bind.dialect.name == "postgresql"
    json_type = JSONB if is_postgres else sa.JSON

    # 1. document_page_assets
    op.create_table(
        "document_page_assets",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("document_id", sa.Uuid(), nullable=False),
        sa.Column("page_number", sa.Integer(), nullable=False),
        sa.Column("width", sa.Integer(), nullable=False, server_default="595"),
        sa.Column("height", sa.Integer(), nullable=False, server_default="842"),
        sa.Column("has_native_text", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("native_text_char_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("is_scanned", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("image_path", sa.String(length=500), nullable=False),
        sa.Column("ocr_text", sa.Text(), nullable=True),
        sa.Column("ocr_confidence", sa.Float(), nullable=True),
        sa.Column("ocr_language", sa.String(length=20), nullable=False, server_default="eng"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["document_id"], ["documents.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_doc_page_assets_doc_page",
        "document_page_assets",
        ["document_id", "page_number"],
    )

    # 2. document_elements
    op.create_table(
        "document_elements",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("document_id", sa.Uuid(), nullable=False),
        sa.Column("page_asset_id", sa.Uuid(), nullable=True),
        sa.Column("page_number", sa.Integer(), nullable=False),
        sa.Column("element_type", sa.String(length=50), nullable=False),
        sa.Column("reading_order", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("bounding_box", json_type, nullable=False),
        sa.Column("content_text", sa.Text(), nullable=False, server_default=""),
        sa.Column("structured_data", json_type, nullable=False),
        sa.Column("asset_path", sa.String(length=500), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=False, server_default="1.0"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["document_id"], ["documents.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["page_asset_id"], ["document_page_assets.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_doc_elements_doc_type",
        "document_elements",
        ["document_id", "element_type"],
    )
    op.create_index(
        "ix_doc_elements_doc_page_order",
        "document_elements",
        ["document_id", "page_number", "reading_order"],
    )

    # 3. multimodal_embeddings
    vector_type = Vector(512) if is_postgres else sa.NullType()
    op.create_table(
        "multimodal_embeddings",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("document_id", sa.Uuid(), nullable=False),
        sa.Column("element_id", sa.Uuid(), nullable=True),
        sa.Column("page_asset_id", sa.Uuid(), nullable=True),
        sa.Column("model_id", sa.String(length=100), nullable=False, server_default="clip-ViT-B-32"),
        sa.Column("embedding_type", sa.String(length=50), nullable=False, server_default="image"),
        sa.Column("vector_dimension", sa.Integer(), nullable=False, server_default="512"),
        sa.Column("vector", vector_type, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["document_id"], ["documents.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["element_id"], ["document_elements.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["page_asset_id"], ["document_page_assets.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_mm_embeddings_doc_type",
        "multimodal_embeddings",
        ["document_id", "embedding_type"],
    )
    op.create_index(
        "ix_mm_embeddings_model",
        "multimodal_embeddings",
        ["model_id"],
    )


def downgrade() -> None:
    op.drop_table("multimodal_embeddings")
    op.drop_table("document_elements")
    op.drop_table("document_page_assets")
