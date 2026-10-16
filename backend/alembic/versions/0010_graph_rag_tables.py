"""create graph_entities, graph_relations, graph_extraction_runs, and graph_evidence tables

Revision ID: 0010_graph_rag_tables
Revises: 0009_agent_traces
Create Date: 2026-10-16 11:00:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

# revision identifiers, used by Alembic.
revision: str = "0010_graph_rag_tables"
down_revision: Union[str, None] = "0009_agent_traces"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    is_postgres = bind.dialect.name == "postgresql"
    json_type = JSONB if is_postgres else sa.JSON

    # 1. graph_entities
    op.create_table(
        "graph_entities",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("canonical_name", sa.String(length=255), nullable=False),
        sa.Column("entity_type", sa.String(length=50), nullable=False),
        sa.Column("aliases", json_type, nullable=False, server_default=sa.text("'[]'::jsonb" if is_postgres else "'[]'")),
        sa.Column("properties", json_type, nullable=False, server_default=sa.text("'{}'::jsonb" if is_postgres else "'{}'")),
        sa.Column("extraction_method", sa.String(length=50), nullable=False, server_default="rule"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("canonical_name", "entity_type", name="uq_graph_entities_name_type"),
    )
    op.create_index(op.f("ix_graph_entities_canonical_name"), "graph_entities", ["canonical_name"], unique=False)
    op.create_index(op.f("ix_graph_entities_entity_type"), "graph_entities", ["entity_type"], unique=False)
    op.create_index("ix_graph_entities_type_name", "graph_entities", ["entity_type", "canonical_name"], unique=False)

    # 2. graph_relations
    op.create_table(
        "graph_relations",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("source_entity_id", sa.Uuid(), nullable=False),
        sa.Column("target_entity_id", sa.Uuid(), nullable=False),
        sa.Column("relation_type", sa.String(length=100), nullable=False),
        sa.Column("properties", json_type, nullable=False, server_default=sa.text("'{}'::jsonb" if is_postgres else "'{}'")),
        sa.Column("extraction_method", sa.String(length=50), nullable=False, server_default="rule"),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.ForeignKeyConstraint(["source_entity_id"], ["graph_entities.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["target_entity_id"], ["graph_entities.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_graph_relations_source_entity_id"), "graph_relations", ["source_entity_id"], unique=False)
    op.create_index(op.f("ix_graph_relations_target_entity_id"), "graph_relations", ["target_entity_id"], unique=False)
    op.create_index(op.f("ix_graph_relations_relation_type"), "graph_relations", ["relation_type"], unique=False)
    op.create_index("ix_graph_relations_source_rel", "graph_relations", ["source_entity_id", "relation_type"], unique=False)
    op.create_index("ix_graph_relations_target_rel", "graph_relations", ["target_entity_id", "relation_type"], unique=False)
    op.create_index("ix_graph_relations_triple", "graph_relations", ["source_entity_id", "relation_type", "target_entity_id"], unique=False)

    # 3. graph_extraction_runs
    op.create_table(
        "graph_extraction_runs",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("document_id", sa.Uuid(), nullable=False),
        sa.Column("model_id", sa.String(length=100), nullable=False),
        sa.Column("extraction_config", json_type, nullable=False, server_default=sa.text("'{}'::jsonb" if is_postgres else "'{}'")),
        sa.Column("status", sa.String(length=50), nullable=False, server_default="completed"),
        sa.Column("metrics", json_type, nullable=False, server_default=sa.text("'{}'::jsonb" if is_postgres else "'{}'")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.ForeignKeyConstraint(["document_id"], ["documents.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_graph_extraction_runs_document_id"), "graph_extraction_runs", ["document_id"], unique=False)

    # 4. graph_evidence
    op.create_table(
        "graph_evidence",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("entity_id", sa.Uuid(), nullable=True),
        sa.Column("relation_id", sa.Uuid(), nullable=True),
        sa.Column("document_id", sa.Uuid(), nullable=False),
        sa.Column("chunk_id", sa.Uuid(), nullable=True),
        sa.Column("section_id", sa.Uuid(), nullable=True),
        sa.Column("evidence_text", sa.Text(), nullable=False),
        sa.Column("page_reference", sa.Integer(), nullable=True),
        sa.Column("extraction_run_id", sa.Uuid(), nullable=True),
        sa.ForeignKeyConstraint(["entity_id"], ["graph_entities.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["relation_id"], ["graph_relations.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["document_id"], ["documents.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["chunk_id"], ["document_chunks.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["section_id"], ["document_sections.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["extraction_run_id"], ["graph_extraction_runs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_graph_evidence_entity_id"), "graph_evidence", ["entity_id"], unique=False)
    op.create_index(op.f("ix_graph_evidence_relation_id"), "graph_evidence", ["relation_id"], unique=False)
    op.create_index(op.f("ix_graph_evidence_document_id"), "graph_evidence", ["document_id"], unique=False)
    op.create_index(op.f("ix_graph_evidence_chunk_id"), "graph_evidence", ["chunk_id"], unique=False)
    op.create_index(op.f("ix_graph_evidence_extraction_run_id"), "graph_evidence", ["extraction_run_id"], unique=False)
    op.create_index("ix_graph_evidence_doc_chunk", "graph_evidence", ["document_id", "chunk_id"], unique=False)


def downgrade() -> None:
    op.drop_table("graph_evidence")
    op.drop_table("graph_extraction_runs")
    op.drop_table("graph_relations")
    op.drop_table("graph_entities")
