"""add evaluation observability, traces, routing metrics, and dataset extensions

Revision ID: 0012_evaluation_observability
Revises: 0011_multimodal_rag_tables
Create Date: 2026-10-17 20:30:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

# revision identifiers, used by Alembic.
revision: str = "0012_evaluation_observability"
down_revision: Union[str, None] = "0011_multimodal_rag_tables"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    is_postgres = bind.dialect.name == "postgresql"
    json_type = JSONB if is_postgres else sa.JSON
    empty_list_default = sa.text("'[]'::jsonb") if is_postgres else sa.text("'[]'")
    empty_dict_default = sa.text("'{}'::jsonb") if is_postgres else sa.text("'{}'")

    # 1. evaluation_datasets extensions
    with op.batch_alter_table("evaluation_datasets") as batch_op:
        batch_op.add_column(
            sa.Column("version", sa.String(50), nullable=False, server_default="1.0")
        )
        batch_op.add_column(
            sa.Column("export_format", sa.String(20), nullable=False, server_default="json")
        )

    # 2. evaluation_questions extensions
    with op.batch_alter_table("evaluation_questions") as batch_op:
        batch_op.add_column(
            sa.Column("acceptable_answers", json_type, nullable=False, server_default=empty_list_default)
        )
        batch_op.add_column(
            sa.Column("expected_routes", json_type, nullable=False, server_default=empty_list_default)
        )
        batch_op.add_column(
            sa.Column("expected_citations", json_type, nullable=False, server_default=empty_list_default)
        )
        batch_op.add_column(
            sa.Column("modality", sa.String(30), nullable=False, server_default="text")
        )
        batch_op.add_column(
            sa.Column("tags", json_type, nullable=False, server_default=empty_list_default)
        )
        batch_op.add_column(
            sa.Column("label_source", sa.String(30), nullable=False, server_default="human")
        )
        batch_op.add_column(
            sa.Column("numerical_answer", json_type, nullable=True)
        )
        batch_op.add_column(
            sa.Column("document_scope", json_type, nullable=False, server_default=empty_list_default)
        )

    # 3. evaluation_runs extensions
    with op.batch_alter_table("evaluation_runs") as batch_op:
        batch_op.add_column(
            sa.Column("progress_current", sa.Integer(), nullable=False, server_default="0")
        )
        batch_op.add_column(
            sa.Column("progress_total", sa.Integer(), nullable=False, server_default="0")
        )
        batch_op.add_column(
            sa.Column("experiment_snapshot", json_type, nullable=False, server_default=empty_dict_default)
        )
        batch_op.add_column(
            sa.Column("routing_summary", json_type, nullable=False, server_default=empty_dict_default)
        )

    # 4. evaluation_item_results extensions
    with op.batch_alter_table("evaluation_item_results") as batch_op:
        batch_op.add_column(
            sa.Column("routing_metrics", json_type, nullable=False, server_default=empty_dict_default)
        )
        batch_op.add_column(
            sa.Column("accuracy_metrics", json_type, nullable=False, server_default=empty_dict_default)
        )
        batch_op.add_column(
            sa.Column("provider_reported_tokens", json_type, nullable=False, server_default=empty_dict_default)
        )
        batch_op.add_column(
            sa.Column("measurement_source", sa.String(20), nullable=False, server_default="measured")
        )
        batch_op.add_column(
            sa.Column("trace_id", sa.String(100), nullable=True)
        )
        batch_op.create_index("ix_evaluation_item_results_trace_id", ["trace_id"])

    # 5. create evaluation_traces table
    op.create_table(
        "evaluation_traces",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column(
            "run_id",
            sa.Uuid(),
            sa.ForeignKey("evaluation_runs.id", ondelete="CASCADE"),
            nullable=True,
        ),
        sa.Column(
            "item_result_id",
            sa.Uuid(),
            sa.ForeignKey("evaluation_item_results.id", ondelete="CASCADE"),
            nullable=True,
        ),
        sa.Column("trace_id", sa.String(100), nullable=False),
        sa.Column("request_id", sa.String(100), nullable=True),
        sa.Column("document_ids", json_type, nullable=False, server_default=empty_list_default),
        sa.Column("selected_routes", json_type, nullable=False, server_default=empty_list_default),
        sa.Column("provider", sa.String(50), nullable=False, server_default="mock"),
        sa.Column("model", sa.String(100), nullable=False, server_default="mock-instruct-v1"),
        sa.Column("stages", json_type, nullable=False, server_default=empty_list_default),
        sa.Column("total_latency_ms", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("error_info", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_evaluation_traces_trace_id", "evaluation_traces", ["trace_id"])
    op.create_index("ix_evaluation_traces_run_id", "evaluation_traces", ["run_id"])
    op.create_index("ix_evaluation_traces_item_result_id", "evaluation_traces", ["item_result_id"])


def downgrade() -> None:
    op.drop_table("evaluation_traces")

    with op.batch_alter_table("evaluation_item_results") as batch_op:
        batch_op.drop_index("ix_evaluation_item_results_trace_id")
        batch_op.drop_column("trace_id")
        batch_op.drop_column("measurement_source")
        batch_op.drop_column("provider_reported_tokens")
        batch_op.drop_column("accuracy_metrics")
        batch_op.drop_column("routing_metrics")

    with op.batch_alter_table("evaluation_runs") as batch_op:
        batch_op.drop_column("routing_summary")
        batch_op.drop_column("experiment_snapshot")
        batch_op.drop_column("progress_total")
        batch_op.drop_column("progress_current")

    with op.batch_alter_table("evaluation_questions") as batch_op:
        batch_op.drop_column("document_scope")
        batch_op.drop_column("numerical_answer")
        batch_op.drop_column("label_source")
        batch_op.drop_column("tags")
        batch_op.drop_column("modality")
        batch_op.drop_column("expected_citations")
        batch_op.drop_column("expected_routes")
        batch_op.drop_column("acceptable_answers")

    with op.batch_alter_table("evaluation_datasets") as batch_op:
        batch_op.drop_column("export_format")
        batch_op.drop_column("version")
