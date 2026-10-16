import re
import time
import uuid
from typing import List, Dict, Any, Optional
from sqlalchemy import text, desc
from sqlalchemy.orm import Session

from app.models.retrieval import RetrievalIndex
from app.rag.retrieval.schemas import IndexCreateRequest, IndexResponse
from app.rag.retrieval.base import METRIC_TO_OPCLASS


def sanitize_ident(name: str) -> str:
    """Sanitizes an SQL identifier to prevent injection."""
    cleaned = re.sub(r"[^a-zA-Z0-9_]", "", name)
    if not cleaned or not cleaned[0].isalpha():
        cleaned = "idx_" + cleaned
    return cleaned[:63]


def format_bytes(num_bytes: int) -> str:
    """Formats bytes into human readable format."""
    if num_bytes < 1024:
        return f"{num_bytes} B"
    elif num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024:.1f} kB"
    else:
        return f"{num_bytes / (1024 * 1024):.2f} MB"


class IndexManager:
    """
    Manages the lifecycle of PostgreSQL pgvector indexes (HNSW and IVFFlat),
    validating parameters, tracking build duration, and querying physical disk sizes.
    """

    def create_index(
        self,
        session: Session,
        request: IndexCreateRequest,
        dimension: int,
    ) -> IndexResponse:
        is_postgres = session.bind.dialect.name == "postgresql" if session.bind else False
        index_type = request.index_type.lower()
        if index_type not in ("hnsw", "ivfflat"):
            raise ValueError(f"Unsupported index type '{index_type}'. Must be 'hnsw' or 'ivfflat'.")

        metric = request.distance_metric.lower()
        if metric not in METRIC_TO_OPCLASS:
            raise ValueError(
                f"Unsupported distance metric '{metric}'. Available: {list(METRIC_TO_OPCLASS.keys())}"
            )
        opclass = METRIC_TO_OPCLASS[metric]

        # Generate unique sanitized index name
        suffix = uuid.uuid4().hex[:8]
        raw_name = f"idx_{index_type}_{metric[:3]}_{dimension}_{suffix}"
        index_name = sanitize_ident(raw_name)

        # Validate parameters and construct WITH clause
        with_params = []
        clean_params = dict(request.parameters)

        if index_type == "hnsw":
            m = clean_params.get("m", 16)
            ef_c = clean_params.get("ef_construction", 64)
            try:
                m = max(4, min(64, int(m)))
                ef_c = max(8, min(512, int(ef_c)))
            except (ValueError, TypeError):
                m, ef_c = 16, 64
            clean_params["m"] = m
            clean_params["ef_construction"] = ef_c
            with_params.append(f"m = {m}")
            with_params.append(f"ef_construction = {ef_c}")

        elif index_type == "ivfflat":
            lists = clean_params.get("lists", 10)
            try:
                lists = max(1, min(1000, int(lists)))
            except (ValueError, TypeError):
                lists = 10
            clean_params["lists"] = lists
            with_params.append(f"lists = {lists}")

        with_clause = f"WITH ({', '.join(with_params)})" if with_params else ""

        # Scope index to specific experiment or table-wide dimension
        if request.embedding_experiment_id:
            where_clause = f"WHERE experiment_id = '{request.embedding_experiment_id}' AND vector IS NOT NULL"
        else:
            where_clause = f"WHERE vector_dimension = {dimension} AND vector IS NOT NULL"

        create_sql = f"""
            CREATE INDEX {index_name}
            ON chunk_embeddings
            USING {index_type} (((vector)::vector({dimension})) {opclass})
            {with_clause}
            {where_clause}
        """

        build_duration_ms = 0.0
        index_size_bytes = 0
        index_size_pretty = "0 B"
        status = "ready"
        error_msg = None

        if is_postgres:
            start_time = time.perf_counter()
            try:
                session.execute(text(create_sql))
                session.commit()
                build_duration_ms = (time.perf_counter() - start_time) * 1000.0

                # Query physical relation size
                size_row = session.execute(
                    text(f"SELECT pg_relation_size('{index_name}') AS bytes, pg_size_pretty(pg_relation_size('{index_name}')) AS pretty")
                ).fetchone()
                if size_row:
                    index_size_bytes = int(size_row.bytes or 0)
                    index_size_pretty = str(size_row.pretty or "0 B")
            except Exception as e:
                session.rollback()
                status = "failed"
                error_msg = str(e)
                raise RuntimeError(f"Failed to create {index_type.upper()} index in PostgreSQL: {e}")
        else:
            # Emulated in SQLite for tests
            build_duration_ms = 1.5
            index_size_bytes = 16384
            index_size_pretty = "16 kB (Emulated)"

        # Persist index metadata record
        db_index = RetrievalIndex(
            index_name=index_name,
            index_type=index_type,
            distance_metric=metric,
            opclass=opclass,
            vector_dimension=dimension,
            model_id=request.model_id,
            embedding_experiment_id=request.embedding_experiment_id,
            parameters=clean_params,
            status=status,
            build_duration_ms=round(build_duration_ms, 2),
            index_size_bytes=index_size_bytes,
            error_message=error_msg,
        )
        session.add(db_index)
        session.commit()
        session.refresh(db_index)

        return IndexResponse(
            id=db_index.id,
            index_name=db_index.index_name,
            index_type=db_index.index_type,
            distance_metric=db_index.distance_metric,
            opclass=db_index.opclass,
            vector_dimension=db_index.vector_dimension,
            model_id=db_index.model_id,
            embedding_experiment_id=db_index.embedding_experiment_id,
            parameters=db_index.parameters,
            status=db_index.status,
            build_duration_ms=db_index.build_duration_ms,
            index_size_bytes=db_index.index_size_bytes,
            index_size_pretty=index_size_pretty,
            error_message=db_index.error_message,
            created_at=db_index.created_at.isoformat(),
        )

    def list_indexes(self, session: Session) -> List[IndexResponse]:
        """Lists all registered indexes and updates live size info from PostgreSQL."""
        is_postgres = session.bind.dialect.name == "postgresql" if session.bind else False
        indexes = session.query(RetrievalIndex).order_by(desc(RetrievalIndex.created_at)).all()

        results = []
        for idx in indexes:
            size_pretty = format_bytes(idx.index_size_bytes)
            if is_postgres and idx.status == "ready":
                try:
                    size_row = session.execute(
                        text(f"SELECT pg_relation_size('{idx.index_name}') AS bytes, pg_size_pretty(pg_relation_size('{idx.index_name}')) AS pretty")
                    ).fetchone()
                    if size_row and size_row.bytes is not None:
                        idx.index_size_bytes = int(size_row.bytes)
                        size_pretty = str(size_row.pretty)
                except Exception:
                    pass

            results.append(
                IndexResponse(
                    id=idx.id,
                    index_name=idx.index_name,
                    index_type=idx.index_type,
                    distance_metric=idx.distance_metric,
                    opclass=idx.opclass,
                    vector_dimension=idx.vector_dimension,
                    model_id=idx.model_id,
                    embedding_experiment_id=idx.embedding_experiment_id,
                    parameters=idx.parameters,
                    status=idx.status,
                    build_duration_ms=idx.build_duration_ms,
                    index_size_bytes=idx.index_size_bytes,
                    index_size_pretty=size_pretty,
                    error_message=idx.error_message,
                    created_at=idx.created_at.isoformat(),
                )
            )
        return results

    def drop_index(self, session: Session, index_id: uuid.UUID) -> bool:
        """Safely drops the index in PostgreSQL and deletes its tracking record."""
        is_postgres = session.bind.dialect.name == "postgresql" if session.bind else False
        idx = session.query(RetrievalIndex).filter(RetrievalIndex.id == index_id).first()
        if not idx:
            return False

        if is_postgres:
            try:
                session.execute(text(f"DROP INDEX IF EXISTS {idx.index_name}"))
                session.commit()
            except Exception as e:
                session.rollback()
                raise RuntimeError(f"Failed to drop PostgreSQL index {idx.index_name}: {e}")

        session.delete(idx)
        session.commit()
        return True


index_manager = IndexManager()
