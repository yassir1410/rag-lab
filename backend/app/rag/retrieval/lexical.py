import time
import re
import uuid
from typing import List, Dict, Any, Optional, Tuple
from sqlalchemy import text, desc
from sqlalchemy.orm import Session

from app.models.chunking import DocumentChunk
from app.models.document import Document
from app.rag.retrieval.schemas import SearchResultItem, RetrievalFilter


def sanitize_tsquery_term(term: str) -> str:
    """Removes special tsquery operator characters to prevent syntax errors."""
    return re.sub(r"[&|!*():<>\\]", " ", term).strip()


class LexicalSearcher:
    """
    PostgreSQL Full-Text Search (FTS) engine utilizing GIN indexes,
    tsvector generated columns, and ts_rank_cd relevance scoring.
    Features graceful fallback to token-overlap scoring on SQLite for offline unit tests.
    """

    def __init__(self):
        pass

    def search(
        self,
        session: Session,
        query: str,
        top_k: int = 5,
        filters: Optional[RetrievalFilter] = None,
        language: str = "english",
    ) -> Tuple[List[SearchResultItem], float]:
        """
        Executes lexical full-text search against document_chunks.
        Returns: (results, latency_ms)
        """
        cleaned_query = query.strip()
        if not cleaned_query:
            return [], 0.0

        t0 = time.perf_counter()
        dialect_name = session.bind.dialect.name if session.bind else "postgresql"

        if dialect_name == "postgresql":
            results = self._search_postgres(
                session=session,
                query=cleaned_query,
                top_k=top_k,
                filters=filters,
                language=language,
            )
        else:
            results = self._search_sqlite_fallback(
                session=session,
                query=cleaned_query,
                top_k=top_k,
                filters=filters,
            )

        latency_ms = (time.perf_counter() - t0) * 1000.0
        return results, latency_ms

    def _search_postgres(
        self,
        session: Session,
        query: str,
        top_k: int,
        filters: Optional[RetrievalFilter],
        language: str,
    ) -> List[SearchResultItem]:
        # Validate language config to prevent injection
        valid_languages = {"english", "french", "simple", "german", "spanish"}
        lang_config = language.lower() if language.lower() in valid_languages else "english"

        # Build filter clauses
        where_clauses = [
            f"c.tsv_content @@ plainto_tsquery('{lang_config}', :query_text)"
        ]
        params: Dict[str, Any] = {
            "query_text": query,
            "limit_val": top_k,
        }

        if filters:
            if filters.document_id:
                where_clauses.append("c.document_id = :doc_id")
                params["doc_id"] = str(filters.document_id)
            if filters.chunking_experiment_id:
                where_clauses.append("c.experiment_id = :exp_id")
                params["exp_id"] = str(filters.chunking_experiment_id)
            if filters.page_number is not None:
                where_clauses.append("(c.metadata->>'page_number')::int = :page_num")
                params["page_num"] = filters.page_number
            if filters.mime_type:
                where_clauses.append("d.mime_type = :mime_type")
                params["mime_type"] = filters.mime_type

        where_sql = " AND ".join(where_clauses)

        sql = f"""
            SELECT
                c.id as chunk_id,
                c.document_id,
                d.original_filename as document_name,
                d.mime_type,
                c.sequence_index,
                c.token_count,
                c.char_count,
                c.content,
                c.metadata as chunk_metadata,
                ts_rank_cd(c.tsv_content, plainto_tsquery('{lang_config}', :query_text), 32) as rank_score
            FROM document_chunks c
            JOIN documents d ON c.document_id = d.id
            WHERE {where_sql}
            ORDER BY rank_score DESC, c.sequence_index ASC
            LIMIT :limit_val;
        """

        rows = session.execute(text(sql), params).fetchall()

        # If strict tsquery matched 0 rows, attempt a websearch_to_tsquery fallback
        if not rows:
            fallback_sql = f"""
                SELECT
                    c.id as chunk_id,
                    c.document_id,
                    d.original_filename as document_name,
                    d.mime_type,
                    c.sequence_index,
                    c.token_count,
                    c.char_count,
                    c.content,
                    c.metadata as chunk_metadata,
                    ts_rank_cd(c.tsv_content, websearch_to_tsquery('{lang_config}', :query_text), 32) as rank_score
                FROM document_chunks c
                JOIN documents d ON c.document_id = d.id
                WHERE c.tsv_content @@ websearch_to_tsquery('{lang_config}', :query_text)
                ORDER BY rank_score DESC, c.sequence_index ASC
                LIMIT :limit_val;
            """
            rows = session.execute(text(fallback_sql), params).fetchall()

        results: List[SearchResultItem] = []
        for r in rows:
            meta = r.chunk_metadata or {}
            page_num = meta.get("page_number")
            score = float(r.rank_score) if r.rank_score is not None else 0.0
            # Normalize raw_distance for ranking consistency
            raw_dist = 1.0 / (1.0 + score)

            results.append(
                SearchResultItem(
                    chunk_id=r.chunk_id if isinstance(r.chunk_id, uuid.UUID) else uuid.UUID(str(r.chunk_id)),
                    document_id=r.document_id if isinstance(r.document_id, uuid.UUID) else uuid.UUID(str(r.document_id)),
                    document_name=r.document_name or "Unknown Document",
                    mime_type=r.mime_type or "text/plain",
                    sequence_index=r.sequence_index,
                    token_count=r.token_count,
                    char_count=r.char_count,
                    content=r.content,
                    metadata=meta,
                    page_number=page_num,
                    raw_distance=raw_dist,
                    similarity_score=score,
                    distance_metric="bm25_tsrank",
                    distance_label=f"ts_rank_cd: {score:.4f}",
                    similarity_label=f"Rank: {score:.4f}",
                )
            )

        return results

    def _search_sqlite_fallback(
        self,
        session: Session,
        query: str,
        top_k: int,
        filters: Optional[RetrievalFilter],
    ) -> List[SearchResultItem]:
        """
        In-memory fallback scoring for SQLite unit tests using token-overlap and term frequency.
        """
        tokens = [t.lower() for t in re.findall(r"\b\w{3,}\b", query)]
        if not tokens:
            return []

        q = (
            session.query(DocumentChunk, Document)
            .join(Document, DocumentChunk.document_id == Document.id)
        )

        if filters:
            if filters.document_id:
                q = q.filter(DocumentChunk.document_id == filters.document_id)
            if filters.chunking_experiment_id:
                q = q.filter(DocumentChunk.experiment_id == filters.chunking_experiment_id)

        all_chunks = q.all()
        scored_items = []

        for chunk, doc in all_chunks:
            content_lower = chunk.content.lower()
            matches = sum(content_lower.count(tok) for tok in tokens)
            if matches > 0:
                # Simple BM25-like normalized frequency score
                score = matches / (1.0 + len(chunk.content) / 200.0)
                scored_items.append((score, chunk, doc))

        scored_items.sort(key=lambda x: x[0], reverse=True)
        top_scored = scored_items[:top_k]

        results = []
        for score, chunk, doc in top_scored:
            meta = chunk.metadata_json or {}
            raw_dist = 1.0 / (1.0 + score)
            results.append(
                SearchResultItem(
                    chunk_id=chunk.id,
                    document_id=doc.id,
                    document_name=doc.original_filename,
                    mime_type=doc.mime_type or "text/plain",
                    sequence_index=chunk.sequence_index,
                    token_count=chunk.token_count,
                    char_count=chunk.char_count,
                    content=chunk.content,
                    metadata=meta,
                    page_number=meta.get("page_number"),
                    raw_distance=raw_dist,
                    similarity_score=score,
                    distance_metric="bm25_tsrank",
                    distance_label=f"Score: {score:.4f}",
                    similarity_label=f"Overlap: {score:.4f}",
                )
            )

        return results


lexical_searcher = LexicalSearcher()
