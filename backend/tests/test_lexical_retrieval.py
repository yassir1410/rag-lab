import uuid
import pytest
from app.models.document import Document
from app.models.chunking import ChunkingExperiment, DocumentChunk
from app.rag.retrieval.lexical import LexicalSearcher, sanitize_tsquery_term
from app.rag.retrieval.schemas import RetrievalFilter


def test_sanitize_tsquery_term():
    raw = "PostgreSQL & pgvector | (HNSW) ! IVFFlat*"
    cleaned = sanitize_tsquery_term(raw)
    assert "&" not in cleaned
    assert "|" not in cleaned
    assert "!" not in cleaned
    assert "*" not in cleaned
    assert "(" not in cleaned
    assert ")" not in cleaned
    assert "PostgreSQL" in cleaned
    assert "pgvector" in cleaned


def test_lexical_search_sqlite_fallback(db_session):
    searcher = LexicalSearcher()

    # 1. Create document
    doc = Document(
        original_filename="distributed_systems.txt",
        storage_path="/tmp/dist.txt",
        file_size=1024,
        sha256_hash="hash_dist_123",
        mime_type="text/plain",
        extraction_method="direct_text",
    )
    db_session.add(doc)
    db_session.flush()

    # 2. Create chunks
    c1 = DocumentChunk(
        document_id=doc.id,
        sequence_index=0,
        content="The Raft consensus algorithm ensures state machine replication across a cluster of nodes.",
        token_count=15,
        char_count=90,
    )
    c2 = DocumentChunk(
        document_id=doc.id,
        sequence_index=1,
        content="PostgreSQL pgvector implements HNSW and IVFFlat indexes for approximate nearest neighbor search.",
        token_count=14,
        char_count=95,
    )
    c3 = DocumentChunk(
        document_id=doc.id,
        sequence_index=2,
        content="Docker containers encapsulate application dependencies and guarantee execution reproducibility.",
        token_count=12,
        char_count=88,
    )
    db_session.add_all([c1, c2, c3])
    db_session.commit()

    # 3. Query for Raft consensus
    results, lat_ms = searcher.search(
        session=db_session,
        query="Raft consensus replication",
        top_k=2,
    )
    assert len(results) > 0
    assert results[0].chunk_id == c1.id
    assert results[0].similarity_score > 0
    assert results[0].distance_metric == "bm25_tsrank"
    assert lat_ms >= 0

    # 4. Query for pgvector
    results_pg, _ = searcher.search(
        session=db_session,
        query="pgvector approximate nearest neighbor",
        top_k=2,
    )
    assert len(results_pg) > 0
    assert results_pg[0].chunk_id == c2.id

    # 5. Empty query
    empty_res, _ = searcher.search(session=db_session, query="   ")
    assert empty_res == []


def test_lexical_search_filtering(db_session):
    searcher = LexicalSearcher()

    doc1 = Document(
        original_filename="doc1.txt",
        storage_path="/tmp/doc1.txt",
        file_size=100,
        sha256_hash="hash_1",
        mime_type="text/plain",
        extraction_method="direct_text",
    )
    doc2 = Document(
        original_filename="doc2.txt",
        storage_path="/tmp/doc2.txt",
        file_size=100,
        sha256_hash="hash_2",
        mime_type="text/plain",
        extraction_method="direct_text",
    )
    db_session.add_all([doc1, doc2])
    db_session.flush()

    c1 = DocumentChunk(
        document_id=doc1.id,
        sequence_index=0,
        content="Microservices communicate via gRPC and REST APIs with asynchronous message brokers.",
        token_count=12,
        char_count=80,
    )
    c2 = DocumentChunk(
        document_id=doc2.id,
        sequence_index=0,
        content="Microservices require distributed tracing and centralized log aggregation systems.",
        token_count=11,
        char_count=78,
    )
    db_session.add_all([c1, c2])
    db_session.commit()

    # Filter to doc2 only
    filt = RetrievalFilter(document_id=doc2.id)
    results, _ = searcher.search(
        session=db_session,
        query="microservices architecture",
        top_k=5,
        filters=filt,
    )
    assert len(results) == 1
    assert results[0].document_id == doc2.id
