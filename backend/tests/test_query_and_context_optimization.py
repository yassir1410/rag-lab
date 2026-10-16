import uuid
import pytest
from app.models.document import Document
from app.models.chunking import DocumentChunk
from app.rag.retrieval.schemas import SearchResultItem
from app.rag.retrieval.query_rewriter import query_processor
from app.rag.retrieval.context_optimizer import context_optimizer, jaccard_similarity
from app.evaluation.generation.claim_extractor import claim_extractor


def test_query_processor_keywords_and_rewriting():
    q = "Explain how PostgreSQL pgvector implements HNSW for fast RAG search"
    keywords = query_processor.extract_keywords(q)
    assert "PostgreSQL" in keywords
    assert "pgvector" in keywords
    assert "HNSW" in keywords
    assert "how" not in keywords  # stopword

    rewritten = query_processor.rewrite_query("fast RAG with HNSW and IVFFlat")
    assert "Retrieval-Augmented Generation" in rewritten or "HNSW" in rewritten

    expanded = query_processor.expand_multi_query(q, max_queries=3)
    assert len(expanded) >= 2
    assert q in expanded


def test_context_optimizer_redundancy_and_cutoff():
    # 1. Jaccard similarity
    t1 = "PostgreSQL pgvector provides HNSW index for vector similarity search."
    t2 = "PostgreSQL pgvector provides HNSW index for vector similarity search."
    t3 = "Docker containers package software into standardized units."
    assert jaccard_similarity(t1, t2) == 1.0
    assert jaccard_similarity(t1, t3) < 0.2

    # 2. Prune redundancy
    doc_id = uuid.uuid4()
    item1 = SearchResultItem(
        chunk_id=uuid.uuid4(),
        document_id=doc_id,
        document_name="test.txt",
        mime_type="text/plain",
        sequence_index=1,
        token_count=10,
        char_count=50,
        content=t1,
        raw_distance=0.1,
        similarity_score=0.9,
        distance_metric="cosine",
        distance_label="0.1",
        similarity_label="0.9",
    )
    item2 = SearchResultItem(
        chunk_id=uuid.uuid4(),
        document_id=doc_id,
        document_name="test.txt",
        mime_type="text/plain",
        sequence_index=2,
        token_count=10,
        char_count=50,
        content=t2,  # identical content
        raw_distance=0.11,
        similarity_score=0.89,
        distance_metric="cosine",
        distance_label="0.11",
        similarity_label="0.89",
    )
    item3 = SearchResultItem(
        chunk_id=uuid.uuid4(),
        document_id=doc_id,
        document_name="test.txt",
        mime_type="text/plain",
        sequence_index=3,
        token_count=10,
        char_count=50,
        content=t3,
        raw_distance=0.4,
        similarity_score=0.6,
        distance_metric="cosine",
        distance_label="0.4",
        similarity_label="0.6",
    )

    pruned = context_optimizer.prune_redundancy([item1, item2, item3])
    assert len(pruned) == 2
    assert pruned[0].chunk_id == item1.chunk_id
    assert pruned[1].chunk_id == item3.chunk_id

    # 3. Adaptive cutoff
    cutoff_items = context_optimizer.apply_adaptive_cutoff([item1, item3], relative_threshold=0.8, min_k=1)
    # item3 score (0.6) / item1 score (0.9) = 0.667 < 0.8 => should be cutoff
    assert len(cutoff_items) == 1
    assert cutoff_items[0].chunk_id == item1.chunk_id


def test_neighbor_expansion(db_session):
    doc = Document(
        original_filename="notes.txt",
        storage_path="/tmp/notes.txt",
        file_size=200,
        sha256_hash="hash_notes",
        mime_type="text/plain",
        extraction_method="direct_text",
    )
    db_session.add(doc)
    db_session.flush()

    c0 = DocumentChunk(document_id=doc.id, sequence_index=0, content="Introduction to distributed algorithms.", token_count=5, char_count=40)
    c1 = DocumentChunk(document_id=doc.id, sequence_index=1, content="Paxos and Raft consensus protocols.", token_count=5, char_count=35)
    c2 = DocumentChunk(document_id=doc.id, sequence_index=2, content="Byzantine fault tolerance and state replication.", token_count=6, char_count=45)
    db_session.add_all([c0, c1, c2])
    db_session.commit()

    target_item = SearchResultItem(
        chunk_id=c1.id,
        document_id=doc.id,
        document_name=doc.original_filename,
        mime_type="text/plain",
        sequence_index=1,
        token_count=5,
        char_count=35,
        content=c1.content,
        raw_distance=0.1,
        similarity_score=0.9,
        distance_metric="cosine",
        distance_label="0.1",
        similarity_label="0.9",
    )

    expanded = context_optimizer.expand_neighboring_chunks(db_session, [target_item], max_expansion=1)
    expanded_ids = {item.chunk_id for item in expanded}
    assert c1.id in expanded_ids
    assert c0.id in expanded_ids or c2.id in expanded_ids


def test_claim_extractor_boilerplate_stripping():
    text = (
        "Based on the provided documents, candidate Yassir has 5 years of Python experience [S1]. "
        "According to the context, he deployed microservices using Docker and Kubernetes [S2]. "
        "I hope this helps!"
    )
    claims = claim_extractor.extract_claims(text)
    assert len(claims) == 2  # Conversational filler "I hope this helps!" filtered out
    # Boilerplate stripped
    assert not claims[0]["claim"].lower().startswith("based on the provided documents")
    assert not claims[1]["claim"].lower().startswith("according to the context")
    assert "[S1]" in claims[0]["citations"]
    assert "[S2]" in claims[1]["citations"]
