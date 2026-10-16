import uuid
import pytest
from app.models.document import Document
from app.models.chunking import DocumentChunk
from app.models.evaluation import EvaluationDataset, EvaluationQuestion, EvaluationRun


def test_strategies_catalog_api(client):
    res = client.get("/api/v1/retrieval/strategies")
    assert res.status_code == 200
    data = res.json()
    assert len(data) == 8
    strategy_ids = {s["id"] for s in data}
    assert "dense_only" in strategy_ids
    assert "lexical_only" in strategy_ids
    assert "hybrid_rrf" in strategy_ids
    assert "hybrid_linear" in strategy_ids
    assert "dense_rerank" in strategy_ids
    assert "hybrid_rrf_rerank" in strategy_ids
    assert "multi_query_hybrid_rerank" in strategy_ids
    assert "adaptive_context" in strategy_ids


def test_lexical_and_hybrid_search_api(client, db_session):
    # 1. Create document and chunks
    doc = Document(
        original_filename="distributed_kv.txt",
        storage_path="/tmp/kv.txt",
        file_size=500,
        sha256_hash="hash_kv_123",
        mime_type="text/plain",
        extraction_method="direct_text",
    )
    db_session.add(doc)
    db_session.flush()

    c1 = DocumentChunk(
        document_id=doc.id,
        sequence_index=0,
        content="Distributed Key-Value stores leverage consistent hashing to distribute partitions evenly.",
        token_count=13,
        char_count=85,
    )
    c2 = DocumentChunk(
        document_id=doc.id,
        sequence_index=1,
        content="PostgreSQL pgvector provides HNSW index structures for approximate nearest neighbor retrieval.",
        token_count=13,
        char_count=88,
    )
    db_session.add_all([c1, c2])
    db_session.commit()

    # 2. Test /retrieval/lexical
    lex_res = client.post(
        "/api/v1/retrieval/lexical",
        json={"query": "consistent hashing partitions", "top_k": 2},
    )
    assert lex_res.status_code == 200
    lex_data = lex_res.json()
    assert lex_data["strategy"] == "lexical_only"
    assert len(lex_data["results"]) > 0
    assert lex_data["results"][0]["content"] == c1.content

    # 3. Test /retrieval/hybrid with RRF
    hybrid_res = client.post(
        "/api/v1/retrieval/hybrid",
        json={
            "query": "consistent hashing",
            "top_k": 2,
            "fusion_method": "rrf",
            "rrf_k": 60,
        },
    )
    assert hybrid_res.status_code == 200
    hybrid_data = hybrid_res.json()
    assert hybrid_data["strategy"] in ("hybrid_rrf", "hybrid_rrf_rerank")
    assert len(hybrid_data["results"]) > 0
    assert "timing_breakdown" in hybrid_data

    # 4. Test /retrieval/rerank
    candidate = {
        "chunk_id": str(c1.id),
        "document_id": str(doc.id),
        "document_name": doc.original_filename,
        "mime_type": "text/plain",
        "sequence_index": 0,
        "token_count": 13,
        "char_count": 85,
        "content": c1.content,
        "metadata": {},
        "raw_distance": 0.2,
        "similarity_score": 0.8,
        "distance_metric": "cosine",
        "distance_label": "0.2",
        "similarity_label": "0.8",
    }
    rerank_res = client.post(
        "/api/v1/retrieval/rerank",
        json={
            "query": "consistent hashing partitions",
            "reranker_model": "mock",
            "candidates": [candidate],
            "top_k": 1,
        },
    )
    assert rerank_res.status_code == 200
    rerank_data = rerank_res.json()
    assert len(rerank_data["results"]) == 1
    assert rerank_data["results"][0]["chunk_id"] == str(c1.id)


def test_question_graded_labels_and_audit_api(client, db_session):
    # 1. Create dataset
    ds = EvaluationDataset(name="Audit Test DS", description="Dataset for testing labels")
    db_session.add(ds)
    db_session.flush()

    c1 = uuid.uuid4()
    c2 = uuid.uuid4()

    q = EvaluationQuestion(
        dataset_id=ds.id,
        question="What consensus protocol is used?",
        reference_answer="Raft consensus protocol is used.",
        relevant_chunk_ids=[str(c1)],
        relevant_document_ids=[],
        category="factual",
        difficulty="easy",
        is_answerable=True,
    )
    db_session.add(q)
    db_session.commit()

    # 2. Assign graded relevance labels (c1 = 2, c2 = 1)
    label_res = client.post(
        f"/api/v1/evaluation/questions/{q.id}/labels",
        json={
            "relevance_labels": {str(c1): 2, str(c2): 1},
            "reviewer_notes": "Audited with ground-truth evidence.",
            "reviewer": "test_auditor",
        },
    )
    assert label_res.status_code == 200
    data = label_res.json()
    assert data["is_reviewed"] is True
    assert data["revision"] == 2
    assert data["relevance_labels"][str(c1)] == 2
    assert data["relevance_labels"][str(c2)] == 1
    assert len(data["audit_history"]) == 1
    assert data["audit_history"][0]["reviewer"] == "test_auditor"

    # 3. Retrieve audit trail
    audit_res = client.get(f"/api/v1/evaluation/questions/{q.id}/audit")
    assert audit_res.status_code == 200
    audit_data = audit_res.json()
    assert audit_data["revision"] == 2
    assert audit_data["is_reviewed"] is True
    assert len(audit_data["audit_history"]) == 1
