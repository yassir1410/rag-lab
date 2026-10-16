import uuid
import pytest
from app.rag.retrieval.schemas import SearchResultItem
from app.rag.retrieval.fusion import ReciprocalRankFusion, min_max_normalize
from app.rag.retrieval.reranker import NoOpReranker, MockReranker, reranker_registry


def _make_dummy_item(content: str, score: float, chunk_id: uuid.UUID = None) -> SearchResultItem:
    cid = chunk_id or uuid.uuid4()
    return SearchResultItem(
        chunk_id=cid,
        document_id=uuid.uuid4(),
        document_name="test.txt",
        mime_type="text/plain",
        sequence_index=0,
        token_count=10,
        char_count=50,
        content=content,
        raw_distance=1.0 - score,
        similarity_score=score,
        distance_metric="cosine",
        distance_label=f"dist: {1.0 - score:.3f}",
        similarity_label=f"sim: {score:.3f}",
    )


def test_min_max_normalize():
    scores = [10.0, 20.0, 30.0]
    norm = min_max_normalize(scores)
    assert norm[0] == 0.0
    assert norm[1] == 0.5
    assert norm[2] == 1.0

    # Constant scores
    assert min_max_normalize([5.0, 5.0]) == [1.0, 1.0]
    # Empty
    assert min_max_normalize([]) == []


def test_rrf_scoring_and_ranking():
    fusion = ReciprocalRankFusion(default_k=60)

    c1 = uuid.uuid4()
    c2 = uuid.uuid4()
    c3 = uuid.uuid4()

    # Dense: c1 (rank 1), c2 (rank 2)
    dense_items = [
        _make_dummy_item("Dense result 1", 0.95, c1),
        _make_dummy_item("Dense result 2", 0.85, c2),
    ]

    # Lexical: c2 (rank 1), c3 (rank 2)
    lexical_items = [
        _make_dummy_item("Lexical result 2", 0.70, c2),
        _make_dummy_item("Lexical result 3", 0.60, c3),
    ]

    # Expected RRF scores with k=60:
    # c2: rank 2 in dense + rank 1 in lexical => 1/(60+2) + 1/(60+1) = 1/62 + 1/61 = 0.016129 + 0.016393 = ~0.03252
    # c1: rank 1 in dense => 1/(60+1) = ~0.01639
    # c3: rank 2 in lexical => 1/(60+2) = ~0.01613
    # c2 should win top rank because it was retrieved by BOTH systems!
    fused, audits = fusion.fuse_rrf(dense_items, lexical_items, k=60, top_k=3)

    assert len(fused) == 3
    assert fused[0].chunk_id == c2
    assert fused[1].chunk_id == c1
    assert fused[2].chunk_id == c3

    # Check candidate audit metadata
    c2_audit = next(a for a in audits if a.chunk_id == c2)
    assert c2_audit.dense_rank == 2
    assert c2_audit.lexical_rank == 1
    assert c2_audit.final_rank == 1


def test_linear_fusion():
    fusion = ReciprocalRankFusion()

    c1 = uuid.uuid4()
    c2 = uuid.uuid4()

    dense = [_make_dummy_item("Item 1", 1.0, c1), _make_dummy_item("Item 2", 0.0, c2)]
    lexical = [_make_dummy_item("Item 1", 0.0, c1), _make_dummy_item("Item 2", 1.0, c2)]

    # Alpha 0.8 favors dense (c1)
    blended_dense, _ = fusion.fuse_linear(dense, lexical, alpha=0.8, top_k=2)
    assert blended_dense[0].chunk_id == c1

    # Alpha 0.2 favors lexical (c2)
    blended_lex, _ = fusion.fuse_linear(dense, lexical, alpha=0.2, top_k=2)
    assert blended_lex[0].chunk_id == c2


def test_rerankers():
    items = [
        _make_dummy_item("The weather today is sunny and pleasant.", 0.9),
        _make_dummy_item("PostgreSQL pgvector provides high-performance HNSW vector indexing.", 0.8),
    ]
    query = "PostgreSQL pgvector index"

    # 1. NoOp Reranker preserves order
    noop = NoOpReranker()
    r_noop, lat, _ = noop.rerank(query, items, top_k=2)
    assert r_noop[0].content == items[0].content
    assert lat == 0.0

    # 2. Mock Reranker boosts keyword-matched item to rank 1
    mock = MockReranker()
    r_mock, lat_mock, scores = mock.rerank(query, items, top_k=2)
    assert r_mock[0].content == items[1].content
    assert lat_mock >= 0
    assert len(scores) == 2
    assert scores[0] > scores[1]

    # 3. Registry access
    reg_tiny = reranker_registry.get_reranker("tinybert")
    assert reg_tiny is not None
    reg_none = reranker_registry.get_reranker("none")
    assert isinstance(reg_none, NoOpReranker)
