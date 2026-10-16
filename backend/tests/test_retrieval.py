import uuid
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.models.document import Document
from app.models.chunking import ChunkingExperiment, DocumentChunk
from app.models.embeddings import EmbeddingExperiment, ChunkEmbedding
from app.models.retrieval import RetrievalIndex, RetrievalExperiment
from app.rag.embeddings.base import DeterministicMockEmbeddingProvider
from app.rag.embeddings.registry import embedding_registry, ModelSpecification
from app.rag.retrieval.base import compute_similarity_score, parse_execution_plan
from app.rag.retrieval.exact import ExactVectorSearcher
from app.rag.retrieval.hnsw import HNSWVectorSearcher
from app.rag.retrieval.ivfflat import IVFFlatVectorSearcher
from app.rag.retrieval.index_manager import IndexManager, format_bytes
from app.rag.retrieval.benchmark import RetrievalBenchmarker
from app.rag.retrieval.synthetic import generate_synthetic_benchmark_corpus
from app.rag.retrieval.schemas import (
    SearchRequest,
    RetrievalFilter,
    IndexCreateRequest,
    BenchmarkRequest,
    RetrievalExperimentCreate,
)


@pytest.fixture(autouse=True)
def setup_mock_model():
    """Ensure deterministic mock model is available in the registry for fast testing."""
    if "mock-retrieval-v1" not in embedding_registry._specs:
        embedding_registry.register(
            ModelSpecification(
                model_id="mock-retrieval-v1",
                display_name="Mock Retrieval Model",
                provider="mock",
                dimension=4,
                max_seq_length=128,
                language="English",
                description="Mock 4D model for testing",
                query_prefix="query: ",
                passage_prefix="passage: ",
                approx_size_mb=0.0,
            )
        )


@pytest.fixture
def sample_data(db_session: Session):
    """Seed documents, chunks, and embeddings for testing."""
    doc = Document(
        original_filename="ai_systems.pdf",
        mime_type="application/pdf",
        file_size=2048,
        sha256_hash="hash_ai_systems_12345",
        storage_path="/tmp/ai_systems.pdf",
        extraction_method="pymupdf",
    )
    db_session.add(doc)
    db_session.commit()

    chunk_exp = ChunkingExperiment(
        document_id=doc.id,
        strategy="recursive",
        config={"chunk_size": 200},
        metrics={"chunk_count": 3},
    )
    db_session.add(chunk_exp)
    db_session.commit()

    chunks = [
        DocumentChunk(
            experiment_id=chunk_exp.id,
            document_id=doc.id,
            sequence_index=0,
            content="Vector search enables high-dimensional nearest neighbor retrieval.",
            token_count=10,
            char_count=65,
            metadata_json={"page_number": 1},
        ),
        DocumentChunk(
            experiment_id=chunk_exp.id,
            document_id=doc.id,
            sequence_index=1,
            content="HNSW builds multi-layer proximity graphs with logarithmic search complexity.",
            token_count=10,
            char_count=75,
            metadata_json={"page_number": 2},
        ),
        DocumentChunk(
            experiment_id=chunk_exp.id,
            document_id=doc.id,
            sequence_index=2,
            content="IVFFlat uses inverted file clustering with coarse Voronoi centroids.",
            token_count=10,
            char_count=68,
            metadata_json={"page_number": 2},
        ),
    ]
    for c in chunks:
        db_session.add(c)
    db_session.commit()

    embed_exp = EmbeddingExperiment(
        name="Test 4D Embeddings",
        document_id=doc.id,
        chunking_experiment_id=chunk_exp.id,
        model_id="mock-retrieval-v1",
        provider="mock",
        vector_dimension=4,
        is_normalized=True,
    )
    db_session.add(embed_exp)
    db_session.commit()

    # Pre-calculated 4D unit vectors for predictable mathematical similarity
    vectors = [
        [1.0, 0.0, 0.0, 0.0],  # chunk 0: aligned with query [1, 0, 0, 0]
        [0.0, 1.0, 0.0, 0.0],  # chunk 1: orthogonal to query
        [0.7071, 0.7071, 0.0, 0.0],  # chunk 2: 45 degrees to query
    ]

    chunk_embeddings = []
    for i, (c, v) in enumerate(zip(chunks, vectors)):
        ce = ChunkEmbedding(
            experiment_id=embed_exp.id,
            chunk_id=c.id,
            sequence_index=i,
            text_snippet=c.content,
            token_count=c.token_count,
            vector=v,
            vector_dimension=4,
        )
        chunk_embeddings.append(ce)
        db_session.add(ce)
    db_session.commit()

    return {
        "doc": doc,
        "chunk_exp": chunk_exp,
        "embed_exp": embed_exp,
        "chunks": chunks,
    }


# --- Unit Tests: Mathematical Similarity Mappings ---

def test_compute_similarity_score_cosine():
    # Distance = 0.0 -> Similarity = 1.0
    sim, dist_lbl, sim_lbl = compute_similarity_score(0.0, "cosine")
    assert sim == 1.0
    assert "Cosine Distance" in dist_lbl
    assert "Cosine Similarity" in sim_lbl

    # Distance = 1.0 (orthogonal) -> Similarity = 0.0
    sim, _, _ = compute_similarity_score(1.0, "cosine")
    assert sim == 0.0

    # Distance = 2.0 (opposite) -> Similarity = -1.0
    sim, _, _ = compute_similarity_score(2.0, "cosine")
    assert sim == -1.0


def test_compute_similarity_score_l2():
    # L2 distance = 0.0 -> Harmonic similarity = 1.0
    sim, dist_lbl, sim_lbl = compute_similarity_score(0.0, "l2")
    assert sim == 1.0
    assert "L2 Distance" in dist_lbl
    assert "Harmonic L2 Similarity" in sim_lbl

    # L2 distance = 1.0 -> Similarity = 0.5
    sim, _, _ = compute_similarity_score(1.0, "l2")
    assert sim == 0.5


def test_compute_similarity_score_inner_product():
    # pgvector returns -(u · v). For u · v = 0.85, raw_distance is -0.85
    sim, dist_lbl, sim_lbl = compute_similarity_score(-0.85, "inner_product")
    assert sim == 0.85
    assert "Negative Dot Product" in dist_lbl
    assert "Inner Product" in sim_lbl


# --- Unit Tests: Vector Searchers & Ordering ---

def test_exact_vector_searcher_ordering(db_session: Session, sample_data):
    searcher = ExactVectorSearcher()
    query_vec = [1.0, 0.0, 0.0, 0.0]

    results, plan, lat = searcher.search(
        session=db_session,
        query_vector=query_vec,
        dimension=4,
        model_id="mock-retrieval-v1",
        metric="cosine",
        top_k=3,
    )

    assert len(results) == 3
    # First result must be chunk 0 (distance = 0.0, sim = 1.0)
    assert results[0].sequence_index == 0
    assert pytest.approx(results[0].raw_distance, 1e-4) == 0.0
    assert results[0].similarity_score == 1.0

    # Second result must be chunk 2 (45 degrees, raw distance ~ 0.2929)
    assert results[1].sequence_index == 2
    assert results[1].similarity_score > results[2].similarity_score

    # Third result must be chunk 1 (orthogonal, raw distance = 1.0, sim = 0.0)
    assert results[2].sequence_index == 1
    assert pytest.approx(results[2].similarity_score, 1e-4) == 0.0


def test_metadata_filtering(db_session: Session, sample_data):
    searcher = ExactVectorSearcher()
    query_vec = [1.0, 0.0, 0.0, 0.0]

    # Filter by page 2: chunk 0 is on page 1, so it must be excluded!
    filters = RetrievalFilter(page_number=2)
    results, plan, lat = searcher.search(
        session=db_session,
        query_vector=query_vec,
        dimension=4,
        model_id="mock-retrieval-v1",
        metric="cosine",
        top_k=5,
        filters=filters,
    )

    assert len(results) == 2
    for r in results:
        assert r.page_number == 2


def test_model_isolation(db_session: Session, sample_data):
    searcher = ExactVectorSearcher()
    query_vec = [1.0, 0.0, 0.0, 0.0]

    # Querying a model that has no embeddings must return empty list without crashing
    results, plan, lat = searcher.search(
        session=db_session,
        query_vector=query_vec,
        dimension=4,
        model_id="non-existent-model",
        metric="cosine",
        top_k=5,
    )
    assert len(results) == 0


def test_hnsw_and_ivfflat_searchers(db_session: Session, sample_data):
    hnsw = HNSWVectorSearcher()
    ivfflat = IVFFlatVectorSearcher()
    query_vec = [1.0, 0.0, 0.0, 0.0]

    # HNSW search
    h_res, h_plan, h_lat = hnsw.search(
        session=db_session,
        query_vector=query_vec,
        dimension=4,
        model_id="mock-retrieval-v1",
        metric="cosine",
        top_k=2,
        ef_search=50,
    )
    assert len(h_res) == 2
    assert h_res[0].sequence_index == 0

    # IVFFlat search with warning inspection
    ivf_res, ivf_plan, ivf_lat, ivf_warns = ivfflat.search(
        session=db_session,
        query_vector=query_vec,
        dimension=4,
        model_id="mock-retrieval-v1",
        metric="cosine",
        top_k=2,
        probes=2,
    )
    assert len(ivf_res) == 2
    assert ivf_res[0].sequence_index == 0


# --- Unit Tests: Index Manager & Parameter Validation ---

def test_index_manager_lifecycle(db_session: Session, sample_data):
    manager = IndexManager()

    # Create HNSW Index
    req = IndexCreateRequest(
        index_type="hnsw",
        distance_metric="cosine",
        model_id="mock-retrieval-v1",
        parameters={"m": 32, "ef_construction": 128},
    )
    idx_resp = manager.create_index(db_session, req, dimension=4)
    assert idx_resp.index_type == "hnsw"
    assert idx_resp.parameters["m"] == 32
    assert idx_resp.parameters["ef_construction"] == 128
    assert idx_resp.status == "ready"

    # List indexes
    all_idxs = manager.list_indexes(db_session)
    assert any(i.id == idx_resp.id for i in all_idxs)

    # Drop index
    dropped = manager.drop_index(db_session, idx_resp.id)
    assert dropped is True
    assert not any(i.id == idx_resp.id for i in manager.list_indexes(db_session))


def test_index_manager_invalid_type(db_session: Session):
    manager = IndexManager()
    req = IndexCreateRequest(
        index_type="invalid_index_type",
        distance_metric="cosine",
        model_id="mock-retrieval-v1",
    )
    with pytest.raises(ValueError, match="Unsupported index type"):
        manager.create_index(db_session, req, dimension=4)


# --- Unit Tests: Benchmarking & Synthetic Generator ---

def test_benchmarker_metrics(db_session: Session, sample_data):
    benchmarker = RetrievalBenchmarker()
    query_vec = [1.0, 0.0, 0.0, 0.0]
    req = BenchmarkRequest(
        query="What is vector search?",
        model_id="mock-retrieval-v1",
        distance_metric="cosine",
        top_k=2,
        runs=3,
    )

    resp = benchmarker.run_benchmark(db_session, query_vec, dimension=4, request=req)
    assert "exact" in resp.algorithms
    assert "hnsw" in resp.algorithms
    assert "ivfflat" in resp.algorithms

    # Exact recall is always 1.0
    assert resp.algorithms["exact"].recall_at_k == 1.0
    assert resp.algorithms["exact"].throughput_qps > 0.0


def test_synthetic_dataset_generator(db_session: Session):
    # Generate 50 synthetic vectors using mock model
    res = generate_synthetic_benchmark_corpus(
        session=db_session,
        model_id="mock-retrieval-v1",
        target_count=50,
    )
    assert res["count"] >= 50
    assert res["vector_dimension"] == 4

    # Verify rows exist in DB
    rows = (
        db_session.query(ChunkEmbedding)
        .filter(ChunkEmbedding.experiment_id == uuid.UUID(res["experiment_id"]))
        .all()
    )
    assert len(rows) >= 50


# --- FastAPI Endpoint Integration Tests ---

def test_api_search_endpoint(client: TestClient, sample_data):
    payload = {
        "query": "vector search nearest neighbors",
        "model_id": "mock-retrieval-v1",
        "search_algorithm": "exact",
        "distance_metric": "cosine",
        "top_k": 3,
        "explain_plan": True,
    }
    response = client.post("/api/v1/retrieval/search", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["query"] == payload["query"]
    assert len(data["results"]) == 3
    assert data["results"][0]["similarity_score"] >= data["results"][1]["similarity_score"]
    assert "query_latency_ms" in data
    assert "execution_plan" in data


def test_api_indexes_crud(client: TestClient, sample_data):
    # 1. Create index
    create_payload = {
        "index_type": "hnsw",
        "distance_metric": "cosine",
        "model_id": "mock-retrieval-v1",
        "parameters": {"m": 16, "ef_construction": 64},
    }
    res_create = client.post("/api/v1/retrieval/indexes", json=create_payload)
    assert res_create.status_code == 201
    created_idx = res_create.json()
    idx_id = created_idx["id"]

    # 2. List indexes
    res_list = client.get("/api/v1/retrieval/indexes")
    assert res_list.status_code == 200
    index_list = res_list.json()
    assert any(i["id"] == idx_id for i in index_list)

    # 3. Drop index
    res_del = client.delete(f"/api/v1/retrieval/indexes/{idx_id}")
    assert res_del.status_code == 204

    # 4. Verify dropped
    res_list2 = client.get("/api/v1/retrieval/indexes")
    assert not any(i["id"] == idx_id for i in res_list2.json())


def test_api_benchmark_endpoint(client: TestClient, sample_data):
    payload = {
        "query": "vector proximity graph",
        "model_id": "mock-retrieval-v1",
        "distance_metric": "cosine",
        "top_k": 2,
        "runs": 2,
    }
    response = client.post("/api/v1/retrieval/benchmark", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "algorithms" in data
    assert "exact" in data["algorithms"]
    assert "hnsw" in data["algorithms"]
    assert "ivfflat" in data["algorithms"]


def test_api_experiments_crud(client: TestClient, sample_data):
    exp_payload = {
        "name": "Validation Run 1",
        "query_text": "transformer embeddings",
        "model_id": "mock-retrieval-v1",
        "distance_metric": "cosine",
        "top_k": 3,
        "results": [{"chunk_id": str(sample_data["chunks"][0].id), "score": 0.98}],
        "metrics": {"recall_at_k": 1.0, "latency_mean_ms": 1.2},
    }
    res_save = client.post("/api/v1/retrieval/experiments", json=exp_payload)
    assert res_save.status_code == 201
    saved = res_save.json()
    exp_id = saved["id"]

    # List experiments
    res_list = client.get("/api/v1/retrieval/experiments")
    assert res_list.status_code == 200
    assert any(e["id"] == exp_id for e in res_list.json())

    # Delete experiment
    res_del = client.delete(f"/api/v1/retrieval/experiments/{exp_id}")
    assert res_del.status_code == 204


def test_api_synthetic_dataset_endpoint(client: TestClient):
    response = client.post("/api/v1/retrieval/synthetic-dataset?model_id=mock-retrieval-v1&count=60")
    assert response.status_code == 200
    data = response.json()
    assert data["count"] >= 60
    assert data["model_id"] == "mock-retrieval-v1"
