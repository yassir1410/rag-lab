import math
import uuid
import pytest
from app.rag.embeddings.base import DeterministicMockEmbeddingProvider
from app.rag.embeddings.similarity import (
    cosine_similarity,
    dot_product,
    euclidean_distance,
    pairwise_similarity_matrix,
)
from app.rag.embeddings.normalization import compute_l2_norm, l2_normalize, inspect_normalization
from app.rag.embeddings.tokenization import TokenizerInspector
from app.rag.embeddings.dimensionality import DimensionalityReducer
from app.rag.embeddings.evaluation import EmbeddingBenchmarkSuite
from app.rag.embeddings.registry import embedding_registry
from app.rag.embeddings.service import EmbeddingService
from app.models.document import Document, DocumentSection
from app.models.chunking import ChunkingExperiment, DocumentChunk
from app.models.embeddings import EmbeddingExperiment, ChunkEmbedding
from app.schemas.embeddings import EmbeddingExperimentCreate, ProjectEmbeddingsRequest


# --- Mathematical Utilities Tests ---

def test_cosine_similarity_known_vectors():
    # Identical vectors -> 1.0
    assert pytest.approx(cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]), 1e-5) == 1.0
    # Orthogonal vectors -> 0.0
    assert pytest.approx(cosine_similarity([1.0, 0.0], [0.0, 1.0]), 1e-5) == 0.0
    # Opposite vectors -> -1.0
    assert pytest.approx(cosine_similarity([1.0, 0.0], [-1.0, 0.0]), 1e-5) == -1.0


def test_dot_product_and_euclidean():
    u = [1.0, 2.0, 3.0]
    v = [4.0, 5.0, 6.0]
    # 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    assert dot_product(u, v) == 32.0

    # 3-4-5 right triangle distance
    p1 = [0.0, 0.0]
    p2 = [3.0, 4.0]
    assert pytest.approx(euclidean_distance(p1, p2), 1e-5) == 5.0


def test_zero_vector_and_dimension_mismatch():
    zero_vec = [0.0, 0.0, 0.0]
    unit_vec = [1.0, 0.0, 0.0]
    # Cosine similarity with zero vector returns 0.0 without crash
    assert cosine_similarity(zero_vec, unit_vec) == 0.0

    # Dimension mismatch raises ValueError
    with pytest.raises(ValueError, match="Dimension mismatch"):
        cosine_similarity([1.0, 2.0], [1.0, 2.0, 3.0])


def test_l2_normalization():
    vec = [3.0, 4.0]
    assert pytest.approx(compute_l2_norm(vec), 1e-5) == 5.0

    norm_vec = l2_normalize(vec)
    assert pytest.approx(compute_l2_norm(norm_vec), 1e-5) == 1.0
    assert pytest.approx(norm_vec[0], 1e-5) == 0.6
    assert pytest.approx(norm_vec[1], 1e-5) == 0.8

    # When normalized, dot product equals cosine similarity
    u = [10.0, 20.0, 30.0]
    v = [5.0, 1.0, 4.0]
    u_norm = l2_normalize(u)
    v_norm = l2_normalize(v)
    assert pytest.approx(dot_product(u_norm, v_norm), 1e-5) == cosine_similarity(u, v)


def test_pairwise_similarity_matrix():
    vectors = [
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
    ]
    mat = pairwise_similarity_matrix(vectors, metric="cosine")
    assert len(mat) == 3
    assert len(mat[0]) == 3
    # Diagonal is 1.0
    assert mat[0][0] == 1.0
    assert mat[1][1] == 1.0
    # Orthogonal
    assert mat[0][1] == 0.0
    assert mat[1][0] == 0.0


# --- Tokenization Inspector Tests ---

def test_tokenization_inspector():
    inspector = TokenizerInspector(model_id="sentence-transformers/all-MiniLM-L6-v2", max_seq_length=10)
    text = "FastAPI with dense vector embeddings"
    res = inspector.inspect(text)

    assert res["char_count"] == len(text)
    assert res["word_count"] == 5
    assert res["token_count"] > 0
    assert len(res["tokens"]) == len(res["token_ids"])

    # Test truncation warning on long input exceeding limit
    long_text = "word " * 20
    res_long = inspector.inspect(long_text)
    assert res_long["is_truncated"] is True
    assert res_long["truncated_tokens_count"] > 0
    assert len(res_long["warnings"]) > 0


# --- Dimensionality Reduction Tests ---

def test_dimensionality_reduction_pca_and_tsne():
    reducer = DimensionalityReducer()
    mock_vectors = [
        [1.0, 0.5, 0.2, 0.1],
        [0.9, 0.4, 0.1, 0.0],
        [0.1, 0.8, 0.9, 0.7],
        [0.2, 0.7, 0.8, 0.6],
        [0.5, 0.5, 0.5, 0.5],
    ]

    # PCA 2D
    pca_res = reducer.project(mock_vectors, algorithm="pca", n_components=2)
    assert pca_res["algorithm"] == "pca"
    assert len(pca_res["points"]) == 5
    assert len(pca_res["points"][0]) == 2
    assert "explained_variance_ratio" in pca_res

    # t-SNE with clamped perplexity
    tsne_res = reducer.project(mock_vectors, algorithm="tsne", n_components=2, perplexity=30)
    assert tsne_res["algorithm"] == "tsne"
    assert len(tsne_res["points"]) == 5


# --- Benchmark Suite Tests ---

def test_embedding_benchmark_suite():
    mock_provider = DeterministicMockEmbeddingProvider(dimension=64)
    suite = EmbeddingBenchmarkSuite(mock_provider)
    res = suite.run_benchmark()

    assert res["dimension"] == 64
    assert res["total_benchmark_pairs"] > 0
    assert "quality_metrics" in res
    assert "performance_metrics" in res
    assert "storage_footprint" in res
    assert res["storage_footprint"]["bytes_per_vector"] == 64 * 4


# --- Service and DB Persistence Tests ---

def test_embedding_service_experiment_persistence(db_session):
    # Setup test document and chunks
    doc = Document(
        original_filename="test_emb.txt",
        mime_type="text/plain",
        file_size=100,
        sha256_hash="hash_test_emb_001",
        storage_path="/tmp/test_emb.txt",
        extraction_method="txt",
        status="processed",
    )
    db_session.add(doc)
    db_session.commit()

    chunk_exp = ChunkingExperiment(
        document_id=doc.id,
        strategy="recursive",
        config={"chunk_size": 200},
        metrics={"total_chunks": 2},
    )
    db_session.add(chunk_exp)
    db_session.commit()

    c1 = DocumentChunk(
        experiment_id=chunk_exp.id,
        document_id=doc.id,
        sequence_index=0,
        content="Retrieval Augmented Generation with vector databases.",
        token_count=7,
        char_count=52,
        metadata_json={},
    )
    c2 = DocumentChunk(
        experiment_id=chunk_exp.id,
        document_id=doc.id,
        sequence_index=1,
        content="Dense embeddings project concepts into geometric space.",
        token_count=7,
        char_count=53,
        metadata_json={},
    )
    db_session.add_all([c1, c2])
    db_session.commit()

    service = EmbeddingService()
    # Use deterministic mock model for fast DB test
    req = EmbeddingExperimentCreate(
        name="Test Experiment",
        chunking_experiment_id=chunk_exp.id,
        model_id="mock-deterministic-v1",
        batch_size=16,
        normalize=True,
    )

    exp = service.create_experiment(db_session, req)
    assert exp.id is not None
    assert exp.document_id == doc.id
    assert exp.vector_dimension == 384
    assert exp.metrics["total_items"] == 2

    # Verify ChunkEmbedding rows
    embeddings = db_session.query(ChunkEmbedding).filter_by(experiment_id=exp.id).all()
    assert len(embeddings) == 2
    assert embeddings[0].vector is not None

    # Verify get_experiment
    details = service.get_experiment(db_session, exp.id)
    assert details["embeddings_count"] == 2
    assert len(details["sample_embeddings"]) == 2

    # Verify cascade delete
    deleted = service.delete_experiment(db_session, exp.id)
    assert deleted is True
    assert db_session.get(EmbeddingExperiment, exp.id) is None
    assert db_session.query(ChunkEmbedding).filter_by(experiment_id=exp.id).count() == 0


# --- API Endpoint Integration Tests ---

def test_embeddings_api_flow(client):
    # 1. Models list
    res = client.get("/api/v1/embeddings/models")
    assert res.status_code == 200
    models = res.json()
    assert len(models) >= 3
    model_ids = [m["model_id"] for m in models]
    assert "sentence-transformers/all-MiniLM-L6-v2" in model_ids
    assert "mock-deterministic-v1" in model_ids

    # 2. Tokenize endpoint
    tok_res = client.post(
        "/api/v1/embeddings/tokenize",
        json={"text": "Test embedding tokenization", "model_id": "mock-deterministic-v1"},
    )
    assert tok_res.status_code == 200
    tok_data = tok_res.json()
    assert tok_data["token_count"] > 0
    assert "tokens" in tok_data

    # 3. Similarity endpoint
    sim_res = client.post(
        "/api/v1/embeddings/similarity",
        json={
            "text_a": "FastAPI builds async Python APIs",
            "text_b": "Fast asynchronous REST services with Python",
            "model_id": "mock-deterministic-v1",
            "normalize": True,
        },
    )
    assert sim_res.status_code == 200
    sim_data = sim_res.json()
    assert "cosine_similarity" in sim_data
    assert "euclidean_distance" in sim_data
    assert "dot_product" in sim_data

    # 4. Matrix endpoint
    mat_res = client.post(
        "/api/v1/embeddings/matrix",
        json={
            "texts": [
                "Machine learning models",
                "Deep neural networks",
                "French bread and pastries",
            ],
            "model_id": "mock-deterministic-v1",
            "metric": "cosine",
            "normalize": True,
        },
    )
    assert mat_res.status_code == 200
    mat_data = mat_res.json()
    assert len(mat_data["matrix"]) == 3
    assert len(mat_data["matrix"][0]) == 3

    # 5. Benchmark endpoint
    bench_res = client.post(
        "/api/v1/embeddings/benchmark",
        json={"model_id": "mock-deterministic-v1"},
    )
    assert bench_res.status_code == 200
    bench_data = bench_res.json()
    assert "quality_metrics" in bench_data
    assert "performance_metrics" in bench_data
