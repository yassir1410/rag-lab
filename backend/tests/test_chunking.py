import uuid
import pytest
from fastapi.testclient import TestClient

from app.rag.chunking.strategies.fixed_size import FixedSizeChunking
from app.rag.chunking.strategies.recursive import RecursiveChunking
from app.rag.chunking.strategies.sentence import SentenceChunking
from app.rag.chunking.strategies.semantic import SemanticChunking, DeterministicMockEmbeddingProvider
from app.rag.chunking.strategies.parent_child import ParentChildChunking
from app.rag.chunking.strategies.structure_aware import StructureAwareChunking
from app.rag.chunking.tokenizer import default_tokenizer
from app.rag.chunking.metrics import MetricsCalculator


def test_fixed_size_character_chunking():
    strategy = FixedSizeChunking()
    text = "abcdefghijklmnopqrstuvwxyz"
    doc_id = uuid.uuid4()
    result = strategy.split(text, doc_id, {"chunk_size": 10, "chunk_overlap": 2, "measurement_unit": "character"})

    assert len(result.chunks) == 3
    # First chunk: 0 to 10
    assert result.chunks[0].content == "abcdefghij"
    assert result.chunks[0].metadata.start_char == 0
    assert result.chunks[0].metadata.end_char == 10
    # Second chunk: 8 to 18 (with 2 overlap: 'ij')
    assert result.chunks[1].content == "ijklmnopqr"
    assert result.chunks[1].metadata.start_char == 8
    assert result.chunks[1].metadata.end_char == 18
    # Third chunk: 16 to 26 (with 2 overlap: 'qr')
    assert result.chunks[2].content == "qrstuvwxyz"
    assert result.chunks[2].metadata.start_char == 16
    assert result.chunks[2].metadata.end_char == 26
    assert result.metrics["total_chunks"] == 3


def test_fixed_size_token_chunking():
    strategy = FixedSizeChunking()
    text = "The quick brown fox jumps over the lazy dog repeatedly and smoothly."
    doc_id = uuid.uuid4()
    result = strategy.split(text, doc_id, {"chunk_size": 4, "chunk_overlap": 1, "measurement_unit": "token"})

    assert len(result.chunks) >= 2
    for c in result.chunks:
        assert c.token_count <= 4
        assert c.metadata.start_char is not None
        assert c.metadata.end_char is not None


def test_fixed_size_invalid_config():
    strategy = FixedSizeChunking()
    with pytest.raises(ValueError):
        strategy.validate_config({"chunk_size": 100, "chunk_overlap": 100})
    with pytest.raises(ValueError):
        strategy.validate_config({"chunk_size": -10, "chunk_overlap": 0})


def test_recursive_chunking_hierarchy():
    strategy = RecursiveChunking()
    text = (
        "Paragraph 1 sentence 1. Paragraph 1 sentence 2.\n\n"
        "Paragraph 2 sentence 1. Paragraph 2 sentence 2.\n\n"
        "Paragraph 3 sentence 1. Paragraph 3 sentence 2."
    )
    doc_id = uuid.uuid4()
    result = strategy.split(text, doc_id, {"chunk_size": 50, "chunk_overlap": 10})

    assert len(result.chunks) >= 3
    # Ensure all chunks fit within budget
    for c in result.chunks:
        assert c.char_count <= 60  # with separator margins


def test_sentence_chunking_bilingual():
    strategy = SentenceChunking()
    # French text with Dr., M., and Mme. abbreviations
    text = (
        "Le Dr. Martin et Mme. Dupont travaillent sur le projet RAG. "
        "Ils analysent les résultats de l'algorithme. "
        "La précision atteint 94,5% selon les métriques. "
        "C'est une avancée significative."
    )
    doc_id = uuid.uuid4()
    result = strategy.split(text, doc_id, {"chunk_size": 100, "sentence_overlap": 1})

    assert len(result.chunks) >= 2
    # Ensure "Dr. Martin et Mme. Dupont" wasn't split into two sentences on Dr. or Mme.
    first_chunk = result.chunks[0].content
    assert "Dr. Martin et Mme. Dupont" in first_chunk


def test_semantic_chunking_with_mock_provider():
    mock_provider = DeterministicMockEmbeddingProvider(dim=32)
    strategy = SemanticChunking(embedding_provider=mock_provider)
    text = (
        "Quantum computing relies on quantum bits or qubits. "
        "Superposition and entanglement are fundamental quantum properties. "
        "The French culinary tradition emphasizes seasonal vegetables and refined sauces. "
        "Gastronomy is celebrated in Paris with Michelin starred restaurants."
    )
    doc_id = uuid.uuid4()
    result = strategy.split(
        text, doc_id, {"similarity_threshold": 0.85, "min_chunk_size": 20, "max_chunk_size": 300}
    )

    assert len(result.chunks) >= 2
    assert "semantic_metrics" in result.metrics
    assert result.metrics["semantic_metrics"] is not None


def test_parent_child_hierarchy():
    strategy = ParentChildChunking()
    text = (
        "Section 1 details the foundational principles of generative models. "
        "Transformers utilize self-attention mechanisms to weigh token dependencies. "
        "Feed-forward layers process multi-head attention outputs. "
        "Section 2 discusses training dynamics and gradient descent optimizers. "
        "AdamW adapts learning rates for sparse parameter updates."
    )
    doc_id = uuid.uuid4()
    result = strategy.split(
        text,
        doc_id,
        {
            "parent_chunk_size": 200,
            "parent_overlap": 20,
            "child_chunk_size": 60,
            "child_overlap": 10,
        },
    )

    parent_chunks = [c for c in result.chunks if c.metadata.custom.get("tier") == "parent"]
    child_chunks = [c for c in result.chunks if c.metadata.custom.get("tier") == "child"]

    assert len(parent_chunks) >= 1
    assert len(child_chunks) >= 2

    # Verify every child has a valid parent_chunk_id that exists in parent_chunks
    parent_ids = {str(p.id) for p in parent_chunks}
    for child in child_chunks:
        assert child.metadata.parent_chunk_id in parent_ids


def test_structure_aware_preserves_tables_and_code():
    strategy = StructureAwareChunking()
    doc_id = uuid.uuid4()

    table_content = "| Metric | Score |\n| --- | --- |\n| Precision | 0.95 |"
    code_content = "```python\ndef retrieve(query):\n    return vector_store.search(query)\n```"

    document_context = {
        "sections": [
            {
                "id": str(uuid.uuid4()),
                "content": "Overview of evaluation metrics and testing procedures.",
                "content_type": "text",
                "heading": "Introduction",
                "page_number": 1,
            },
            {
                "id": str(uuid.uuid4()),
                "content": table_content,
                "content_type": "table",
                "heading": "Evaluation",
                "page_number": 1,
            },
            {
                "id": str(uuid.uuid4()),
                "content": code_content,
                "content_type": "code",
                "heading": "Implementation",
                "page_number": 2,
            },
        ]
    }

    result = strategy.split(
        text="dummy text",
        document_id=doc_id,
        config_dict={"chunk_size": 500, "chunk_overlap": 50},
        document_context=document_context,
    )

    # Verify table and code block each remained intact as atomic chunks
    table_chunks = [c for c in result.chunks if c.metadata.content_type == "table"]
    code_chunks = [c for c in result.chunks if c.metadata.content_type == "code"]

    assert len(table_chunks) == 1
    assert table_chunks[0].content == table_content
    assert len(code_chunks) == 1
    assert code_chunks[0].content == code_content
    assert code_chunks[0].metadata.page_numbers == [2]


def test_chunking_api_preview_and_experiments(client: TestClient, synthetic_txt_bytes):
    # 1. Ingest a document first
    res = client.post(
        "/api/v1/documents",
        files={"file": ("test_chunking.txt", synthetic_txt_bytes, "text/plain")},
    )
    assert res.status_code == 201
    doc_id = res.json()["id"]

    # 2. Get strategies
    strat_res = client.get("/api/v1/chunking/strategies")
    assert strat_res.status_code == 200
    strategies = strat_res.json()
    assert len(strategies) >= 6
    strat_names = {s["strategy_name"] for s in strategies}
    assert "fixed_size" in strat_names
    assert "recursive" in strat_names
    assert "sentence" in strat_names
    assert "structure_aware" in strat_names

    # 3. Post preview
    prev_res = client.post(
        "/api/v1/chunking/preview",
        json={
            "document_id": doc_id,
            "strategy": "fixed_size",
            "config": {"chunk_size": 100, "chunk_overlap": 20},
        },
    )
    assert prev_res.status_code == 200
    preview_data = prev_res.json()
    assert preview_data["strategy"] == "fixed_size"
    assert len(preview_data["chunks"]) >= 2
    assert "metrics" in preview_data

    # 4. Create and persist experiment
    exp_res = client.post(
        "/api/v1/chunking/experiments",
        json={
            "document_id": doc_id,
            "strategy": "recursive",
            "config": {"chunk_size": 150, "chunk_overlap": 25},
        },
    )
    assert exp_res.status_code == 201
    exp_data = exp_res.json()
    exp_id = exp_data["id"]
    assert exp_data["chunks_count"] >= 1

    # 5. List experiments
    list_res = client.get(f"/api/v1/chunking/experiments?document_id={doc_id}")
    assert list_res.status_code == 200
    assert list_res.json()["total"] >= 1

    # 6. Get experiment details
    get_res = client.get(f"/api/v1/chunking/experiments/{exp_id}")
    assert get_res.status_code == 200
    assert len(get_res.json()["chunks"]) == exp_data["chunks_count"]

    # 7. Compare strategies
    comp_res = client.post(
        "/api/v1/chunking/compare",
        json={
            "document_id": doc_id,
            "strategies": [
                {"strategy": "fixed_size", "config": {"chunk_size": 120, "chunk_overlap": 20}},
                {"strategy": "sentence", "config": {"chunk_size": 120, "sentence_overlap": 1}},
            ],
        },
    )
    assert comp_res.status_code == 200
    comp_data = comp_res.json()
    assert len(comp_data["results"]) == 2

    # 8. Delete experiment
    del_res = client.delete(f"/api/v1/chunking/experiments/{exp_id}")
    assert del_res.status_code == 204

    # Verify deleted
    get_after_del = client.get(f"/api/v1/chunking/experiments/{exp_id}")
    assert get_after_del.status_code == 404
