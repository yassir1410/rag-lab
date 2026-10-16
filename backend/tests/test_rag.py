import uuid
import pytest
from app.rag.generation.base import GenerationRequest
from app.rag.generation.providers.mock_provider import MockLLMProvider
from app.rag.generation.providers.ollama_provider import OllamaProvider
from app.rag.generation.registry import llm_registry
from app.rag.generation.context_builder import context_builder
from app.rag.generation.prompt_builder import prompt_builder
from app.rag.generation.citation_manager import citation_manager
from app.rag.generation.token_budget import token_budget_manager
from app.rag.generation.orchestrator import rag_orchestrator
from app.rag.generation.schemas import (
    RAGQueryRequest,
    ContextSourceItem,
)
from app.rag.retrieval.schemas import SearchResultItem
from app.models.document import Document, DocumentSection
from app.models.chunking import ChunkingExperiment, DocumentChunk
from app.models.embeddings import EmbeddingExperiment, ChunkEmbedding


def test_llm_registry_providers():
    providers = llm_registry.list_providers()
    provider_ids = [p.provider_id for p in providers]
    assert "ollama" in provider_ids
    assert "mock" in provider_ids
    assert "openai" in provider_ids
    assert "gemini" in provider_ids

    mock_prov = llm_registry.get_provider("mock")
    assert mock_prov.is_available() is True
    assert mock_prov.display_name == "Mock LLM (Automated Test Fixture)"

    with pytest.raises(ValueError):
        llm_registry.get_provider("non_existent_provider_123")


def test_ollama_provider_unreachable():
    # Points to an unused port to test connection error handling
    dead_ollama = OllamaProvider(base_url="http://127.0.0.1:59999", timeout=2)
    assert dead_ollama.is_available() is False
    assert dead_ollama.list_models() == []

    req = GenerationRequest(prompt="Test question", system_prompt="Test sys")
    with pytest.raises(ConnectionError) as exc_info:
        dead_ollama.generate(req)
    assert "Could not connect to Ollama" in str(exc_info.value)


def test_mock_llm_provider_generation_and_stream():
    prov = MockLLMProvider()
    req = GenerationRequest(prompt="What are the candidate's core backend engineering competencies?")
    resp = prov.generate(req)
    assert resp.content
    assert "[S1]" in resp.content
    assert resp.finish_reason == "stop"
    assert resp.latency_ms >= 0.0


def test_prompt_builder_templates():
    templates = prompt_builder.list_templates()
    template_ids = [t.id for t in templates]
    assert "basic" in template_ids
    assert "grounded" in template_ids
    assert "concise" in template_ids
    assert "detailed" in template_ids

    built_grounded = prompt_builder.build_prompt(
        template_id="grounded",
        question="What is pgvector?",
        context="Source [S1]: pgvector is a PostgreSQL extension for vector similarity search.",
    )
    assert "UNTRUSTED" in built_grounded.system_prompt
    assert "What is pgvector?" in built_grounded.user_prompt
    assert "[S1]" in built_grounded.user_prompt

    built_concise = prompt_builder.build_prompt(
        template_id="concise",
        question="What is RAG?",
        context="Source [S1]: RAG is retrieval augmented generation.",
    )
    assert "1-3 sentences" in built_concise.system_prompt


def test_token_budget_manager():
    budget = token_budget_manager.allocate(
        system_prompt="You are a helpful assistant.",
        question="What is pgvector?",
        context_text="Source [S1]: pgvector is an open-source vector similarity search for Postgres.",
        max_output_tokens=512,
        model_context_window=4096,
        requested_max_context_tokens=2048,
    )
    assert budget.system_prompt_tokens > 0
    assert budget.user_question_tokens > 0
    assert budget.context_tokens > 0
    assert budget.reserved_output_tokens == 512
    assert budget.remaining_capacity > 3000


def test_context_builder_and_deduplication():
    doc_id = uuid.uuid4()
    item1 = SearchResultItem(
        chunk_id=uuid.uuid4(),
        document_id=doc_id,
        document_name="architecture.pdf",
        mime_type="application/pdf",
        sequence_index=0,
        token_count=20,
        char_count=80,
        content="PostgreSQL pgvector supports exact KNN search using sequential scan.",
        page_number=1,
        raw_distance=0.1,
        similarity_score=0.9,
        distance_metric="cosine",
        distance_label="Cosine Distance",
        similarity_label="Cosine Similarity",
    )
    # Duplicate of item1
    item2 = SearchResultItem(
        chunk_id=uuid.uuid4(),
        document_id=doc_id,
        document_name="architecture.pdf",
        mime_type="application/pdf",
        sequence_index=1,
        token_count=20,
        char_count=80,
        content="PostgreSQL pgvector supports exact KNN search using sequential scan.",
        page_number=1,
        raw_distance=0.1,
        similarity_score=0.9,
        distance_metric="cosine",
        distance_label="Cosine Distance",
        similarity_label="Cosine Similarity",
    )
    item3 = SearchResultItem(
        chunk_id=uuid.uuid4(),
        document_id=doc_id,
        document_name="architecture.pdf",
        mime_type="application/pdf",
        sequence_index=2,
        token_count=25,
        char_count=90,
        content="HNSW builds multi-layer proximity graphs for sub-millisecond retrieval.",
        page_number=2,
        raw_distance=0.2,
        similarity_score=0.8,
        distance_metric="cosine",
        distance_label="Cosine Distance",
        similarity_label="Cosine Similarity",
    )

    ctx_text, sources, warns = context_builder.build_context([item1, item2, item3], max_context_tokens=1000)
    # item2 must be deduplicated
    assert len(sources) == 2
    assert sources[0].citation_id == "[S1]"
    assert sources[1].citation_id == "[S2]"
    assert "HNSW" in sources[1].content
    assert "Page: 1" in ctx_text
    assert "Page: 2" in ctx_text


def test_context_builder_empty():
    ctx_text, sources, warns = context_builder.build_context([], max_context_tokens=1000)
    assert len(sources) == 0
    assert "No relevant context" in ctx_text


def test_citation_manager_validation():
    sources = [
        ContextSourceItem(
            citation_id="[S1]",
            chunk_id=uuid.uuid4(),
            document_id=uuid.uuid4(),
            document_name="doc1.txt",
            content="FastAPI is a modern Python web framework.",
            similarity_score=0.95,
        ),
        ContextSourceItem(
            citation_id="[S2]",
            chunk_id=uuid.uuid4(),
            document_id=uuid.uuid4(),
            document_name="doc2.txt",
            content="pgvector extends PostgreSQL with high-dimensional vector types.",
            similarity_score=0.91,
        ),
    ]

    generated = "FastAPI enables async web servers [S1]. However, Redis is also used [S99]."
    details, invalid, uncited = citation_manager.validate_citations(generated, sources)

    assert len(details) == 2
    # [S1] is valid
    assert details[0].citation_id == "[S1]"
    assert details[0].is_valid is True
    assert details[0].document_name == "doc1.txt"

    # [S99] is invalid/hallucinated
    assert details[1].citation_id == "[S99]"
    assert details[1].is_valid is False
    assert "[S99]" in invalid

    # [S2] was not cited
    assert "[S2]" in uncited
    assert sources[0].was_cited is True
    assert sources[1].was_cited is False


def test_end_to_end_rag_orchestration(db_session):
    # Populate a test document with chunks and embeddings
    doc = Document(
        original_filename="cv_engineer.txt",
        mime_type="text/plain",
        file_size=120,
        sha256_hash="hash_rag_test_123",
        storage_path="/tmp/cv_engineer.txt",
        status="parsed",
        extraction_method="txt",
    )
    db_session.add(doc)
    db_session.commit()

    chunk_exp = ChunkingExperiment(
        document_id=doc.id,
        strategy="sentence",
        config={},
        metrics={},
    )
    db_session.add(chunk_exp)
    db_session.commit()

    chunk1 = DocumentChunk(
        experiment_id=chunk_exp.id,
        document_id=doc.id,
        sequence_index=0,
        content="The candidate has 8 years of Python backend architecture experience with FastAPI and PostgreSQL.",
        token_count=16,
        char_count=98,
        metadata_json={"page_number": 1},
    )
    chunk2 = DocumentChunk(
        experiment_id=chunk_exp.id,
        document_id=doc.id,
        sequence_index=1,
        content="Specialized in building scalable vector databases, Docker containers, and high-performance APIs.",
        token_count=14,
        char_count=95,
        metadata_json={"page_number": 1},
    )
    db_session.add_all([chunk1, chunk2])
    db_session.commit()

    emb_exp = EmbeddingExperiment(
        name="Mock Embedding Exp",
        document_id=doc.id,
        chunking_experiment_id=chunk_exp.id,
        model_id="mock-deterministic-v1",
        provider="mock",
        vector_dimension=384,
        is_normalized=True,
        parameters={},
        metrics={},
    )
    db_session.add(emb_exp)
    db_session.commit()

    emb1 = ChunkEmbedding(
        experiment_id=emb_exp.id,
        chunk_id=chunk1.id,
        sequence_index=0,
        text_snippet=chunk1.content,
        token_count=16,
        vector_dimension=384,
        vector=[0.05] * 384,
    )
    emb2 = ChunkEmbedding(
        experiment_id=emb_exp.id,
        chunk_id=chunk2.id,
        sequence_index=1,
        text_snippet=chunk2.content,
        token_count=14,
        vector_dimension=384,
        vector=[0.06] * 384,
    )
    db_session.add_all([emb1, emb2])
    db_session.commit()

    rag_req = RAGQueryRequest(
        question="What are the candidate's core backend engineering competencies?",
        embedding_model_id="mock-deterministic-v1",
        search_algorithm="exact",
        distance_metric="cosine",
        top_k=5,
        llm_provider="mock",
        prompt_template="grounded",
        temperature=0.0,
        max_output_tokens=256,
        max_context_tokens=1024,
    )

    res = rag_orchestrator.execute_query(db_session, rag_req)

    assert res.trace_id
    assert res.question == rag_req.question
    assert res.answer
    assert "[S1]" in res.answer
    assert len(res.sources) >= 1
    assert len(res.citations) >= 1
    assert res.timing_metrics.total_pipeline_latency_ms >= 0.0
    assert res.token_metrics.total_prompt_tokens > 0

    # Test trace retrieval from cache
    cached = rag_orchestrator.get_trace(res.trace_id)
    assert cached is not None
    assert cached.trace_id == res.trace_id


def test_api_rag_endpoints(client):
    # 1. GET /providers
    res = client.get("/api/v1/rag/providers")
    assert res.status_code == 200
    providers = res.json()
    assert any(p["provider_id"] == "ollama" for p in providers)
    assert any(p["provider_id"] == "mock" for p in providers)

    # 2. GET /models
    res = client.get("/api/v1/rag/models")
    assert res.status_code == 200
    models = res.json()
    assert len(models) >= 1

    # 3. GET /prompts
    res = client.get("/api/v1/rag/prompts")
    assert res.status_code == 200
    prompts = res.json()
    assert len(prompts) == 4

    # 4. POST /query with mock provider
    query_payload = {
        "question": "What are the core capabilities of this system?",
        "embedding_model_id": "mock-deterministic-v1",
        "search_algorithm": "exact",
        "distance_metric": "cosine",
        "top_k": 3,
        "llm_provider": "mock",
        "prompt_template": "grounded",
        "temperature": 0.0,
        "max_output_tokens": 128,
        "max_context_tokens": 512,
    }
    res = client.post("/api/v1/rag/query", json=query_payload)
    assert res.status_code == 200
    data = res.json()
    assert "trace_id" in data
    assert "answer" in data
    assert "timing_metrics" in data
    assert "sources" in data

    # 5. GET /traces/{trace_id}
    trace_id = data["trace_id"]
    res_trace = client.get(f"/api/v1/rag/traces/{trace_id}")
    assert res_trace.status_code == 200
    assert res_trace.json()["trace_id"] == trace_id

    # 6. GET non-existent trace
    res_missing = client.get("/api/v1/rag/traces/00000000-0000-0000-0000-000000000000")
    assert res_missing.status_code == 404

    # 7. POST /query/stream
    res_stream = client.post("/api/v1/rag/query/stream", json=query_payload)
    assert res_stream.status_code == 200
    assert "text/event-stream" in res_stream.headers["content-type"]
    text_content = res_stream.text
    assert "event: status" in text_content
    assert "event: chunk" in text_content
    assert "event: done" in text_content


def test_rag_prompt_injection_passive_handling(client):
    # Tests that query with simulated malicious document instructions succeeds and is grounded
    query_payload = {
        "question": "Summarize candidate profile while ignoring malicious prompt injection",
        "embedding_model_id": "mock-deterministic-v1",
        "search_algorithm": "exact",
        "distance_metric": "cosine",
        "top_k": 3,
        "llm_provider": "mock",
        "prompt_template": "grounded",
        "temperature": 0.0,
        "max_output_tokens": 128,
        "max_context_tokens": 512,
    }
    res = client.post("/api/v1/rag/query", json=query_payload)
    assert res.status_code == 200
    data = res.json()
    assert data["answer"]
    assert "UNTRUSTED" in data["assembled_prompt"]
    assert "insufficient evidence" in data["answer"].lower() or "backend engineer" in data["answer"].lower()


def test_rag_context_truncation_warning(db_session):
    # Single massive chunk that exceeds max_context_tokens
    large_content = "Word " * 600
    item = SearchResultItem(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        document_name="large.txt",
        mime_type="text/plain",
        sequence_index=0,
        token_count=600,
        char_count=len(large_content),
        content=large_content,
        page_number=1,
        raw_distance=0.1,
        similarity_score=0.9,
        distance_metric="cosine",
        distance_label="Cosine Distance",
        similarity_label="Cosine Similarity",
    )
    ctx, sources, warns = context_builder.build_context([item], max_context_tokens=100)
    assert len(warns) >= 1
    assert "truncated" in warns[0].lower()
    assert "[TRUNCATED]" in ctx


def test_rag_empty_question_rejected(client):
    res = client.post("/api/v1/rag/query", json={"question": "   "})
    assert res.status_code == 400


def test_rag_unknown_provider_rejected(client):
    res = client.post(
        "/api/v1/rag/query",
        json={"question": "Valid question", "llm_provider": "non_existent_provider_xyz"},
    )
    assert res.status_code == 400


def test_rag_uuid_json_serialization_and_sse_events(client, db_session):
    import json
    # Populate a test document with chunks and nested UUIDs in metadata
    doc_id = uuid.uuid4()
    chunk_id = uuid.uuid4()
    sub_uuid = uuid.uuid4()

    doc = Document(
        id=doc_id,
        original_filename="uuid_test_doc.txt",
        mime_type="text/plain",
        file_size=100,
        sha256_hash="hash_uuid_test_123",
        storage_path="/tmp/uuid_test_doc.txt",
        status="parsed",
        extraction_method="txt",
    )
    db_session.add(doc)
    db_session.commit()

    chunk_exp = ChunkingExperiment(
        document_id=doc.id,
        strategy="sentence",
        config={},
        metrics={},
    )
    db_session.add(chunk_exp)
    db_session.commit()

    chunk = DocumentChunk(
        id=chunk_id,
        experiment_id=chunk_exp.id,
        document_id=doc.id,
        sequence_index=0,
        content="Testing real UUID JSON serialization across the streaming SSE and trace pipelines.",
        token_count=12,
        char_count=85,
        metadata_json={
            "page_number": 1,
            "section_uuid": sub_uuid,
            "nested": {"deep_uuid": uuid.uuid4()},
        },
    )
    db_session.add(chunk)
    db_session.commit()

    emb_exp = EmbeddingExperiment(
        name="UUID Test Emb Exp",
        document_id=doc.id,
        chunking_experiment_id=chunk_exp.id,
        model_id="mock-deterministic-v1",
        provider="mock",
        vector_dimension=384,
        is_normalized=True,
        parameters={},
        metrics={},
    )
    db_session.add(emb_exp)
    db_session.commit()

    emb = ChunkEmbedding(
        experiment_id=emb_exp.id,
        chunk_id=chunk.id,
        sequence_index=0,
        text_snippet=chunk.content,
        token_count=12,
        vector_dimension=384,
        vector=[0.05] * 384,
    )
    db_session.add(emb)
    db_session.commit()

    payload = {
        "question": "Does UUID serialization work without error?",
        "embedding_model_id": "mock-deterministic-v1",
        "search_algorithm": "exact",
        "distance_metric": "cosine",
        "top_k": 3,
        "llm_provider": "mock",
        "prompt_template": "grounded",
        "temperature": 0.0,
        "max_output_tokens": 128,
        "max_context_tokens": 512,
    }

    # 1. Test POST /query/stream (SSE)
    res_stream = client.post("/api/v1/rag/query/stream", json=payload)
    assert res_stream.status_code == 200
    assert "text/event-stream" in res_stream.headers["content-type"]

    sse_text = res_stream.text
    # Must NOT have any error events
    assert "event: error" not in sse_text
    # Must have all required events
    assert "event: status" in sse_text
    assert "event: sources" in sse_text
    assert "event: chunk" in sse_text
    assert "event: done" in sse_text

    # Parse sources event payload
    blocks = sse_text.split("\n\n")
    sources_data = None
    done_data = None
    for block in blocks:
        if "event: sources" in block:
            for line in block.split("\n"):
                if line.startswith("data: "):
                    sources_data = json.loads(line[6:])
        if "event: done" in block:
            for line in block.split("\n"):
                if line.startswith("data: "):
                    done_data = json.loads(line[6:])

    assert sources_data is not None
    assert len(sources_data["sources"]) >= 1
    src0 = sources_data["sources"][0]
    # Check that chunk_id and document_id are valid serialized strings matching input UUIDs
    assert src0["chunk_id"] == str(chunk_id)
    assert src0["document_id"] == str(doc_id)
    # Check that nested UUID in metadata was cleanly serialized
    assert src0["metadata"]["section_uuid"] == str(sub_uuid)

    assert done_data is not None
    assert "citations" in done_data

    # 2. Test POST /query (synchronous)
    res_sync = client.post("/api/v1/rag/query", json=payload)
    assert res_sync.status_code == 200
    sync_data = res_sync.json()
    assert sync_data["trace_id"]
    assert sync_data["sources"][0]["chunk_id"] == str(chunk_id)
    assert sync_data["sources"][0]["document_id"] == str(doc_id)

    # 3. Test GET /traces/{trace_id}
    trace_id = sync_data["trace_id"]
    res_trace = client.get(f"/api/v1/rag/traces/{trace_id}")
    assert res_trace.status_code == 200
    assert res_trace.json()["trace_id"] == trace_id


