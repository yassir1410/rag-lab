import uuid
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.models.evaluation import EvaluationDataset, EvaluationQuestion, EvaluationRun
from app.models.document import Document
from app.models.chunking import ChunkingExperiment, DocumentChunk
from app.models.embeddings import EmbeddingExperiment, ChunkEmbedding
from app.evaluation.cost import pricing_registry


def test_model_pricing_registry():
    spec_gpt = pricing_registry.get_pricing("gpt-4o-mini")
    assert spec_gpt is not None
    assert spec_gpt.input_cost_per_1m_usd == 0.15

    spec_gemini = pricing_registry.get_pricing("gemini-3.8-flash")
    assert spec_gemini is not None
    assert spec_gemini.provider_id == "gemini"

    calc = pricing_registry.calculate_cost(
        model_id="gpt-4o-mini",
        input_tokens=1000,
        output_tokens=500,
        judge_tokens=200,
        judge_model_id="gemini-1.5-flash",
    )
    assert calc.total_tokens == 1700
    assert calc.total_estimated_cost_usd > 0.0
    assert calc.currency == "USD"

    # Ollama free local check
    calc_ollama = pricing_registry.calculate_cost(
        model_id="qwen2.5:3b",
        input_tokens=2000,
        output_tokens=500,
    )
    assert calc_ollama.total_estimated_cost_usd == 0.0


def test_dataset_crud_and_questions_flow(client: TestClient):
    # 1. Create dataset
    res = client.post(
        "/api/v1/evaluation/datasets",
        json={"name": "Engineering Golden Set", "description": "Benchmark questions for resume testing"},
    )
    assert res.status_code == 201
    dataset = res.json()
    dataset_id = dataset["id"]
    assert dataset["name"] == "Engineering Golden Set"
    assert dataset["question_count"] == 0

    # 2. Add question to dataset
    q_payload = {
        "question": "What backend frameworks does the candidate use?",
        "reference_answer": "FastAPI, Express, and Spring Boot.",
        "category": "factual",
        "difficulty": "easy",
        "is_answerable": True,
        "is_reviewed": False,
    }
    q_res = client.post(f"/api/v1/evaluation/datasets/{dataset_id}/questions", json=q_payload)
    assert q_res.status_code == 201
    q_data = q_res.json()
    q_id = q_data["id"]
    assert q_data["category"] == "factual"
    assert q_data["is_reviewed"] is False

    # 3. Duplicate question check
    dup_res = client.post(f"/api/v1/evaluation/datasets/{dataset_id}/questions", json=q_payload)
    assert dup_res.status_code == 400

    # 4. Review and approve question
    review_res = client.post(
        f"/api/v1/evaluation/questions/{q_id}/review",
        json={"is_reviewed": True, "reviewer_notes": "Verified against resume text."},
    )
    assert review_res.status_code == 200
    assert review_res.json()["is_reviewed"] is True
    assert review_res.json()["reviewer_notes"] == "Verified against resume text."

    # 5. List questions
    list_q_res = client.get(f"/api/v1/evaluation/datasets/{dataset_id}/questions")
    assert list_q_res.status_code == 200
    assert len(list_q_res.json()) == 1

    # 6. Export dataset
    export_res = client.get(f"/api/v1/evaluation/datasets/{dataset_id}/export")
    assert export_res.status_code == 200
    export_data = export_res.json()
    assert export_data["total_questions"] == 1
    assert export_data["questions"][0]["question"] == "What backend frameworks does the candidate use?"

    # 7. Import questions into a new dataset
    new_ds_res = client.post(
        "/api/v1/evaluation/datasets",
        json={"name": "Imported Dataset"},
    )
    new_ds_id = new_ds_res.json()["id"]

    import_res = client.post(
        f"/api/v1/evaluation/datasets/{new_ds_id}/import",
        json={
            "questions": [
                {
                    "question": "Where did the candidate study?",
                    "reference_answer": "INSEA in Rabat.",
                    "category": "factual",
                },
                {
                    "question": "What is the candidate's favorite color?",
                    "reference_answer": "Not mentioned in the documents.",
                    "is_answerable": False,
                    "category": "unanswerable",
                },
            ]
        },
    )
    assert import_res.status_code == 200
    assert import_res.json()["imported_questions"] == 2


def test_evaluation_run_execution_and_comparison(client: TestClient, db_session: Session):
    # Setup test document and chunks in db
    doc = Document(
        original_filename="test_cv.pdf",
        storage_path="/tmp/test_cv.pdf",
        mime_type="application/pdf",
        file_size=1000,
        sha256_hash="hash-eval-1",
        extraction_method="pymupdf",
        status="processed",
    )
    db_session.add(doc)
    db_session.commit()

    chunk_exp = ChunkingExperiment(
        document_id=doc.id,
        strategy="fixed_size",
        config={"chunk_size": 400},
        metrics={"total_chunks": 2},
    )
    db_session.add(chunk_exp)
    db_session.commit()

    c1 = DocumentChunk(
        experiment_id=chunk_exp.id,
        document_id=doc.id,
        sequence_index=0,
        content="Yassir FAHIMI Software Engineer with Python and FastAPI backend experience.",
        token_count=50,
        char_count=150,
    )
    c2 = DocumentChunk(
        experiment_id=chunk_exp.id,
        document_id=doc.id,
        sequence_index=1,
        content="Graduated from INSEA Rabat with State Engineering Degree in Software.",
        token_count=50,
        char_count=140,
    )
    db_session.add_all([c1, c2])
    db_session.commit()

    # Create dataset
    ds = EvaluationDataset(name="Evaluation Flow Dataset")
    db_session.add(ds)
    db_session.commit()

    q1 = EvaluationQuestion(
        dataset_id=ds.id,
        question="What backend framework does Yassir use?",
        reference_answer="FastAPI and Python.",
        relevant_chunk_ids=[str(c1.id)],
        category="factual",
        is_answerable=True,
    )
    q2 = EvaluationQuestion(
        dataset_id=ds.id,
        question="Where did the candidate graduate?",
        reference_answer="INSEA Rabat.",
        relevant_chunk_ids=[str(c2.id)],
        category="factual",
        is_answerable=True,
    )
    db_session.add_all([q1, q2])
    db_session.commit()

    # 1. Run End-to-End Evaluation Run with Mock LLM
    run_req = {
        "dataset_id": str(ds.id),
        "name": "End-to-End Mock Run",
        "run_type": "end_to_end",
        "embedding_model_id": "sentence-transformers/all-MiniLM-L6-v2",
        "search_algorithm": "exact",
        "top_k": 3,
        "llm_provider": "mock",
        "llm_model": "mock-instruct-v1",
        "judge_provider": "heuristic",
    }
    run_res = client.post("/api/v1/evaluation/runs", json=run_req)
    assert run_res.status_code == 201
    run_data = run_res.json()
    assert run_data["status"] == "completed"
    assert "mean_precision@3" in run_data["aggregate_metrics"]
    assert "mean_faithfulness" in run_data["aggregate_metrics"]
    assert run_data["latency_summary"]["p50_ms"] >= 0.0

    run_id = run_data["id"]

    # 2. Inspect per-question results
    items_res = client.get(f"/api/v1/evaluation/runs/{run_id}/questions")
    assert items_res.status_code == 200
    items = items_res.json()
    assert len(items) == 2
    assert items[0]["faithfulness_score"] is not None
    assert "precision@3" in items[0]["retrieval_metrics"]

    # 3. Create second run (Retrieval-Only)
    run_ret_req = {
        "dataset_id": str(ds.id),
        "name": "Retrieval Only Run",
        "run_type": "retrieval_only",
        "embedding_model_id": "sentence-transformers/all-MiniLM-L6-v2",
        "search_algorithm": "exact",
        "top_k": 3,
    }
    run_ret_res = client.post("/api/v1/evaluation/runs", json=run_ret_req)
    assert run_ret_res.status_code == 201
    run_ret_id = run_ret_res.json()["id"]

    # 4. Compare runs
    compare_res = client.get(f"/api/v1/evaluation/compare?run_a={run_id}&run_b={run_ret_id}")
    assert compare_res.status_code == 200
    comp = compare_res.json()
    assert "metrics_comparison" in comp
    assert "latency_comparison" in comp


def test_metrics_educational_catalog_endpoint(client: TestClient):
    res = client.get("/api/v1/evaluation/metrics")
    assert res.status_code == 200
    catalog = res.json()
    assert len(catalog) >= 7
    metric_ids = [m["id"] for m in catalog]
    assert "precision_at_k" in metric_ids
    assert "recall_at_k" in metric_ids
    assert "mrr" in metric_ids
    assert "ndcg_at_k" in metric_ids
    assert "faithfulness" in metric_ids
    assert "citation_support_precision" in metric_ids
    assert "answer_relevance" in metric_ids
