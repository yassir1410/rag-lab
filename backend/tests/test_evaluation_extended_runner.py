import uuid
import pytest
from app.db.session import SessionLocal
from app.models.evaluation import EvaluationDataset, EvaluationQuestion, EvaluationRun, EvaluationItemResult
from app.evaluation.runner import evaluation_runner


def test_extended_runner_unified_rag_execution():
    db = SessionLocal()
    try:
        # Create dataset
        ds = EvaluationDataset(name="Unified Runner Test Dataset", version="13.0")
        db.add(ds)
        db.commit()

        # Add question with expected routes and numerical ground truth
        q = EvaluationQuestion(
            dataset_id=ds.id,
            question="According to the planetary data table on page 1, what is Jupiter's surface gravity?",
            reference_answer="According to the table, Jupiter's surface gravity is 23.1 m/s².",
            acceptable_answers=["23.1 m/s² [TABLE_P1]", "23.1 m/s²"],
            expected_routes=["multimodal_table"],
            expected_citations=["[TABLE_P1]"],
            modality="table",
            category="factual",
            numerical_answer={"value": 23.1, "tolerance": 0.5, "unit": "m/s²"},
        )
        db.add(q)
        db.commit()

        # Create Unified evaluation run with mock provider
        run = EvaluationRun(
            dataset_id=ds.id,
            name="Unified Mock Test Run",
            status="pending",
            run_type="unified",
            config={
                "routing_mode": "auto",
                "llm_provider": "mock",
                "llm_model": "mock-instruct-v1",
                "top_k": 3,
                "max_questions": 1,
            },
        )
        db.add(run)
        db.commit()

        # Execute run
        completed_run = evaluation_runner.run_evaluation(db, run.id)

        assert completed_run.status == "completed"
        assert completed_run.progress_total == 1
        assert completed_run.progress_current == 1
        assert "dataset_version" in completed_run.experiment_snapshot

        # Verify Item Result
        item = db.query(EvaluationItemResult).filter(EvaluationItemResult.run_id == run.id).first()
        assert item is not None
        assert item.status == "success"
        assert item.trace_id is not None
        assert "f1@3" in item.retrieval_metrics
        assert "exact_match" in item.accuracy_metrics
        assert "route_accuracy" in item.routing_metrics

        # Cleanup
        db.delete(ds)
        db.commit()
    finally:
        db.close()


def test_extended_runner_safe_cancellation():
    db = SessionLocal()
    try:
        ds = EvaluationDataset(name="Cancel Runner Test Dataset")
        db.add(ds)
        db.commit()

        # Add 3 questions
        for i in range(3):
            q = EvaluationQuestion(
                dataset_id=ds.id,
                question=f"Test question {i}?",
                reference_answer="Test answer",
            )
            db.add(q)
        db.commit()

        run = EvaluationRun(
            dataset_id=ds.id,
            name="Cancel Test Run",
            status="pending",
            run_type="retrieval_only",
            config={"top_k": 3},
        )
        db.add(run)
        db.commit()

        # Trigger cancellation before execution
        evaluation_runner.cancel_run(run.id)
        cancelled_run = evaluation_runner.run_evaluation(db, run.id)

        assert cancelled_run.status == "cancelled"

        # Cleanup
        db.delete(ds)
        db.commit()
    finally:
        db.close()


def test_extended_runner_retrieval_only_f1():
    db = SessionLocal()
    try:
        ds = EvaluationDataset(name="F1 Metric Runner Dataset")
        db.add(ds)
        db.commit()

        q = EvaluationQuestion(
            dataset_id=ds.id,
            question="Mars atmospheric pressure?",
            reference_answer="6 millibars",
            relevant_chunk_ids=[],
        )
        db.add(q)
        db.commit()

        run = EvaluationRun(
            dataset_id=ds.id,
            name="F1 Metric Test Run",
            status="pending",
            run_type="retrieval_only",
            config={"top_k": 5, "search_algorithm": "exact"},
        )
        db.add(run)
        db.commit()

        completed_run = evaluation_runner.run_evaluation(db, run.id)
        assert completed_run.status == "completed"
        assert "mean_f1@5" in completed_run.aggregate_metrics

        db.delete(ds)
        db.commit()
    finally:
        db.close()
