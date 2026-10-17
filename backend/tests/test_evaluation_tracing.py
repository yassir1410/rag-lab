import uuid
import pytest
from app.db.session import SessionLocal
from app.evaluation.tracing import EvaluationTracer, RedactionPolicy
from app.models.evaluation import EvaluationTrace, EvaluationRun, EvaluationDataset


def test_redaction_policy_secrets():
    # Sensitive keys
    assert RedactionPolicy.redact_value("api_key", "AIzaSySecretKey123") == "[REDACTED_SECRET]"
    assert RedactionPolicy.redact_value("authorization", "Bearer my-secret-token") == "[REDACTED_SECRET]"
    assert RedactionPolicy.redact_value("x-goog-api-key", "my-gemini-key") == "[REDACTED_SECRET]"

    # Base64 image payload
    fake_b64 = "data:image/png;base64," + "A" * 100
    redacted_b64 = RedactionPolicy.redact_value("image_preview", fake_b64)
    assert "[REDACTED_BASE64_IMAGE]" in redacted_b64
    assert fake_b64 not in redacted_b64

    # Safe text
    assert RedactionPolicy.redact_value("query", "Jupiter surface gravity") == "Jupiter surface gravity"


def test_evaluation_tracer_stage_timing():
    tracer = EvaluationTracer(
        provider="mock",
        model="mock-instruct-v1",
        document_ids=["doc-1"],
    )

    with tracer.stage("routing", {"mode": "auto"}):
        pass  # simulated fast step

    with tracer.stage("retrieval", {"top_k": 5}):
        pass

    tracer.finish()
    d = tracer.to_dict()

    assert len(d["stages"]) == 2
    assert d["stages"][0]["stage_name"] == "routing"
    assert d["stages"][0]["duration_ms"] >= 0.0
    assert d["stages"][1]["stage_name"] == "retrieval"
    assert d["total_latency_ms"] >= 0.0


def test_evaluation_trace_database_persistence():
    db = SessionLocal()
    try:
        # Create dummy dataset and run
        ds = EvaluationDataset(name="Trace Test Dataset")
        db.add(ds)
        db.commit()

        run = EvaluationRun(
            dataset_id=ds.id,
            name="Trace Test Run",
            status="completed",
            run_type="unified",
        )
        db.add(run)
        db.commit()

        # Run tracer and persist
        tracer = EvaluationTracer(
            run_id=run.id,
            provider="gemini",
            model="gemini-3.8-flash",
            document_ids=["doc-solar-system"],
        )
        tracer.set_routes(["multimodal_table"])
        with tracer.stage("routing", {"x-goog-api-key": "secret-123"}):
            pass
        with tracer.stage("retrieval", {"evidence_count": 3}):
            pass

        trace_rec = tracer.persist(db)
        db.commit()

        # Verify retrieved trace
        fetched = db.query(EvaluationTrace).filter(EvaluationTrace.id == trace_rec.id).first()
        assert fetched is not None
        assert fetched.trace_id == tracer.trace_id
        assert fetched.run_id == run.id
        assert fetched.provider == "gemini"
        assert fetched.model == "gemini-3.8-flash"
        assert fetched.selected_routes == ["multimodal_table"]
        assert len(fetched.stages) == 2

        # Verify secret was redacted in persisted record
        assert fetched.stages[0]["metadata"]["x-goog-api-key"] == "[REDACTED_SECRET]"

        # Cleanup
        db.delete(ds)
        db.commit()
    finally:
        db.close()
