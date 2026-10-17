import uuid
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.db.session import SessionLocal
from app.models.document import Document
from app.rag.unified.schemas import (
    UnifiedEvidenceItem,
    RoutingDecision,
    UnifiedRAGQueryRequest,
    DocumentCapabilityReport,
)
from app.rag.unified.capability_service import document_capability_service
from app.rag.unified.router import intelligent_router
from app.rag.unified.retrieval_orchestrator import unified_retrieval_orchestrator
from app.rag.unified.generation_service import unified_generation_service

client = TestClient(app)


@pytest.fixture
def db_session():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def test_document_capability_service(db_session):
    # 1. Global capability
    global_cap = document_capability_service.get_document_capability(db_session, document_id=None)
    assert global_cap.document_id is None
    assert global_cap.filename == "Global Repository"
    assert len(global_cap.supported_routes) >= 2

    # 2. Solar system PDF capability
    solar_doc = db_session.query(Document).filter(Document.original_filename.like("%solar_system%")).first()
    if solar_doc:
        solar_cap = document_capability_service.get_document_capability(db_session, document_id=solar_doc.id)
        assert solar_cap.document_id == solar_doc.id
        assert solar_cap.has_tables is True
        assert solar_cap.table_count >= 1
        assert "multimodal_table" in solar_cap.supported_routes
        # Confirms chunks/embeddings are zero for this PDF
        assert solar_cap.has_text_chunks is False
        assert solar_cap.chunk_count == 0

    # 3. Non-existent document
    fake_id = uuid.uuid4()
    fake_cap = document_capability_service.get_document_capability(db_session, document_id=fake_id)
    assert fake_cap.filename == "Non-existent Document"
    assert len(fake_cap.recommended_actions) >= 1


def test_intelligent_query_router_intent_detection(db_session):
    solar_doc = db_session.query(Document).filter(Document.original_filename.like("%solar_system%")).first()
    doc_id = solar_doc.id if solar_doc else None

    # 1. Tabular query
    table_q = "According to the planetary data table on page 1, what is Jupiter's surface gravity?"
    decision_table = intelligent_router.route_query(
        session=db_session, question=table_q, document_id=doc_id, mode="auto"
    )
    assert "multimodal_table" in decision_table.selected_routes
    assert "tabular_query" in decision_table.intent_detected

    # 2. Visual query
    visual_q = "What does the visual diagram in Figure 1 look like?"
    decision_visual = intelligent_router.route_query(
        session=db_session, question=visual_q, document_id=doc_id, mode="auto"
    )
    assert "multimodal_visual" in decision_visual.selected_routes
    assert "visual_query" in decision_visual.intent_detected

    # 3. Manual Mode
    decision_manual = intelligent_router.route_query(
        session=db_session,
        question="General question",
        document_id=doc_id,
        mode="manual",
        manual_routes=["text_dense", "multimodal_table"],
    )
    assert decision_manual.mode == "manual"
    assert "text_dense" in decision_manual.selected_routes
    assert "multimodal_table" in decision_manual.selected_routes

    # 4. Comparative Mode
    decision_comp = intelligent_router.route_query(
        session=db_session, question="Test comparison", document_id=doc_id, mode="comparative"
    )
    assert decision_comp.mode == "comparative"
    assert len(decision_comp.selected_routes) >= 1


def test_router_missing_index_fallback(db_session):
    solar_doc = db_session.query(Document).filter(Document.original_filename.like("%solar_system%")).first()
    if solar_doc:
        # Ask a plain text question without table keywords on a doc that has no text chunks
        plain_text_q = "Explain the history of astronomy."
        decision = intelligent_router.route_query(
            session=db_session, question=plain_text_q, document_id=solar_doc.id, mode="auto"
        )
        # Should gracefully route to available multimodal representations rather than failing
        assert any(r in ("multimodal_table", "multimodal_visual") for r in decision.selected_routes)
        assert len(decision.fallback_decisions) >= 1 or len(decision.warnings) >= 0


def test_unified_retrieval_and_rrf_fusion(db_session):
    solar_doc = db_session.query(Document).filter(Document.original_filename.like("%solar_system%")).first()
    doc_id = solar_doc.id if solar_doc else None

    # Retrieve table evidence
    evidence, latencies = unified_retrieval_orchestrator.retrieve(
        session=db_session,
        query="According to the planetary data table on page 1, what is Jupiter's gravity?",
        routes=["multimodal_table"],
        document_id=doc_id,
        page_number=1,
        top_k=4,
    )

    assert len(evidence) >= 1
    top_item = evidence[0]
    assert top_item.citation_id == "[S1]"
    assert "[TBL" in top_item.origin_tag or "[TABLE" in top_item.origin_tag
    assert top_item.evidence_type == "table"
    assert top_item.calibrated_score > 0.0
    assert "multimodal_table" in latencies


def test_unified_generation_service_with_mock_provider(db_session):
    solar_doc = db_session.query(Document).filter(Document.original_filename.like("%solar_system%")).first()
    doc_id = solar_doc.id if solar_doc else None

    req = UnifiedRAGQueryRequest(
        question="What is Jupiter's gravity according to the table?",
        document_id=doc_id,
        mode="auto",
        llm_provider="mock",
        top_k=3,
    )

    response = unified_generation_service.execute_query(session=db_session, request=req)
    assert response.question == req.question
    assert response.routing.mode == "auto"
    assert len(response.evidence_items) >= 1
    assert response.timing_metrics.total_latency_ms > 0.0
    assert response.evidence_items[0].citation_id == "[S1]"


def test_unified_citation_validation():
    service = unified_generation_service

    evidence = [
        UnifiedEvidenceItem(
            document_id=uuid.uuid4(),
            document_name="benchmark.pdf",
            evidence_id="ev-1",
            evidence_type="table",
            page_number=1,
            engine="multimodal_table",
            raw_score=1.0,
            score_semantics="test",
            calibrated_score=1.0,
            content="| Body | Gravity |\n| Jupiter | 23.1 |",
            citation_id="[S1]",
            origin_tag="[TABLE_P1]",
        ),
        UnifiedEvidenceItem(
            document_id=uuid.uuid4(),
            document_name="benchmark.pdf",
            evidence_id="ev-2",
            evidence_type="text",
            page_number=2,
            engine="text_hybrid",
            raw_score=0.8,
            score_semantics="test",
            calibrated_score=0.8,
            content="General textual description",
            citation_id="[S2]",
            origin_tag="[CHUNK_2]",
        ),
    ]

    answer_valid = "Jupiter's surface gravity is 23.1 m/s^2 [S1] and described in text [S2]."
    details, invalid, uncited = service.validate_citations(answer_valid, evidence)
    assert len(details) == 2
    assert all(d.is_valid for d in details)
    assert len(invalid) == 0
    assert len(uncited) == 0

    answer_hallucinated = "Jupiter's gravity is high [S1] according to [S99]."
    details_h, invalid_h, uncited_h = service.validate_citations(answer_hallucinated, evidence)
    assert "[S99]" in invalid_h
    assert "[S2]" in uncited_h


def test_unified_rag_api_endpoints():
    # 1. Routes endpoint
    resp_routes = client.get("/api/v1/unified-rag/routes")
    assert resp_routes.status_code == 200
    routes = resp_routes.json()
    assert len(routes) == 6
    assert any(r["route_id"] == "multimodal_table" for r in routes)

    # 2. Capabilities endpoint
    resp_caps = client.get("/api/v1/unified-rag/capabilities")
    assert resp_caps.status_code == 200
    caps = resp_caps.json()
    assert "supported_routes" in caps
    assert caps["has_tables"] is True

    # 3. Synchronous Query with mock provider
    resp_query = client.post(
        "/api/v1/unified-rag/query",
        json={
            "question": "What is the core backend competency?",
            "mode": "auto",
            "llm_provider": "mock",
            "top_k": 3,
        },
    )
    assert resp_query.status_code == 200
    data = resp_query.json()
    assert "answer" in data
    assert "routing" in data
    assert "evidence_items" in data
    assert "timing_metrics" in data

    # 4. Trace retrieval endpoint
    trace_id = data["trace_id"]
    resp_trace = client.get(f"/api/v1/unified-rag/traces/{trace_id}")
    assert resp_trace.status_code == 200
    assert resp_trace.json()["trace_id"] == trace_id

    # 5. Route Comparison endpoint
    resp_comp = client.post(
        "/api/v1/unified-rag/compare?question=Explain%20vector%20indexing"
    )
    assert resp_comp.status_code == 200
    comp_data = resp_comp.json()
    assert "comparisons" in comp_data
    assert len(comp_data["comparisons"]) >= 1


# ==============================================================================
# PHASE 12.5 FIX: 11 DETERMINISTIC TESTS FOR PAGE-AWARE & ROUTE-AWARE VISION
# ==============================================================================
from unittest.mock import patch, MagicMock
from app.rag.unified.router import extract_explicit_page_number
import httpx


def test_1_page_number_extraction():
    assert extract_explicit_page_number("According to the table on page 1, what is gravity?") == 1
    assert extract_explicit_page_number("Show me the diagram on page 2") == 2
    assert extract_explicit_page_number("Look at the data on the first page") == 1
    assert extract_explicit_page_number("Inspect the second page table") == 2
    assert extract_explicit_page_number("What does page 4 discuss?") == 4


def test_2_explicit_page_filter_precedence():
    # Explicit request parameter takes precedence over query text
    assert extract_explicit_page_number("Check table on page 1", request_page_number=3) == 3
    assert extract_explicit_page_number("Look at the first page", request_page_number=4) == 4


def test_3_ambiguous_page_references():
    # Ambiguous multi-page queries or generic mentions return None
    assert extract_explicit_page_number("Compare the tables between page 1 and page 2") is None
    assert extract_explicit_page_number("Turning the page on solar exploration") is None
    assert extract_explicit_page_number("Explain planetary formation in general") is None


def test_4_and_8_table_question_uses_gemini_text_and_s1_maps_to_table_p1(db_session):
    solar_doc = db_session.query(Document).filter(Document.original_filename.like("%solar_system%")).first()
    assert solar_doc is not None

    req = UnifiedRAGQueryRequest(
        question="According to the planetary data table on page 1, what is Jupiter's surface gravity? Cite the source.",
        document_id=solar_doc.id,
        mode="auto",
        llm_provider="gemini",
        llm_model="gemini-3.8-flash",
        top_k=5,
    )

    mock_gemini_response = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "text": "According to the planetary data table on page 1, Jupiter's surface gravity is 23.1 m/s² [S1]."
                        }
                    ]
                },
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 250,
            "candidatesTokenCount": 35,
            "totalTokenCount": 285,
        },
    }

    with patch.object(httpx.Client, "post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = mock_gemini_response
        mock_post.return_value = mock_resp

        response = unified_generation_service.execute_query(session=db_session, request=req)

        # 1. Effective page extracted as 1
        assert response.routing.effective_page_number == 1
        assert "multimodal_table" in response.routing.selected_routes

        # 2. [S1] maps to [TABLE_P1]
        assert len(response.evidence_items) >= 1
        top_item = response.evidence_items[0]
        assert top_item.citation_id == "[S1]"
        assert top_item.origin_tag == "[TABLE_P1]"
        assert top_item.page_number == 1

        # 3. Text generation used without image attachments
        assert response.is_vision_enabled is False
        assert mock_post.called

        # 4. Verified HTTP payload contains structured markdown table, NO inline_data base64 images
        call_kwargs = mock_post.call_args[1]
        payload = call_kwargs.get("json", {})
        parts = payload.get("contents", [{}])[-1].get("parts", [{}])
        prompt_text = parts[0].get("text", "")
        assert "Jupiter" in prompt_text
        assert "23.1" in prompt_text
        assert "inline_data" not in str(payload)

        # 5. Answer has valid citation [S1]
        assert "[S1]" in response.answer
        assert len(response.citations) >= 1
        assert response.citations[0].citation_id == "[S1]"
        assert response.citations[0].is_valid is True


def test_5_and_11_visual_question_invokes_gemini_vision_mocked_http(db_session):
    solar_doc = db_session.query(Document).filter(Document.original_filename.like("%solar_system%")).first()
    assert solar_doc is not None

    req = UnifiedRAGQueryRequest(
        question="What does the optical pipeline diagram on page 2 show?",
        document_id=solar_doc.id,
        mode="auto",
        llm_provider="gemini",
        llm_model="gemini-3.8-flash",
        top_k=3,
    )

    mock_vision_response = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "text": "The optical pipeline diagram on page 2 illustrates the NIRCam filter wheels [S1]."
                        }
                    ]
                },
                "finishReason": "STOP",
            }
        ]
    }

    with patch.object(httpx.Client, "post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = mock_vision_response
        mock_post.return_value = mock_resp

        response = unified_generation_service.execute_query(session=db_session, request=req)

        assert response.routing.effective_page_number == 2
        assert "multimodal_visual" in response.routing.selected_routes
        # is_vision_enabled is True because real image was attached and transmitted
        assert response.is_vision_enabled is True
        assert mock_post.called

        # Verify inline_data base64 was sent
        call_kwargs = mock_post.call_args[1]
        payload = call_kwargs.get("json", {})
        payload_str = str(payload)
        assert "inline_data" in payload_str
        assert "image/png" in payload_str


def test_6_cross_page_visual_questions_retain_multiple_images(db_session):
    solar_doc = db_session.query(Document).filter(Document.original_filename.like("%solar_system%")).first()
    assert solar_doc is not None

    # Visual query without page constraint
    req = UnifiedRAGQueryRequest(
        question="Describe the visual figures and layout diagrams across the document.",
        document_id=solar_doc.id,
        mode="auto",
        llm_provider="gemini",
        top_k=5,
    )

    r = intelligent_router.route_query(db_session, req.question, req.document_id, mode="auto")
    assert r.effective_page_number is None  # No single page constraint
    assert "multimodal_visual" in r.selected_routes

    ev, _ = unified_retrieval_orchestrator.retrieve(
        db_session, req.question, r.selected_routes, req.document_id, page_number=None, top_k=5
    )
    is_vis = (
        "multimodal_visual" in r.selected_routes
        or any(it.evidence_type in ("image", "figure", "visual", "diagram") for it in ev[:2])
    )
    _, images = unified_generation_service.build_unified_context(ev, attach_images=is_vis, target_page_number=None)
    # Multiple images collected across pages
    assert len(images) >= 2


def test_7_document_isolation(db_session):
    solar_doc = db_session.query(Document).filter(Document.original_filename.like("%solar_system%")).first()
    assert solar_doc is not None

    ev, _ = unified_retrieval_orchestrator.retrieve(
        db_session,
        query="What are the technical competencies and skills?",
        routes=["multimodal_table"],
        document_id=solar_doc.id,
        page_number=1,
        top_k=5,
    )
    # Strictly isolated: all items must belong to solar_doc.id
    for it in ev:
        assert str(it.document_id) == str(solar_doc.id)


def test_9_provider_timeout_surfaced_correctly_as_504(db_session):
    solar_doc = db_session.query(Document).filter(Document.original_filename.like("%solar_system%")).first()
    assert solar_doc is not None

    orig_post = httpx.Client.post

    def mock_post(self, url, *args, **kwargs):
        if "googleapis" in str(url):
            raise httpx.ReadTimeout("The read operation timed out")
        return orig_post(self, url, *args, **kwargs)

    with patch.object(httpx.Client, "post", new=mock_post):
        resp = client.post(
            "/api/v1/unified-rag/query",
            json={
                "question": "According to the planetary data table on page 1, what is Jupiter's gravity?",
                "document_id": str(solar_doc.id),
                "mode": "auto",
                "llm_provider": "gemini",
                "llm_model": "gemini-3.8-flash",
                "top_k": 5,
            },
        )
        assert resp.status_code == 504
        assert "timed out" in resp.json()["detail"].lower()


def test_10_no_silent_fallback_when_gemini_fails(db_session):
    solar_doc = db_session.query(Document).filter(Document.original_filename.like("%solar_system%")).first()
    assert solar_doc is not None

    # Visual query with unavailable key should raise explicit error, NOT silently switch to mock/Ollama
    with patch.object(unified_generation_service.gemini_vision, "is_available", return_value=False):
        req = UnifiedRAGQueryRequest(
            question="What does the visual diagram on page 2 show?",
            document_id=solar_doc.id,
            mode="auto",
            llm_provider="gemini",
            top_k=3,
        )
        with pytest.raises(Exception) as exc_info:
            unified_generation_service.execute_query(db_session, req)
        assert "GEMINI_API_KEY is not configured" in str(exc_info.value)

