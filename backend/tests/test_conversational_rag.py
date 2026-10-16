import uuid
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.main import app
from app.db.session import SessionLocal
from app.models.conversation import Conversation, ConversationMessage, ConversationSummary
from app.rag.conversation.repository import conversation_repo
from app.rag.conversation.memory import memory_registry
from app.rag.conversation.schemas import MemoryConfig, TurnExecutionRequest
from app.rag.conversation.query_rewriter import query_rewriter
from app.rag.conversation.orchestrator import conversational_orchestrator
from app.rag.conversation.evaluation import conversational_benchmark_runner


@pytest.fixture
def db_session():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def client():
    return TestClient(app)


def test_conversation_crud_and_cascade(db_session: Session):
    # 1. Create conversation
    conv = conversation_repo.create_conversation(
        session=db_session,
        title="Test Conversation Session",
        settings={"memory_strategy": "sliding_window"},
    )
    assert conv.id is not None
    assert conv.title == "Test Conversation Session"
    assert conv.status == "active"

    # 2. Add Messages
    m1 = conversation_repo.add_message(
        session=db_session,
        conversation_id=conv.id,
        role="user",
        content="Hello, what is Yassir's profile?",
    )
    assert m1.sequence_number == 1

    m2 = conversation_repo.add_message(
        session=db_session,
        conversation_id=conv.id,
        role="assistant",
        content="Yassir is a software engineer.",
    )
    assert m2.sequence_number == 2

    # 3. Add Summary
    summary = conversation_repo.save_summary(
        session=db_session,
        conversation_id=conv.id,
        summary_text="User asked about candidate profile. Assistant confirmed software engineer.",
        covered_until_sequence=2,
    )
    assert summary.covered_until_sequence == 2

    # 4. Read & Verify
    retrieved = conversation_repo.get_conversation(db_session, conv.id)
    assert retrieved is not None
    messages = conversation_repo.get_all_ordered_messages(db_session, conv.id)
    assert len(messages) == 2

    # 5. Delete conversation & verify cascade
    success = conversation_repo.delete_conversation(db_session, conv.id)
    assert success is True

    # Check child records deleted
    remaining_msgs = db_session.query(ConversationMessage).filter(ConversationMessage.conversation_id == conv.id).all()
    assert len(remaining_msgs) == 0
    remaining_sums = db_session.query(ConversationSummary).filter(ConversationSummary.conversation_id == conv.id).all()
    assert len(remaining_sums) == 0


def test_message_turn_validation_and_ordering(db_session: Session):
    conv = conversation_repo.create_conversation(db_session, title="Validation Test")
    try:
        # Invalid first role (assistant first is rejected)
        with pytest.raises(ValueError, match="first message"):
            from app.rag.conversation.history_manager import history_manager
            history_manager.validate_turn_order("assistant", None)

        # Invalid role name
        with pytest.raises(ValueError, match="Invalid message role"):
            from app.rag.conversation.history_manager import history_manager
            history_manager.validate_incoming_role("system_admin")

        # Empty content
        with pytest.raises(ValueError, match="cannot be empty"):
            from app.rag.conversation.history_manager import history_manager
            history_manager.sanitize_message_content("   ")

    finally:
        conversation_repo.delete_conversation(db_session, conv.id)


def test_memory_strategy_sliding_window(db_session: Session):
    conv = conversation_repo.create_conversation(db_session, title="Sliding Window Test")
    try:
        # Create 3 turns (6 messages)
        for i in range(1, 4):
            conversation_repo.add_message(db_session, conv.id, "user", f"Question {i} about engineering.")
            conversation_repo.add_message(db_session, conv.id, "assistant", f"Answer {i} detailing technologies.")

        messages = conversation_repo.get_all_ordered_messages(db_session, conv.id)
        assert len(messages) == 6

        strat = memory_registry.get_strategy("sliding_window")

        # Test window of 1 turn (should select only last user + assistant)
        res_1 = strat.select_memory(
            messages=messages,
            summary=None,
            config=MemoryConfig(sliding_window_turns=1, include_assistant_replies=True, max_memory_tokens=1000),
        )
        assert len(res_1.included_messages) == 2
        assert res_1.included_messages[0].content == "Question 3 about engineering."
        assert res_1.included_messages[1].content == "Answer 3 detailing technologies."
        assert len(res_1.excluded_messages) == 4

        # Test assistant replies excluded
        res_no_asst = strat.select_memory(
            messages=messages,
            summary=None,
            config=MemoryConfig(sliding_window_turns=2, include_assistant_replies=False, max_memory_tokens=1000),
        )
        assert all(m.role == "user" for m in res_no_asst.included_messages)
        assert len(res_no_asst.included_messages) == 2

    finally:
        conversation_repo.delete_conversation(db_session, conv.id)


def test_memory_strategy_token_bounded(db_session: Session):
    conv = conversation_repo.create_conversation(db_session, title="Token Bounded Test")
    try:
        # Add messages with known content lengths
        conversation_repo.add_message(db_session, conv.id, "user", "Short question one.")
        conversation_repo.add_message(db_session, conv.id, "assistant", "Short answer one.")
        conversation_repo.add_message(db_session, conv.id, "user", "A very long question number two with multiple paragraphs and descriptions that consume significant tokens.")
        conversation_repo.add_message(db_session, conv.id, "assistant", "A very long detailed technical explanation describing microservices, databases, and continuous integration pipelines.")

        messages = conversation_repo.get_all_ordered_messages(db_session, conv.id)
        strat = memory_registry.get_strategy("token_bounded")

        # Budget large enough for everything
        res_large = strat.select_memory(
            messages=messages,
            summary=None,
            config=MemoryConfig(max_memory_tokens=1000),
            total_context_budget=2000,
            prompt_and_retrieval_tokens=500,
        )
        assert len(res_large.included_messages) == 4

        # Budget constrained to fit only the last message
        res_small = strat.select_memory(
            messages=messages,
            summary=None,
            config=MemoryConfig(max_memory_tokens=25),
            total_context_budget=500,
            prompt_and_retrieval_tokens=475,
        )
        assert len(res_small.included_messages) <= 1
        assert len(res_small.excluded_messages) >= 3

    finally:
        conversation_repo.delete_conversation(db_session, conv.id)


def test_history_aware_query_rewriting():
    from app.rag.conversation.schemas import ConversationMessageResponse

    # 1. Standalone question: should NOT be rewritten
    history = [
        ConversationMessageResponse(
            id=uuid.uuid4(),
            conversation_id=uuid.uuid4(),
            role="user",
            content="Hello!",
            sequence_number=1,
            metadata_json={},
            created_at="2026-10-16T10:00:00Z",
        )
    ]
    res_standalone = query_rewriter.rewrite_query(
        query="What programming languages does Yassir use?",
        history=history,
        llm_provider="mock",
    )
    assert res_standalone.is_standalone is True
    assert res_standalone.is_rewritten is False
    assert res_standalone.rewritten_query == "What programming languages does Yassir use?"

    # 2. Contextual follow-up with pronoun "he"
    history_context = [
        ConversationMessageResponse(
            id=uuid.uuid4(),
            conversation_id=uuid.uuid4(),
            role="user",
            content="Tell me about Yassir's profile at Miacorp.",
            sequence_number=1,
            metadata_json={},
            created_at="2026-10-16T10:00:00Z",
        ),
        ConversationMessageResponse(
            id=uuid.uuid4(),
            conversation_id=uuid.uuid4(),
            role="assistant",
            content="At Miacorp, he built APIs in Flask and React with OAuth 2.0.",
            sequence_number=2,
            metadata_json={},
            created_at="2026-10-16T10:00:01Z",
        ),
    ]

    res_followup = query_rewriter.rewrite_query(
        query="Where did he study?",
        history=history_context,
        llm_provider="mock",
    )
    assert res_followup.is_rewritten is True
    assert "Yassir" in res_followup.rewritten_query

    # 3. Contextual follow-up: "Which of them was used for authentication?"
    res_auth = query_rewriter.rewrite_query(
        query="Which of them was used for authentication?",
        history=history_context,
        llm_provider="mock",
    )
    assert res_auth.is_rewritten is True
    assert "authentication" in res_auth.rewritten_query.lower()
    assert "miacorp" in res_auth.rewritten_query.lower()


def test_conversational_rag_orchestrator_multi_turn(db_session: Session):
    conv = conversation_repo.create_conversation(
        session=db_session,
        title="Multi-Turn Test",
        settings={
            "memory_strategy": "sliding_window",
            "retrieval_strategy": "hybrid_rrf",
            "llm_provider": "mock",
            "enable_query_rewriting": True,
        },
    )
    try:
        # Turn 1
        turn_1_req = TurnExecutionRequest(
            message="What technologies did Yassir use during his Miacorp internship?"
        )
        resp_1 = conversational_orchestrator.execute_turn(
            session=db_session,
            conversation_id=conv.id,
            request=turn_1_req,
        )
        assert resp_1.user_message.content == turn_1_req.message
        assert resp_1.assistant_message.role == "assistant"
        assert len(resp_1.sources) > 0
        assert resp_1.assistant_message.sequence_number == 2

        # Turn 2: Follow-up using referring expression
        turn_2_req = TurnExecutionRequest(
            message="Which of them was used for authentication?"
        )
        resp_2 = conversational_orchestrator.execute_turn(
            session=db_session,
            conversation_id=conv.id,
            request=turn_2_req,
        )
        assert resp_2.rewritten_query_info.is_rewritten is True
        assert "authentication" in resp_2.rewritten_query_info.rewritten_query.lower()
        assert resp_2.assistant_message.sequence_number == 4
        assert len(resp_2.sources) > 0

    finally:
        conversation_repo.delete_conversation(db_session, conv.id)


def test_conversational_api_flow(client: TestClient):
    # 1. Create conversation via API
    create_resp = client.post("/api/v1/conversations", json={
        "title": "API Multi-Turn Test",
        "settings": {"memory_strategy": "sliding_window", "llm_provider": "mock"}
    })
    assert create_resp.status_code == 201
    conv_data = create_resp.json()
    conv_id = conv_data["id"]

    try:
        # 2. List memory strategies
        strat_resp = client.get("/api/v1/conversations/memory-strategies")
        assert strat_resp.status_code == 200
        strategies = strat_resp.json()
        assert len(strategies) >= 4

        # 3. Post synchronous message
        msg_resp = client.post(f"/api/v1/conversations/{conv_id}/messages", json={
            "message": "Which engineering degree did Yassir earn?"
        })
        assert msg_resp.status_code == 200
        turn_data = msg_resp.json()
        assert turn_data["user_message"]["sequence_number"] == 1
        assert turn_data["assistant_message"]["sequence_number"] == 2
        assert len(turn_data["sources"]) > 0

        # 4. Preview rewrite
        prev_resp = client.post(
            f"/api/v1/conversations/{conv_id}/rewrite-preview?query=Where+did+he+study%3F"
        )
        assert prev_resp.status_code == 200
        prev_data = prev_resp.json()
        assert prev_data["is_rewritten"] is True

        # 5. Inspect memory
        mem_resp = client.get(f"/api/v1/conversations/{conv_id}/memory")
        assert mem_resp.status_code == 200
        mem_data = mem_resp.json()
        assert len(mem_data["included_messages"]) == 2

        # 6. Compare strategies
        comp_resp = client.post(
            f"/api/v1/conversations/{conv_id}/compare-strategies?query=Where+did+he+study%3F"
        )
        assert comp_resp.status_code == 200
        comp_data = comp_resp.json()
        assert len(comp_data) == 4

    finally:
        # Clean up
        del_resp = client.delete(f"/api/v1/conversations/{conv_id}")
        assert del_resp.status_code == 204


def test_conversational_benchmark_scenarios(db_session: Session):
    results = conversational_benchmark_runner.run_benchmark(
        session=db_session,
        memory_strategy="sliding_window",
        retrieval_strategy="hybrid_rrf",
        llm_provider="mock",
        top_k=5,
    )
    assert len(results) == 12
    passed = [r for r in results if r.passed]
    assert len(passed) >= 9  # At least 75% pass rate across synthetic edge cases
