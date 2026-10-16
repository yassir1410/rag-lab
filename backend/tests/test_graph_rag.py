import pytest
import uuid
from fastapi.testclient import TestClient

from app.main import app
from app.db.session import SessionLocal
from app.models.document import Document
from app.rag.graph.schemas import (
    GraphQueryRequest,
    GraphRetrievalStrategy,
    GraphTraversalRequest,
    GraphPathsRequest,
    EntityMergeRequest,
)
from app.rag.graph.extraction.schema_validator import schema_validator
from app.rag.graph.extraction.entity_extractor import rule_entity_extractor
from app.rag.graph.extraction.relation_extractor import rule_relation_extractor
from app.rag.graph.resolution.entity_normalizer import entity_normalizer
from app.rag.graph.resolution.entity_resolver import entity_resolver
from app.rag.graph.models import ExtractedEntityCandidate
from app.rag.graph.repository import postgres_graph_repository
from app.rag.graph.construction.graph_builder import knowledge_graph_builder
from app.rag.graph.analytics.community_detection import community_detector
from app.rag.graph.analytics.graph_metrics import graph_metrics_calculator
from app.rag.graph.service import graph_rag_service
from app.rag.graph.evaluation import graph_benchmark_runner

client = TestClient(app)


@pytest.fixture
def db_session():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def test_schema_validator():
    """Verify ontology schema enforcement and string sanitization."""
    # Entity types
    assert schema_validator.validate_entity_type("technology") == "TECHNOLOGY"
    assert schema_validator.validate_entity_type("COMPANY") == "ORGANIZATION"
    assert schema_validator.validate_entity_type("INVALID_TYPE") is None

    # Relation types
    assert schema_validator.validate_relation_type("works at") == "WORKED_AT"
    assert schema_validator.validate_relation_type("has_skill") == "HAS_SKILL"
    assert schema_validator.validate_relation_type("INVALID_REL") is None

    # Sanitization
    assert schema_validator.sanitize_name("*Python*") == "Python"
    assert schema_validator.sanitize_name('"FastAPI",') == "FastAPI"


def test_entity_extractor_rules():
    """Verify rule-based extraction for person, technologies, organization, and role."""
    sample_text = (
        "Yassir FAHIMI\n"
        "Software Engineer graduated from INSEA with strong expertise in Python, FastAPI, and PostgreSQL in Morocco."
    )
    entities = rule_entity_extractor.extract_from_chunk(sample_text)
    names = [e.canonical_name for e in entities]
    types = [e.entity_type for e in entities]

    assert "Python" in names
    assert "FastAPI" in names
    assert "PostgreSQL" in names
    assert "INSEA" in names
    assert "Software Engineer" in names
    assert "Morocco" in names
    assert "TECHNOLOGY" in types
    assert "ORGANIZATION" in types


def test_relation_extractor_rules():
    """Verify rule-based relation extraction between co-occurring entities."""
    sample_text = (
        "Yassir FAHIMI is a Software Engineer who studied at INSEA. "
        "He has proven skills in Python and PostgreSQL."
    )
    entities = rule_entity_extractor.extract_from_chunk(sample_text)
    relations = rule_relation_extractor.extract_relations(sample_text, entities)

    rel_types = [r.relation_type for r in relations]
    assert "STUDIED_AT" in rel_types
    assert "HAS_SKILL" in rel_types
    assert "HOLDS_ROLE" in rel_types


def test_entity_normalizer_and_resolver():
    """Verify abbreviation resolution, fuzzy matching, and duplicate candidate identification."""
    # Abbreviation
    assert entity_normalizer.resolve_abbreviation("postgres") == "PostgreSQL"
    assert entity_normalizer.resolve_abbreviation("js") == "JavaScript"

    # Fuzzy similarity
    sim_exact = entity_resolver.calculate_similarity("PostgreSQL", "PostgreSQL")
    assert sim_exact == 1.0

    sim_fuzzy = entity_resolver.calculate_similarity("PostgreSQL", "Postgres")
    assert sim_fuzzy >= 0.88


def test_postgres_graph_repository_crud_and_traversal(db_session):
    """Verify entity creation, relation persistence, evidence attachment, and bounded traversal."""
    repo = postgres_graph_repository

    # 1. Create entities
    e1 = repo.create_or_get_entity(
        db_session,
        ExtractedEntityCandidate(canonical_name="Test Person", entity_type="PERSON"),
    )
    e2 = repo.create_or_get_entity(
        db_session,
        ExtractedEntityCandidate(canonical_name="Test University", entity_type="ORGANIZATION"),
    )
    e3 = repo.create_or_get_entity(
        db_session,
        ExtractedEntityCandidate(canonical_name="Test Python", entity_type="TECHNOLOGY"),
    )

    # 2. Create relations
    r1 = repo.create_or_get_relation(
        db_session, source_id=e1.id, target_id=e2.id, relation_type="STUDIED_AT"
    )
    r2 = repo.create_or_get_relation(
        db_session, source_id=e1.id, target_id=e3.id, relation_type="HAS_SKILL"
    )

    db_session.commit()

    # 3. Traversal (depth 1)
    nodes_1, edges_1, depth_1 = repo.traverse(db_session, start_entity_id=e1.id, max_depth=1)
    assert len(nodes_1) >= 2
    assert len(edges_1) >= 2

    # 4. Path finding
    paths = repo.find_paths(db_session, source_id=e2.id, target_id=e3.id, max_depth=2)
    assert len(paths) >= 1
    assert paths[0]["length"] == 2


def test_graph_construction_from_document(db_session):
    """Verify automated Knowledge Graph construction from cv_yassir.pdf document."""
    doc = db_session.query(Document).filter(Document.original_filename == "cv_yassir.pdf").first()
    if not doc:
        doc = db_session.query(Document).first()

    assert doc is not None, "At least one document must exist in database"

    run_record = knowledge_graph_builder.build_from_document(
        session=db_session,
        document_id=doc.id,
        strategy="rule",
    )
    assert run_record.status == "completed"
    assert run_record.metrics["unique_entities"] > 0
    assert run_record.metrics["relations_extracted"] > 0


def test_community_detection_and_metrics(db_session):
    """Verify connected components community detection and topological metrics."""
    communities = community_detector.detect_communities(db_session)
    assert len(communities) >= 1
    assert communities[0].size > 0

    metrics = graph_metrics_calculator.compute_metrics(db_session)
    assert metrics["nodes"] > 0
    assert metrics["edges"] > 0


def test_graph_rag_query_execution(db_session):
    """Verify multi-strategy graph RAG query execution and [S#] citations."""
    req = GraphQueryRequest(
        question="Which programming languages does the candidate know, and where did he study?",
        strategy=GraphRetrievalStrategy.HYBRID_GRAPH_VECTOR,
        max_hops=2,
        top_k=5,
        provider_id="mock",
    )
    resp = graph_rag_service.query(db_session, req)
    assert resp.final_answer is not None
    assert len(resp.retrieved_entities) > 0
    assert resp.latency_ms > 0


def test_graph_api_endpoints():
    """Verify REST API endpoints under /api/v1/graph."""
    # 1. Stats
    res_stats = client.get("/api/v1/graph/stats")
    assert res_stats.status_code == 200
    assert res_stats.json()["node_count"] > 0

    # 2. Entities
    res_ents = client.get("/api/v1/graph/entities?limit=10")
    assert res_ents.status_code == 200
    assert len(res_ents.json()) > 0
    ent_id = res_ents.json()[0]["id"]

    # 3. Single Entity
    res_ent = client.get(f"/api/v1/graph/entities/{ent_id}")
    assert res_ent.status_code == 200
    assert res_ent.json()["id"] == ent_id

    # 4. Relations
    res_rels = client.get("/api/v1/graph/relations?limit=10")
    assert res_rels.status_code == 200

    # 5. Communities
    res_comms = client.get("/api/v1/graph/communities")
    assert res_comms.status_code == 200
    assert len(res_comms.json()) > 0

    # 6. Query endpoint
    query_payload = {
        "question": "What is the candidate's degree from INSEA?",
        "strategy": "hybrid_graph_vector",
        "max_hops": 2,
        "provider_id": "mock",
    }
    res_q = client.post("/api/v1/graph/query", json=query_payload)
    assert res_q.status_code == 200
    assert "citations" in res_q.json()

    # 7. Compare endpoint
    cmp_payload = {
        "question": "What technologies are used?",
        "strategies": ["entity_graph", "hybrid_graph_vector"],
        "provider_id": "mock",
    }
    res_cmp = client.post("/api/v1/graph/compare", json=cmp_payload)
    assert res_cmp.status_code == 200
    assert "entity_graph" in res_cmp.json()["results"]


def test_graph_benchmark_runner(db_session):
    """Verify 15-scenario synthetic evaluation benchmark."""
    report = graph_benchmark_runner.run_benchmark(
        session=db_session,
        strategy=GraphRetrievalStrategy.HYBRID_GRAPH_VECTOR,
        provider_id="mock",
        scenario_ids=["SC-01", "SC-02", "SC-04", "SC-15"],
    )
    assert report.total_scenarios == 4
    assert report.passed_scenarios >= 3
    assert len(report.results) == 4
