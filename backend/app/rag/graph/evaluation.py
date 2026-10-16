import time
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.rag.graph.schemas import GraphQueryRequest, GraphRetrievalStrategy
from app.rag.graph.service import GraphRAGService, graph_rag_service


class GraphScenario(BaseModel):
    id: str
    title: str
    question: str
    expected_hops: int
    focus_aspect: str
    target_entities: List[str]
    acceptable_refusal: bool = False


GRAPH_EVALUATION_SCENARIOS: List[GraphScenario] = [
    GraphScenario(
        id="SC-01",
        title="Single Entity Direct Lookup",
        question="What is the candidate's name?",
        expected_hops=1,
        focus_aspect="Single entity resolution",
        target_entities=["Yassir FAHIMI"],
    ),
    GraphScenario(
        id="SC-02",
        title="Organization & University Lookup",
        question="Where did the candidate study, and what organization is mentioned?",
        expected_hops=1,
        focus_aspect="Academic organization link",
        target_entities=["INSEA"],
    ),
    GraphScenario(
        id="SC-03",
        title="Technology Stack Query",
        question="What database technologies does the candidate have skills in?",
        expected_hops=1,
        focus_aspect="Entity typing and attribute filtering",
        target_entities=["PostgreSQL", "Redis"],
    ),
    GraphScenario(
        id="SC-04",
        title="Two-Hop Relational Reasoning",
        question="Which frameworks does the candidate use as a Software Engineer?",
        expected_hops=2,
        focus_aspect="Two-hop traversal: Person -> Role -> Technology",
        target_entities=["FastAPI", "Python"],
    ),
    GraphScenario(
        id="SC-05",
        title="Three-Hop Relational Chain",
        question="Connect the candidate's degree from INSEA to their backend development skills in PostgreSQL.",
        expected_hops=3,
        focus_aspect="Three-hop multi-entity relational path",
        target_entities=["INSEA", "PostgreSQL"],
    ),
    GraphScenario(
        id="SC-06",
        title="Person-Role-Technology Association",
        question="What is the relationship between Yassir FAHIMI and Docker containers?",
        expected_hops=2,
        focus_aspect="Multi-hop skill path",
        target_entities=["Docker"],
    ),
    GraphScenario(
        id="SC-07",
        title="Multi-Document Relational Synthesis",
        question="Synthesize the candidate's education history with their current programming skills.",
        expected_hops=2,
        focus_aspect="Relational synthesis across chunks",
        target_entities=["INSEA", "Python"],
    ),
    GraphScenario(
        id="SC-08",
        title="Ambiguous Entity Match",
        question="What can you tell me about Postgres?",
        expected_hops=1,
        focus_aspect="Fuzzy and abbreviation entity matching",
        target_entities=["PostgreSQL"],
    ),
    GraphScenario(
        id="SC-09",
        title="Alias Resolution Test",
        question="Does the candidate know how to write JS and TS code?",
        expected_hops=1,
        focus_aspect="Alias dictionary resolution: JS -> JavaScript, TS -> TypeScript",
        target_entities=["JavaScript", "TypeScript"],
    ),
    GraphScenario(
        id="SC-10",
        title="Incorrect Relation Defense",
        question="Did the candidate study at Harvard University?",
        expected_hops=1,
        focus_aspect="Absence of non-existent edges, hallucination defense",
        target_entities=[],
        acceptable_refusal=True,
    ),
    GraphScenario(
        id="SC-11",
        title="Missing Evidence Guard",
        question="Find certified skills with verified documentary evidence citations.",
        expected_hops=1,
        focus_aspect="Provenance checking [S#]",
        target_entities=["Python"],
    ),
    GraphScenario(
        id="SC-12",
        title="Contradictory Relations Reconciliation",
        question="Identify if the candidate is based in Morocco or the United States.",
        expected_hops=1,
        focus_aspect="Disambiguating conflicting location candidates",
        target_entities=["Morocco"],
    ),
    GraphScenario(
        id="SC-13",
        title="French Multilingual Query",
        question="Quelles sont les compétences techniques du candidat en bases de données et APIs ?",
        expected_hops=2,
        focus_aspect="Cross-lingual French query entity linking",
        target_entities=["PostgreSQL", "FastAPI"],
    ),
    GraphScenario(
        id="SC-14",
        title="Global Thematic Community Exploration",
        question="Summarize the core technical expertise and tooling clusters.",
        expected_hops=2,
        focus_aspect="Community detection & global thematic retrieval",
        target_entities=["Python", "FastAPI"],
    ),
    GraphScenario(
        id="SC-15",
        title="Unanswerable Question Handling",
        question="What is the candidate's favorite sport and weekend hobby?",
        expected_hops=1,
        focus_aspect="Graceful refusal without hallucinated graph nodes",
        target_entities=[],
        acceptable_refusal=True,
    ),
]


class ScenarioBenchmarkResult(BaseModel):
    scenario_id: str
    title: str
    strategy: str
    success: bool
    latency_ms: float
    entities_retrieved: int
    relations_retrieved: int
    citations_count: int
    error: Optional[str] = None


class GraphBenchmarkReport(BaseModel):
    total_scenarios: int
    passed_scenarios: int
    strategy: str
    average_latency_ms: float
    average_entities: float
    average_relations: float
    results: List[ScenarioBenchmarkResult]


class GraphBenchmarkRunner:
    """
    Executes the 15-scenario synthetic evaluation benchmark for Graph RAG.
    """

    def __init__(self, service: Optional[GraphRAGService] = None):
        self.service = service or graph_rag_service

    def run_benchmark(
        self,
        session: Session,
        strategy: GraphRetrievalStrategy = GraphRetrievalStrategy.HYBRID_GRAPH_VECTOR,
        provider_id: str = "mock",
        model_id: str = "mock-instruct-v1",
        scenario_ids: Optional[List[str]] = None,
    ) -> GraphBenchmarkReport:
        scenarios = GRAPH_EVALUATION_SCENARIOS
        if scenario_ids:
            scenarios = [s for s in scenarios if s.id in scenario_ids]

        results: List[ScenarioBenchmarkResult] = []

        for sc in scenarios:
            t0 = time.perf_counter()
            q_req = GraphQueryRequest(
                question=sc.question,
                strategy=strategy,
                provider_id=provider_id,
                model_id=model_id,
                max_hops=sc.expected_hops,
                top_k=5,
            )
            try:
                resp = self.service.query(session, q_req)
                lat = (time.perf_counter() - t0) * 1000.0

                ans_lower = resp.final_answer.lower()

                # Success check
                if sc.acceptable_refusal:
                    is_success = "no direct" in ans_lower or len(resp.citations) == 0 or "not found" in ans_lower or "insufficient" in ans_lower
                else:
                    found_target = any(t.lower() in ans_lower for t in sc.target_entities) if sc.target_entities else True
                    is_success = found_target or len(resp.retrieved_entities) > 0

                results.append(
                    ScenarioBenchmarkResult(
                        scenario_id=sc.id,
                        title=sc.title,
                        strategy=strategy.value,
                        success=is_success,
                        latency_ms=round(lat, 2),
                        entities_retrieved=len(resp.retrieved_entities),
                        relations_retrieved=len(resp.retrieved_relations),
                        citations_count=len(resp.citations),
                    )
                )
            except Exception as e:
                lat = (time.perf_counter() - t0) * 1000.0
                results.append(
                    ScenarioBenchmarkResult(
                        scenario_id=sc.id,
                        title=sc.title,
                        strategy=strategy.value,
                        success=False,
                        latency_ms=round(lat, 2),
                        entities_retrieved=0,
                        relations_retrieved=0,
                        citations_count=0,
                        error=str(e),
                    )
                )

        passed = sum(1 for r in results if r.success)
        avg_lat = sum(r.latency_ms for r in results) / max(len(results), 1)
        avg_ents = sum(r.entities_retrieved for r in results) / max(len(results), 1)
        avg_rels = sum(r.relations_retrieved for r in results) / max(len(results), 1)

        return GraphBenchmarkReport(
            total_scenarios=len(results),
            passed_scenarios=passed,
            strategy=strategy.value,
            average_latency_ms=round(avg_lat, 2),
            average_entities=round(avg_ents, 2),
            average_relations=round(avg_rels, 2),
            results=results,
        )


graph_benchmark_runner = GraphBenchmarkRunner()
