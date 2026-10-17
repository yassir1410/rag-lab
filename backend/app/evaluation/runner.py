import time
import math
import uuid
import logging
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, AsyncIterator, Set
from sqlalchemy.orm import Session

from app.db.session import SessionLocal
from app.models.evaluation import (
    EvaluationDataset,
    EvaluationQuestion,
    EvaluationRun,
    EvaluationItemResult,
    EvaluationTrace,
)
from app.rag.retrieval.service import retrieval_service
from app.rag.retrieval.schemas import SearchRequest
from app.rag.generation.orchestrator import RAGOrchestrator
from app.rag.generation.schemas import RAGQueryRequest
from app.rag.unified.generation_service import unified_generation_service
from app.rag.unified.schemas import UnifiedRAGQueryRequest
from app.evaluation.retrieval.metrics import (
    precision_at_k,
    recall_at_k,
    f1_at_k,
    hit_rate_at_k,
    reciprocal_rank,
    mean_reciprocal_rank,
    average_precision,
    mean_average_precision,
    ndcg_at_k,
    evaluate_unanswerable_metrics,
)
from app.evaluation.generation.faithfulness import faithfulness_evaluator
from app.evaluation.generation.answer_relevance import answer_relevance_evaluator
from app.evaluation.generation.citation_evaluator import citation_evaluator
from app.evaluation.generation.claim_extractor import claim_extractor
from app.evaluation.generation.accuracy import compute_generation_accuracy
from app.evaluation.routing.routing_metrics import (
    route_accuracy,
    evaluate_unnecessary_vision,
    evaluate_unnecessary_agentic,
    evaluate_document_isolation,
)
from app.evaluation.tracing import EvaluationTracer
from app.evaluation.cost import pricing_registry

logger = logging.getLogger(__name__)


def _calculate_percentile(values: List[float], percentile: float) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    k = (len(sorted_vals) - 1) * (percentile / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_vals[int(k)]
    d0 = sorted_vals[int(f)] * (c - k)
    d1 = sorted_vals[int(c)] * (k - f)
    return round(d0 + d1, 2)


class EvaluationRunner:
    """
    End-to-End Evaluation & Observability Orchestration Engine.
    Executes Golden Datasets against different RAG, retrieval, and unified routing configurations.
    Evaluates:
    1. Retrieval quality (P@K, R@K, F1@K, Hit Rate, MRR, MAP, nDCG)
    2. Generation accuracy (Exact Match, Token F1, Normalized Match, Numerical Correctness)
    3. Faithfulness and citation support (Claim-level entailment, ID validity, precision, coverage)
    4. Routing correctness (Route accuracy, unnecessary vision/agentic, document isolation)
    5. Operational performance & latency percentiles
    6. Token usage and versioned cost accounting
    7. Structured trace persistence and error behavior
    """

    def __init__(self):
        self.orchestrator = RAGOrchestrator()
        self.unified_service = unified_generation_service
        self._cancelled_runs: Set[str] = set()

    def cancel_run(self, run_id: uuid.UUID) -> bool:
        self._cancelled_runs.add(str(run_id))
        return True

    def is_cancelled(self, run_id: uuid.UUID) -> bool:
        return str(run_id) in self._cancelled_runs

    def run_evaluation_background(self, run_id: uuid.UUID) -> None:
        """
        Background execution entry point with dedicated DB session lifecycle.
        Ensures thread safety and prevents connection leaks.
        """
        db = SessionLocal()
        try:
            self.run_evaluation(db, run_id)
        except Exception as e:
            logger.exception(f"Background evaluation run '{run_id}' encountered an unhandled error: {e}")
            try:
                run = db.query(EvaluationRun).filter(EvaluationRun.id == run_id).first()
                if run and run.status == "running":
                    run.status = "failed"
                    run.error_message = str(e)
                    db.commit()
            except Exception:
                pass
        finally:
            db.close()

    def run_evaluation(
        self,
        session: Session,
        run_id: uuid.UUID,
    ) -> EvaluationRun:
        run = session.query(EvaluationRun).filter(EvaluationRun.id == run_id).first()
        if not run:
            raise ValueError(f"Evaluation run '{run_id}' not found.")

        dataset = session.query(EvaluationDataset).filter(EvaluationDataset.id == run.dataset_id).first()
        if not dataset:
            run.status = "failed"
            run.error_message = "Evaluation dataset associated with this run does not exist."
            session.commit()
            return run

        config = run.config or {}
        max_questions = config.get("max_questions")
        questions_query = (
            session.query(EvaluationQuestion)
            .filter(EvaluationQuestion.dataset_id == dataset.id)
            .order_by(EvaluationQuestion.created_at)
        )
        if max_questions and max_questions > 0:
            questions_query = questions_query.limit(max_questions)

        questions = questions_query.all()
        if not questions:
            run.status = "failed"
            run.error_message = "Evaluation dataset contains no questions to evaluate."
            session.commit()
            return run

        run.status = "running"
        run.progress_total = len(questions)
        run.progress_current = 0

        # Snapshot experiment configuration for reproducibility
        run.experiment_snapshot = {
            "dataset_id": str(dataset.id),
            "dataset_name": dataset.name,
            "dataset_version": getattr(dataset, "version", "1.0"),
            "run_type": run.run_type or "end_to_end",
            "git_revision": "head",
            "pricing_version": pricing_registry.default_version,
            "config": config,
        }
        session.commit()

        run_type = run.run_type or "end_to_end"
        embedding_model = config.get("embedding_model_id", "sentence-transformers/all-MiniLM-L6-v2")
        search_algo = config.get("search_algorithm", "exact")
        distance_metric = config.get("distance_metric", "cosine")
        top_k = int(config.get("top_k", 5))
        ef_search = int(config.get("ef_search", 40))
        probes = int(config.get("probes", 1))

        retrieval_strategy = config.get("retrieval_strategy", "dense_only")
        hybrid_alpha = float(config.get("hybrid_alpha", 0.5))
        rrf_k = int(config.get("rrf_k", 60))
        reranker_model = config.get("reranker_model", "none")
        enable_query_expansion = bool(config.get("enable_query_expansion", False))
        enable_neighbor_expansion = bool(config.get("enable_neighbor_expansion", False))

        llm_provider = config.get("llm_provider", "mock")
        llm_model = config.get("llm_model", "mock-instruct-v1")
        prompt_template = config.get("prompt_template", "grounded")
        temperature = float(config.get("temperature", 0.0))
        max_context_tokens = int(config.get("max_context_tokens", 2000))
        max_output_tokens = int(config.get("max_output_tokens", 512))

        judge_provider = config.get("judge_provider", "heuristic")
        judge_model = config.get("judge_model")

        all_item_results: List[EvaluationItemResult] = []
        p_at_k_vals: List[float] = []
        r_at_k_vals: List[float] = []
        f1_at_k_vals: List[float] = []
        hit_vals: List[float] = []
        rr_vals: List[float] = []
        ap_vals: List[float] = []
        ndcg_vals: List[float] = []

        em_vals: List[float] = []
        token_f1_vals: List[float] = []
        norm_match_vals: List[float] = []

        route_acc_vals: List[float] = []
        unnecessary_vis_vals: List[float] = []
        unnecessary_agent_vals: List[float] = []
        doc_leakage_vals: List[float] = []

        faith_vals: List[float] = []
        relevance_vals: List[float] = []
        cite_id_vals: List[float] = []
        cite_support_vals: List[float] = []
        cite_cov_vals: List[float] = []

        correct_refusal_vals: List[float] = []
        false_answer_vals: List[float] = []
        over_refusal_vals: List[float] = []
        unsupported_assertion_vals: List[float] = []

        latencies_ms: List[float] = []
        total_in_tokens = 0
        total_out_tokens = 0
        total_ctx_tokens = 0
        total_judge_tokens = 0
        total_cost_usd = 0.0

        for idx, q in enumerate(questions, start=1):
            if self.is_cancelled(run.id):
                run.status = "cancelled"
                break

            t_start = time.perf_counter()
            relevant_chunk_ids_set = {str(c) for c in (q.relevant_chunk_ids or [])}
            relevance_scores_map = q.relevance_labels or {str(c): 1.0 for c in (q.relevant_chunk_ids or [])}

            tracer = EvaluationTracer(
                run_id=run.id,
                request_id=f"eval-{run.id}-{idx}",
                provider=llm_provider,
                model=llm_model,
                document_ids=getattr(q, "document_scope", []) or (q.relevant_document_ids or []),
            )

            try:
                if run_type == "retrieval_only":
                    # 1. Retrieval-Only Evaluation
                    with tracer.stage("retrieval", {"strategy": retrieval_strategy, "top_k": top_k}):
                        s_req = SearchRequest(
                            query=q.question,
                            strategy=retrieval_strategy,
                            model_id=embedding_model,
                            search_algorithm=search_algo,
                            distance_metric=distance_metric,
                            top_k=top_k,
                            hybrid_alpha=hybrid_alpha,
                            rrf_k=rrf_k,
                            reranker_model=reranker_model,
                            enable_query_expansion=enable_query_expansion,
                            enable_neighbor_expansion=enable_neighbor_expansion,
                            ef_search=ef_search,
                            probes=probes,
                            explain_plan=False,
                        )
                        s_resp = retrieval_service.search(session, s_req)
                        retrieved_ids = [str(r.chunk_id) for r in s_resp.results]

                    p_k = precision_at_k(retrieved_ids, relevant_chunk_ids_set, k=top_k)
                    r_k = recall_at_k(retrieved_ids, relevant_chunk_ids_set, k=top_k)
                    f1_k = f1_at_k(retrieved_ids, relevant_chunk_ids_set, k=top_k)
                    hit = hit_rate_at_k(retrieved_ids, relevant_chunk_ids_set, k=top_k)
                    rr = reciprocal_rank(retrieved_ids, relevant_chunk_ids_set, k=top_k)
                    ap = average_precision(retrieved_ids, relevant_chunk_ids_set, k=top_k)
                    ndcg = ndcg_at_k(retrieved_ids, relevance_scores_map, k=top_k)

                    total_lat = (time.perf_counter() - t_start) * 1000.0

                    retrieval_metrics_dict = {
                        f"precision@{top_k}": round(p_k, 4),
                        f"recall@{top_k}": round(r_k, 4),
                        f"f1@{top_k}": round(f1_k, 4),
                        f"hit_rate@{top_k}": round(hit, 4),
                        "reciprocal_rank": round(rr, 4),
                        "average_precision": round(ap, 4),
                        f"ndcg@{top_k}": round(ndcg, 4),
                    }

                    p_at_k_vals.append(p_k)
                    r_at_k_vals.append(r_k)
                    f1_at_k_vals.append(f1_k)
                    hit_vals.append(hit)
                    rr_vals.append(rr)
                    ap_vals.append(ap)
                    ndcg_vals.append(ndcg)
                    latencies_ms.append(total_lat)

                    item_res = EvaluationItemResult(
                        run_id=run.id,
                        question_id=q.id,
                        question_text=q.question,
                        reference_answer=q.reference_answer,
                        generated_answer=None,
                        retrieved_chunk_ids=retrieved_ids,
                        relevant_chunk_ids=list(relevant_chunk_ids_set),
                        retrieval_metrics=retrieval_metrics_dict,
                        faithfulness_score=None,
                        answer_relevance_score=None,
                        citation_metrics={},
                        claim_judgments=[],
                        routing_metrics={},
                        accuracy_metrics={},
                        latency_breakdown={
                            "embedding_latency_ms": round(s_resp.embedding_latency_ms, 2),
                            "retrieval_latency_ms": round(s_resp.query_latency_ms, 2),
                            "total_latency_ms": round(total_lat, 2),
                        },
                        token_breakdown={},
                        provider_reported_tokens={},
                        measurement_source="measured",
                        estimated_cost_usd=0.0,
                        trace_id=tracer.trace_id,
                        status="success",
                    )
                    session.add(item_res)
                    session.flush()

                    tracer.item_result_id = item_res.id
                    tracer.persist(session)
                    all_item_results.append(item_res)

                elif run_type == "unified":
                    # 2. Unified RAG with Intelligent Routing Evaluation
                    doc_id_target = None
                    doc_scope = getattr(q, "document_scope", []) or (q.relevant_document_ids or [])
                    if doc_scope:
                        try:
                            doc_id_target = uuid.UUID(str(doc_scope[0]))
                        except Exception:
                            doc_id_target = None

                    u_req = UnifiedRAGQueryRequest(
                        question=q.question,
                        document_id=doc_id_target,
                        document_ids=[uuid.UUID(str(d)) for d in doc_scope] if doc_scope else None,
                        mode=config.get("routing_mode", "auto"),
                        manual_routes=config.get("manual_routes"),
                        llm_provider=llm_provider,
                        llm_model=llm_model,
                        top_k=top_k,
                        max_context_tokens=max_context_tokens,
                        max_output_tokens=max_output_tokens,
                        temperature=temperature,
                    )

                    u_resp = self.unified_service.execute_query(session, u_req)
                    tracer.set_routes(u_resp.routing.selected_routes)
                    tracer.set_model(llm_provider, u_resp.model_used)

                    # Record timing stages into tracer
                    tracer.record_stage("routing", u_resp.timing_metrics.routing_latency_ms, {"routes": u_resp.routing.selected_routes, "mode": u_resp.routing.mode})
                    tracer.record_stage("retrieval", u_resp.timing_metrics.retrieval_latency_ms, {"per_route_ms": u_resp.timing_metrics.per_route_latency_ms, "evidence_count": len(u_resp.evidence_items)})
                    tracer.record_stage("context_assembly", u_resp.timing_metrics.fusion_latency_ms)
                    tracer.record_stage("generation", u_resp.timing_metrics.generation_latency_ms, {"model": u_resp.model_used, "is_vision_enabled": u_resp.is_vision_enabled})
                    tracer.record_stage("citation_validation", u_resp.timing_metrics.citation_validation_ms, {"citations": len(u_resp.citations), "invalid": len(u_resp.invalid_citations)})

                    # Routing Metrics
                    expected_routes = getattr(q, "expected_routes", []) or ["text_dense"]
                    is_vis_req = getattr(q, "modality", "text") == "visual"
                    is_agent_req = getattr(q, "modality", "text") == "agentic"

                    selected_first_route = u_resp.routing.selected_routes[0] if u_resp.routing.selected_routes else "unknown"
                    r_acc = route_accuracy(selected_first_route, expected_routes)
                    route_acc_vals.append(r_acc)

                    vis_eval = evaluate_unnecessary_vision(
                        is_visual_required=is_vis_req,
                        is_vision_enabled=u_resp.is_vision_enabled,
                        images_attached_count=1 if u_resp.is_vision_enabled else 0,
                    )
                    unnecessary_vis_vals.append(1.0 if vis_eval["unnecessary_vision"] else 0.0)

                    agent_eval = evaluate_unnecessary_agentic(
                        is_agentic_required=is_agent_req,
                        selected_route=selected_first_route,
                    )
                    unnecessary_agent_vals.append(1.0 if agent_eval["unnecessary_agentic"] else 0.0)

                    retrieved_doc_ids = [str(it.document_id) for it in u_resp.evidence_items if it.document_id]
                    doc_iso = evaluate_document_isolation(
                        target_document_ids=doc_scope,
                        retrieved_document_ids=retrieved_doc_ids,
                    )
                    doc_leakage_vals.append(doc_iso["leakage_rate"])

                    routing_metrics_dict = {
                        "route_accuracy": r_acc,
                        "selected_routes": u_resp.routing.selected_routes,
                        "expected_routes": expected_routes,
                        "unnecessary_vision": vis_eval["unnecessary_vision"],
                        "unnecessary_agentic": agent_eval["unnecessary_agentic"],
                        "document_isolation": doc_iso["is_isolated"],
                        "cross_document_leakage_rate": doc_iso["leakage_rate"],
                    }

                    # Retrieval metrics over unified evidence items
                    retrieved_ids = [str(it.evidence_id) for it in u_resp.evidence_items]
                    p_k = precision_at_k(retrieved_ids, relevant_chunk_ids_set, k=top_k)
                    r_k = recall_at_k(retrieved_ids, relevant_chunk_ids_set, k=top_k)
                    f1_k = f1_at_k(retrieved_ids, relevant_chunk_ids_set, k=top_k)
                    hit = hit_rate_at_k(retrieved_ids, relevant_chunk_ids_set, k=top_k)
                    rr = reciprocal_rank(retrieved_ids, relevant_chunk_ids_set, k=top_k)
                    ap = average_precision(retrieved_ids, relevant_chunk_ids_set, k=top_k)
                    ndcg = ndcg_at_k(retrieved_ids, relevance_scores_map, k=top_k)

                    p_at_k_vals.append(p_k)
                    r_at_k_vals.append(r_k)
                    f1_at_k_vals.append(f1_k)
                    hit_vals.append(hit)
                    rr_vals.append(rr)
                    ap_vals.append(ap)
                    ndcg_vals.append(ndcg)

                    # Generation accuracy metrics
                    acceptable = getattr(q, "acceptable_answers", []) or []
                    num_spec = getattr(q, "numerical_answer", None)
                    acc_metrics = compute_generation_accuracy(
                        predicted=u_resp.answer,
                        reference=q.reference_answer,
                        acceptable_answers=acceptable,
                        numerical_spec=num_spec,
                    )
                    em_vals.append(acc_metrics["exact_match"])
                    token_f1_vals.append(acc_metrics["token_f1"])
                    norm_match_vals.append(acc_metrics["normalized_answer_match"])

                    # Faithfulness & Grounding
                    faith_res = faithfulness_evaluator.evaluate(
                        generated_answer=u_resp.answer,
                        context_passages=[it.content for it in u_resp.evidence_items],
                        judge_provider=judge_provider,
                        judge_model=judge_model,
                    )
                    faith_vals.append(faith_res.score)

                    rel_res = answer_relevance_evaluator.evaluate(
                        question=q.question,
                        generated_answer=u_resp.answer,
                        reference_answer=q.reference_answer,
                        is_answerable=q.is_answerable,
                    )
                    relevance_vals.append(rel_res.score)

                    # Citation quality
                    total_citations = len(u_resp.citations) + len(u_resp.invalid_citations)
                    cite_id_validity = len(u_resp.citations) / total_citations if total_citations > 0 else 1.0
                    cite_id_vals.append(cite_id_validity)
                    cite_support_vals.append(1.0 if not u_resp.invalid_citations else 0.0)
                    cite_cov_vals.append(1.0 if total_citations > 0 else 0.0)

                    # Refusal / unanswerable
                    is_ref = claim_extractor.is_refusal(u_resp.answer)
                    u_metrics = evaluate_unanswerable_metrics(
                        is_answerable=q.is_answerable,
                        is_refusal=is_ref,
                        total_claims=faith_res.total_claims,
                        supported_claims=faith_res.supported_claims,
                        unsupported_claims=faith_res.unsupported_claims,
                    )
                    if not q.is_answerable:
                        correct_refusal_vals.append(u_metrics["correct_refusal"])
                        false_answer_vals.append(u_metrics["false_answer"])
                    else:
                        over_refusal_vals.append(u_metrics["over_refusal"])
                    unsupported_assertion_vals.append(u_metrics["unsupported_assertion_rate"])

                    # Tokens & Cost
                    in_tok = len(q.question.split()) * 2
                    out_tok = len(u_resp.answer.split()) * 2
                    total_in_tokens += in_tok
                    total_out_tokens += out_tok

                    cost_calc = pricing_registry.calculate_cost(
                        model_id=u_resp.model_used,
                        input_tokens=in_tok,
                        output_tokens=out_tok,
                    )
                    total_cost_usd += (cost_calc.total_estimated_cost_usd or 0.0)

                    total_lat = (time.perf_counter() - t_start) * 1000.0
                    latencies_ms.append(total_lat)

                    item_res = EvaluationItemResult(
                        run_id=run.id,
                        question_id=q.id,
                        question_text=q.question,
                        reference_answer=q.reference_answer,
                        generated_answer=u_resp.answer,
                        retrieved_chunk_ids=retrieved_ids,
                        relevant_chunk_ids=list(relevant_chunk_ids_set),
                        retrieval_metrics={
                            f"precision@{top_k}": round(p_k, 4),
                            f"recall@{top_k}": round(r_k, 4),
                            f"f1@{top_k}": round(f1_k, 4),
                            f"hit_rate@{top_k}": round(hit, 4),
                            "reciprocal_rank": round(rr, 4),
                            "average_precision": round(ap, 4),
                            f"ndcg@{top_k}": round(ndcg, 4),
                        },
                        faithfulness_score=faith_res.score,
                        answer_relevance_score=rel_res.score,
                        citation_metrics={
                            "citation_id_validity": round(cite_id_validity, 4),
                            "valid_citations_count": len(u_resp.citations),
                            "invalid_citations_count": len(u_resp.invalid_citations),
                            "uncited_sources_count": len(u_resp.uncited_sources),
                        },
                        claim_judgments=[c.model_dump(mode="json") for c in faith_res.claims],
                        routing_metrics=routing_metrics_dict,
                        accuracy_metrics=acc_metrics,
                        latency_breakdown={
                            "routing_ms": round(u_resp.timing_metrics.routing_latency_ms, 2),
                            "retrieval_ms": round(u_resp.timing_metrics.retrieval_latency_ms, 2),
                            "context_assembly_ms": round(u_resp.timing_metrics.fusion_latency_ms, 2),
                            "generation_ms": round(u_resp.timing_metrics.generation_latency_ms, 2),
                            "citation_validation_ms": round(u_resp.timing_metrics.citation_validation_ms, 2),
                            "total_latency_ms": round(total_lat, 2),
                        },
                        token_breakdown={
                            "prompt_tokens": in_tok,
                            "output_tokens": out_tok,
                            "total_tokens": in_tok + out_tok,
                        },
                        provider_reported_tokens={},
                        measurement_source="mock" if llm_provider == "mock" else "measured",
                        estimated_cost_usd=cost_calc.total_estimated_cost_usd or 0.0,
                        trace_id=tracer.trace_id,
                        status="success",
                    )
                    session.add(item_res)
                    session.flush()

                    tracer.item_result_id = item_res.id
                    tracer.persist(session)
                    all_item_results.append(item_res)

                else:
                    # 3. Standard End-to-End RAG Evaluation (Phase 06)
                    with tracer.stage("rag_orchestration", {"retrieval_strategy": retrieval_strategy, "top_k": top_k}):
                        rag_req = RAGQueryRequest(
                            question=q.question,
                            retrieval_strategy=retrieval_strategy,
                            embedding_model_id=embedding_model,
                            search_algorithm=search_algo,
                            distance_metric=distance_metric,
                            top_k=top_k,
                            hybrid_alpha=hybrid_alpha,
                            rrf_k=rrf_k,
                            reranker_model=reranker_model,
                            enable_query_expansion=enable_query_expansion,
                            enable_neighbor_expansion=enable_neighbor_expansion,
                            ef_search=ef_search,
                            probes=probes,
                            llm_provider=llm_provider,
                            llm_model=llm_model,
                            prompt_template=prompt_template,
                            temperature=temperature,
                            max_context_tokens=max_context_tokens,
                            max_output_tokens=max_output_tokens,
                            include_assembled_prompt=False,
                        )
                        rag_resp = self.orchestrator.execute_query(session, rag_req)
                        retrieved_ids = [str(s.chunk_id) for s in rag_resp.sources]

                    # Retrieval metrics
                    p_k = precision_at_k(retrieved_ids, relevant_chunk_ids_set, k=top_k)
                    r_k = recall_at_k(retrieved_ids, relevant_chunk_ids_set, k=top_k)
                    f1_k = f1_at_k(retrieved_ids, relevant_chunk_ids_set, k=top_k)
                    hit = hit_rate_at_k(retrieved_ids, relevant_chunk_ids_set, k=top_k)
                    rr = reciprocal_rank(retrieved_ids, relevant_chunk_ids_set, k=top_k)
                    ap = average_precision(retrieved_ids, relevant_chunk_ids_set, k=top_k)
                    ndcg = ndcg_at_k(retrieved_ids, relevance_scores_map, k=top_k)

                    p_at_k_vals.append(p_k)
                    r_at_k_vals.append(r_k)
                    f1_at_k_vals.append(f1_k)
                    hit_vals.append(hit)
                    rr_vals.append(rr)
                    ap_vals.append(ap)
                    ndcg_vals.append(ndcg)

                    # Generation accuracy
                    acceptable = getattr(q, "acceptable_answers", []) or []
                    num_spec = getattr(q, "numerical_answer", None)
                    acc_metrics = compute_generation_accuracy(
                        predicted=rag_resp.answer,
                        reference=q.reference_answer,
                        acceptable_answers=acceptable,
                        numerical_spec=num_spec,
                    )
                    em_vals.append(acc_metrics["exact_match"])
                    token_f1_vals.append(acc_metrics["token_f1"])
                    norm_match_vals.append(acc_metrics["normalized_answer_match"])

                    # Faithfulness
                    t_judge_start = time.perf_counter()
                    faith_res = faithfulness_evaluator.evaluate(
                        generated_answer=rag_resp.answer,
                        context_passages=[s.content for s in rag_resp.sources],
                        judge_provider=judge_provider,
                        judge_model=judge_model,
                    )
                    judge_lat = (time.perf_counter() - t_judge_start) * 1000.0
                    faith_vals.append(faith_res.score)

                    # Answer relevance
                    rel_res = answer_relevance_evaluator.evaluate(
                        question=q.question,
                        generated_answer=rag_resp.answer,
                        reference_answer=q.reference_answer,
                        is_answerable=q.is_answerable,
                    )
                    relevance_vals.append(rel_res.score)

                    # Citation evaluation
                    cite_report = citation_evaluator.evaluate(
                        generated_answer=rag_resp.answer,
                        sources=rag_resp.sources,
                        faithfulness_result=faith_res,
                    )
                    cite_id_vals.append(cite_report.citation_id_validity)
                    cite_support_vals.append(cite_report.citation_support_precision)
                    cite_cov_vals.append(cite_report.citation_coverage)

                    # Unanswerable / refusal
                    is_ref = claim_extractor.is_refusal(rag_resp.answer)
                    u_metrics = evaluate_unanswerable_metrics(
                        is_answerable=q.is_answerable,
                        is_refusal=is_ref,
                        total_claims=faith_res.total_claims,
                        supported_claims=faith_res.supported_claims,
                        unsupported_claims=faith_res.unsupported_claims,
                    )
                    if not q.is_answerable:
                        correct_refusal_vals.append(u_metrics["correct_refusal"])
                        false_answer_vals.append(u_metrics["false_answer"])
                    else:
                        over_refusal_vals.append(u_metrics["over_refusal"])
                    unsupported_assertion_vals.append(u_metrics["unsupported_assertion_rate"])

                    # Tokens & Cost
                    in_tok = rag_resp.token_metrics.total_prompt_tokens
                    out_tok = rag_resp.token_metrics.reserved_output_tokens // 2
                    ctx_tok = rag_resp.token_metrics.context_tokens
                    judge_tok = 250 if judge_provider and judge_provider != "heuristic" else 0

                    cost_calc = pricing_registry.calculate_cost(
                        model_id=rag_resp.llm_model,
                        input_tokens=in_tok,
                        output_tokens=out_tok,
                        context_tokens=ctx_tok,
                        judge_tokens=judge_tok,
                        judge_model_id=judge_model,
                    )

                    total_in_tokens += in_tok
                    total_out_tokens += out_tok
                    total_ctx_tokens += ctx_tok
                    total_judge_tokens += judge_tok
                    total_cost_usd += (cost_calc.total_estimated_cost_usd or 0.0)

                    total_lat = (time.perf_counter() - t_start) * 1000.0
                    latencies_ms.append(total_lat)

                    item_res = EvaluationItemResult(
                        run_id=run.id,
                        question_id=q.id,
                        question_text=q.question,
                        reference_answer=q.reference_answer,
                        generated_answer=rag_resp.answer,
                        retrieved_chunk_ids=retrieved_ids,
                        relevant_chunk_ids=list(relevant_chunk_ids_set),
                        retrieval_metrics={
                            f"precision@{top_k}": round(p_k, 4),
                            f"recall@{top_k}": round(r_k, 4),
                            f"f1@{top_k}": round(f1_k, 4),
                            f"hit_rate@{top_k}": round(hit, 4),
                            "reciprocal_rank": round(rr, 4),
                            "average_precision": round(ap, 4),
                            f"ndcg@{top_k}": round(ndcg, 4),
                        },
                        faithfulness_score=faith_res.score,
                        answer_relevance_score=rel_res.score,
                        citation_metrics=cite_report.model_dump(mode="json"),
                        claim_judgments=[c.model_dump(mode="json") for c in faith_res.claims],
                        routing_metrics={},
                        accuracy_metrics=acc_metrics,
                        latency_breakdown={
                            "embedding_latency_ms": round(rag_resp.timing_metrics.embedding_latency_ms, 2),
                            "retrieval_latency_ms": round(rag_resp.timing_metrics.retrieval_latency_ms, 2),
                            "context_assembly_ms": round(rag_resp.timing_metrics.context_assembly_ms, 2),
                            "prompt_building_ms": round(rag_resp.timing_metrics.prompt_building_ms, 2),
                            "llm_inference_ms": round(rag_resp.timing_metrics.llm_inference_ms, 2),
                            "citation_validation_ms": round(rag_resp.timing_metrics.citation_validation_ms, 2),
                            "judge_latency_ms": round(judge_lat, 2),
                            "total_latency_ms": round(total_lat, 2),
                        },
                        token_breakdown={
                            "prompt_tokens": in_tok,
                            "output_tokens": out_tok,
                            "context_tokens": ctx_tok,
                            "judge_tokens": judge_tok,
                            "total_tokens": in_tok + out_tok + judge_tok,
                        },
                        provider_reported_tokens={},
                        measurement_source="mock" if llm_provider == "mock" else "measured",
                        estimated_cost_usd=cost_calc.total_estimated_cost_usd or 0.0,
                        trace_id=tracer.trace_id,
                        status="success",
                    )
                    session.add(item_res)
                    session.flush()

                    tracer.item_result_id = item_res.id
                    tracer.persist(session)
                    all_item_results.append(item_res)

            except Exception as e:
                logger.exception(f"Error evaluating question '{q.question}': {e}")
                item_res = EvaluationItemResult(
                    run_id=run.id,
                    question_id=q.id,
                    question_text=q.question,
                    reference_answer=q.reference_answer,
                    generated_answer=None,
                    retrieved_chunk_ids=[],
                    relevant_chunk_ids=list(relevant_chunk_ids_set),
                    retrieval_metrics={},
                    faithfulness_score=None,
                    answer_relevance_score=None,
                    citation_metrics={},
                    claim_judgments=[],
                    routing_metrics={},
                    accuracy_metrics={},
                    latency_breakdown={},
                    token_breakdown={},
                    provider_reported_tokens={},
                    measurement_source="unknown",
                    estimated_cost_usd=0.0,
                    trace_id=tracer.trace_id,
                    status="failed",
                    error_message=str(e),
                )
                session.add(item_res)
                session.flush()

                tracer.set_error(str(e))
                tracer.item_result_id = item_res.id
                tracer.persist(session)
                all_item_results.append(item_res)

            # Update progress checkpoint
            run.progress_current = idx
            session.commit()

        # Compute Aggregates
        n_eval = len(all_item_results)
        n_success = sum(1 for r in all_item_results if r.status == "success")

        agg_metrics: Dict[str, Any] = {
            "total_questions": len(questions),
            "evaluated_questions": n_eval,
            "successful_questions": n_success,
            "failed_questions": n_eval - n_success,
            f"mean_precision@{top_k}": round(sum(p_at_k_vals) / len(p_at_k_vals), 4) if p_at_k_vals else 0.0,
            f"mean_recall@{top_k}": round(sum(r_at_k_vals) / len(r_at_k_vals), 4) if r_at_k_vals else 0.0,
            f"mean_f1@{top_k}": round(sum(f1_at_k_vals) / len(f1_at_k_vals), 4) if f1_at_k_vals else 0.0,
            f"mean_hit_rate@{top_k}": round(sum(hit_vals) / len(hit_vals), 4) if hit_vals else 0.0,
            "mrr": round(mean_reciprocal_rank(rr_vals), 4),
            "map": round(mean_average_precision(ap_vals), 4),
            f"mean_ndcg@{top_k}": round(sum(ndcg_vals) / len(ndcg_vals), 4) if ndcg_vals else 0.0,
        }

        if run_type != "retrieval_only":
            agg_metrics.update(
                {
                    "mean_exact_match": round(sum(em_vals) / len(em_vals), 4) if em_vals else 0.0,
                    "mean_token_f1": round(sum(token_f1_vals) / len(token_f1_vals), 4) if token_f1_vals else 0.0,
                    "mean_normalized_match": round(sum(norm_match_vals) / len(norm_match_vals), 4) if norm_match_vals else 0.0,
                    "mean_faithfulness": round(sum(faith_vals) / len(faith_vals), 4) if faith_vals else 0.0,
                    "mean_answer_relevance": round(sum(relevance_vals) / len(relevance_vals), 4) if relevance_vals else 0.0,
                    "mean_citation_id_validity": round(sum(cite_id_vals) / len(cite_id_vals), 4) if cite_id_vals else 1.0,
                    "mean_citation_support_precision": round(sum(cite_support_vals) / len(cite_support_vals), 4) if cite_support_vals else 1.0,
                    "mean_citation_coverage": round(sum(cite_cov_vals) / len(cite_cov_vals), 4) if cite_cov_vals else 1.0,
                    "correct_refusal_rate": round(sum(correct_refusal_vals) / len(correct_refusal_vals), 4) if correct_refusal_vals else 1.0,
                    "false_answer_rate": round(sum(false_answer_vals) / len(false_answer_vals), 4) if false_answer_vals else 0.0,
                    "over_refusal_rate": round(sum(over_refusal_vals) / len(over_refusal_vals), 4) if over_refusal_vals else 0.0,
                    "unsupported_assertion_rate": round(sum(unsupported_assertion_vals) / len(unsupported_assertion_vals), 4) if unsupported_assertion_vals else 0.0,
                }
            )

        if run_type == "unified":
            run.routing_summary = {
                "mean_route_accuracy": round(sum(route_acc_vals) / len(route_acc_vals), 4) if route_acc_vals else 0.0,
                "unnecessary_vision_rate": round(sum(unnecessary_vis_vals) / len(unnecessary_vis_vals), 4) if unnecessary_vis_vals else 0.0,
                "unnecessary_agentic_rate": round(sum(unnecessary_agent_vals) / len(unnecessary_agent_vals), 4) if unnecessary_agent_vals else 0.0,
                "cross_document_leakage_rate": round(sum(doc_leakage_vals) / len(doc_leakage_vals), 4) if doc_leakage_vals else 0.0,
            }

        agg_metrics["retrieval_strategy"] = retrieval_strategy

        latency_summary = {
            "mean_ms": round(sum(latencies_ms) / len(latencies_ms), 2) if latencies_ms else 0.0,
            "p50_ms": _calculate_percentile(latencies_ms, 50),
            "p95_ms": _calculate_percentile(latencies_ms, 95),
            "p99_ms": _calculate_percentile(latencies_ms, 99),
            "min_ms": round(min(latencies_ms), 2) if latencies_ms else 0.0,
            "max_ms": round(max(latencies_ms), 2) if latencies_ms else 0.0,
        }

        token_summary = {
            "input_tokens": total_in_tokens,
            "output_tokens": total_out_tokens,
            "context_tokens": total_ctx_tokens,
            "judge_tokens": total_judge_tokens,
            "total_tokens": total_in_tokens + total_out_tokens + total_judge_tokens,
        }

        cost_summary = {
            "estimated_cost_usd": round(total_cost_usd, 6),
            "currency": "USD",
            "is_estimated": True,
            "pricing_source": f"ModelPricingRegistry ({pricing_registry.default_version})",
        }

        if run.status != "cancelled":
            run.status = "completed" if n_success > 0 else "failed"

        run.aggregate_metrics = agg_metrics
        run.latency_summary = latency_summary
        run.token_summary = token_summary
        run.cost_summary = cost_summary
        run.completed_at = datetime.now(timezone.utc)

        session.commit()
        session.refresh(run)
        return run


evaluation_runner = EvaluationRunner()
