import json
import uuid
import time
import math
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from fastapi.responses import StreamingResponse
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from sqlalchemy import desc

from app.db.session import get_db
from app.models.evaluation import (
    EvaluationDataset,
    EvaluationQuestion,
    EvaluationRun,
    EvaluationItemResult,
)
from app.evaluation.runner import evaluation_runner, _calculate_percentile
from app.evaluation.retrieval.metrics import precision_at_k, recall_at_k
from app.evaluation.cost import pricing_registry
from app.rag.retrieval.service import retrieval_service
from app.rag.retrieval.schemas import SearchRequest

logger = logging.getLogger(__name__)

router = APIRouter()


# ==========================================
# Pydantic Schemas
# ==========================================

class DatasetCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=150)
    description: Optional[str] = None
    document_collection_id: Optional[uuid.UUID] = None
    metadata_json: Dict[str, Any] = Field(default_factory=dict)


class DatasetUpdateRequest(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=150)
    description: Optional[str] = None
    document_collection_id: Optional[uuid.UUID] = None
    metadata_json: Optional[Dict[str, Any]] = None


class DatasetResponse(BaseModel):
    id: uuid.UUID
    name: str
    description: Optional[str]
    document_collection_id: Optional[uuid.UUID]
    metadata_json: Dict[str, Any]
    question_count: int = 0
    run_count: int = 0
    created_at: str
    updated_at: str


class QuestionCreateRequest(BaseModel):
    question: str = Field(..., min_length=3)
    reference_answer: str = Field(..., min_length=1)
    relevant_document_ids: List[uuid.UUID] = Field(default_factory=list)
    relevant_chunk_ids: List[uuid.UUID] = Field(default_factory=list)
    expected_evidence: Optional[str] = None
    category: str = Field(
        default="factual",
        description="Category: factual, multi_chunk, multi_document, paraphrasing, ambiguous, unanswerable, conflicting, prompt_injection",
    )
    difficulty: str = Field(default="medium", description="easy, medium, hard")
    is_answerable: bool = True
    is_reviewed: bool = False
    reviewer_notes: Optional[str] = None
    metadata_json: Dict[str, Any] = Field(default_factory=dict)


class QuestionUpdateRequest(BaseModel):
    question: Optional[str] = None
    reference_answer: Optional[str] = None
    relevant_document_ids: Optional[List[uuid.UUID]] = None
    relevant_chunk_ids: Optional[List[uuid.UUID]] = None
    expected_evidence: Optional[str] = None
    category: Optional[str] = None
    difficulty: Optional[str] = None
    is_answerable: Optional[bool] = None
    is_reviewed: Optional[bool] = None
    reviewer_notes: Optional[str] = None
    metadata_json: Optional[Dict[str, Any]] = None


class QuestionReviewRequest(BaseModel):
    is_reviewed: bool
    reviewer_notes: Optional[str] = None


class QuestionLabelRequest(BaseModel):
    relevance_labels: Dict[str, int] = Field(..., description="Map of chunk_id -> 0/1/2")
    key_evidence_spans: Optional[List[str]] = Field(default=None)
    reviewer_notes: Optional[str] = None
    reviewer: Optional[str] = "human_evaluator"


class QuestionResponse(BaseModel):
    id: uuid.UUID
    dataset_id: uuid.UUID
    question: str
    reference_answer: str
    relevant_document_ids: List[uuid.UUID]
    relevant_chunk_ids: List[uuid.UUID]
    relevance_labels: Dict[str, int] = Field(default_factory=dict)
    audit_history: List[Dict[str, Any]] = Field(default_factory=list)
    revision: int = 1
    key_evidence_spans: List[str] = Field(default_factory=list)
    expected_evidence: Optional[str]
    category: str
    difficulty: str
    is_answerable: bool
    is_reviewed: bool
    reviewer_notes: Optional[str]
    metadata_json: Dict[str, Any]
    created_at: str


class EvaluationRunCreateRequest(BaseModel):
    dataset_id: uuid.UUID
    name: str = Field(..., min_length=1, max_length=150)
    run_type: str = Field(default="end_to_end", description="retrieval_only, generation_only, end_to_end")
    retrieval_strategy: str = Field(
        default="dense_only",
        description="Retrieval strategy: 'dense_only', 'lexical_only', 'hybrid_linear', 'hybrid_rrf', 'dense_rerank', 'hybrid_rrf_rerank', 'multi_query_hybrid_rerank', 'adaptive_context'",
    )
    embedding_model_id: str = "sentence-transformers/all-MiniLM-L6-v2"
    search_algorithm: str = Field(default="exact", description="exact, hnsw, ivfflat")
    distance_metric: str = Field(default="cosine", description="cosine, l2, inner_product")
    top_k: int = Field(default=5, ge=1, le=50)
    hybrid_alpha: float = Field(default=0.5, ge=0.0, le=1.0)
    rrf_k: int = Field(default=60, ge=1, le=500)
    reranker_model: str = Field(default="none")
    enable_query_expansion: bool = Field(default=False)
    enable_neighbor_expansion: bool = Field(default=False)
    ef_search: Optional[int] = Field(default=40, ge=1, le=500)
    probes: Optional[int] = Field(default=1, ge=1, le=100)
    llm_provider: str = Field(default="mock", description="mock, gemini, openai, ollama")
    llm_model: Optional[str] = None
    prompt_template: str = Field(default="grounded")
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_context_tokens: int = Field(default=2000, ge=100, le=32000)
    max_output_tokens: int = Field(default=512, ge=16, le=4096)
    judge_provider: Optional[str] = Field(default="heuristic", description="heuristic, gemini, openai, ollama, none")
    judge_model: Optional[str] = None
    max_questions: Optional[int] = Field(default=None, description="Guardrail limit for evaluation questions")


class EvaluationRunResponse(BaseModel):
    id: uuid.UUID
    dataset_id: Optional[uuid.UUID]
    dataset_name: Optional[str]
    name: str
    status: str
    run_type: str
    config: Dict[str, Any]
    aggregate_metrics: Dict[str, Any]
    latency_summary: Dict[str, Any]
    token_summary: Dict[str, Any]
    cost_summary: Dict[str, Any]
    error_message: Optional[str]
    created_at: str
    completed_at: Optional[str]


class EvaluationItemResultResponse(BaseModel):
    id: uuid.UUID
    run_id: uuid.UUID
    question_id: Optional[uuid.UUID]
    question_text: str
    reference_answer: Optional[str]
    generated_answer: Optional[str]
    retrieved_chunk_ids: List[str]
    relevant_chunk_ids: List[str]
    retrieval_metrics: Dict[str, Any]
    faithfulness_score: Optional[float]
    answer_relevance_score: Optional[float]
    citation_metrics: Dict[str, Any]
    claim_judgments: List[Dict[str, Any]]
    latency_breakdown: Dict[str, Any]
    token_breakdown: Dict[str, Any]
    estimated_cost_usd: float
    status: str
    error_message: Optional[str]
    created_at: str


# ==========================================
# Dataset CRUD Endpoints
# ==========================================

@router.get("/datasets", response_model=List[DatasetResponse])
def list_datasets(db: Session = Depends(get_db)):
    datasets = db.query(EvaluationDataset).order_by(desc(EvaluationDataset.created_at)).all()
    results = []
    for d in datasets:
        q_count = db.query(EvaluationQuestion).filter(EvaluationQuestion.dataset_id == d.id).count()
        r_count = db.query(EvaluationRun).filter(EvaluationRun.dataset_id == d.id).count()
        results.append(
            DatasetResponse(
                id=d.id,
                name=d.name,
                description=d.description,
                document_collection_id=d.document_collection_id,
                metadata_json=d.metadata_json or {},
                question_count=q_count,
                run_count=r_count,
                created_at=d.created_at.isoformat(),
                updated_at=d.updated_at.isoformat(),
            )
        )
    return results


@router.post("/datasets", response_model=DatasetResponse, status_code=201)
def create_dataset(req: DatasetCreateRequest, db: Session = Depends(get_db)):
    dataset = EvaluationDataset(
        name=req.name,
        description=req.description,
        document_collection_id=req.document_collection_id,
        metadata_json=req.metadata_json or {},
    )
    db.add(dataset)
    db.commit()
    db.refresh(dataset)
    return DatasetResponse(
        id=dataset.id,
        name=dataset.name,
        description=dataset.description,
        document_collection_id=dataset.document_collection_id,
        metadata_json=dataset.metadata_json or {},
        question_count=0,
        run_count=0,
        created_at=dataset.created_at.isoformat(),
        updated_at=dataset.updated_at.isoformat(),
    )


@router.get("/datasets/{dataset_id}", response_model=DatasetResponse)
def get_dataset(dataset_id: uuid.UUID, db: Session = Depends(get_db)):
    d = db.query(EvaluationDataset).filter(EvaluationDataset.id == dataset_id).first()
    if not d:
        raise HTTPException(status_code=404, detail="Dataset not found")
    q_count = db.query(EvaluationQuestion).filter(EvaluationQuestion.dataset_id == d.id).count()
    r_count = db.query(EvaluationRun).filter(EvaluationRun.dataset_id == d.id).count()
    return DatasetResponse(
        id=d.id,
        name=d.name,
        description=d.description,
        document_collection_id=d.document_collection_id,
        metadata_json=d.metadata_json or {},
        question_count=q_count,
        run_count=r_count,
        created_at=d.created_at.isoformat(),
        updated_at=d.updated_at.isoformat(),
    )


@router.put("/datasets/{dataset_id}", response_model=DatasetResponse)
def update_dataset(
    dataset_id: uuid.UUID, req: DatasetUpdateRequest, db: Session = Depends(get_db)
):
    d = db.query(EvaluationDataset).filter(EvaluationDataset.id == dataset_id).first()
    if not d:
        raise HTTPException(status_code=404, detail="Dataset not found")
    if req.name is not None:
        d.name = req.name
    if req.description is not None:
        d.description = req.description
    if req.document_collection_id is not None:
        d.document_collection_id = req.document_collection_id
    if req.metadata_json is not None:
        d.metadata_json = req.metadata_json
    d.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(d)
    q_count = db.query(EvaluationQuestion).filter(EvaluationQuestion.dataset_id == d.id).count()
    r_count = db.query(EvaluationRun).filter(EvaluationRun.dataset_id == d.id).count()
    return DatasetResponse(
        id=d.id,
        name=d.name,
        description=d.description,
        document_collection_id=d.document_collection_id,
        metadata_json=d.metadata_json or {},
        question_count=q_count,
        run_count=r_count,
        created_at=d.created_at.isoformat(),
        updated_at=d.updated_at.isoformat(),
    )


@router.delete("/datasets/{dataset_id}")
def delete_dataset(dataset_id: uuid.UUID, db: Session = Depends(get_db)):
    d = db.query(EvaluationDataset).filter(EvaluationDataset.id == dataset_id).first()
    if not d:
        raise HTTPException(status_code=404, detail="Dataset not found")
    db.delete(d)
    db.commit()
    return {"status": "deleted", "id": str(dataset_id)}


# ==========================================
# Question Management Endpoints
# ==========================================

def _format_question_response(item: EvaluationQuestion) -> QuestionResponse:
    return QuestionResponse(
        id=item.id,
        dataset_id=item.dataset_id,
        question=item.question,
        reference_answer=item.reference_answer,
        relevant_document_ids=[uuid.UUID(str(x)) for x in (item.relevant_document_ids or [])],
        relevant_chunk_ids=[uuid.UUID(str(x)) for x in (item.relevant_chunk_ids or [])],
        relevance_labels=item.relevance_labels or {},
        audit_history=item.audit_history or [],
        revision=item.revision or 1,
        key_evidence_spans=item.key_evidence_spans or [],
        expected_evidence=item.expected_evidence,
        category=item.category,
        difficulty=item.difficulty,
        is_answerable=item.is_answerable,
        is_reviewed=item.is_reviewed,
        reviewer_notes=item.reviewer_notes,
        metadata_json=item.metadata_json or {},
        created_at=item.created_at.isoformat(),
    )


@router.get("/datasets/{dataset_id}/questions", response_model=List[QuestionResponse])
def list_questions(
    dataset_id: uuid.UUID,
    category: Optional[str] = None,
    is_reviewed: Optional[bool] = None,
    db: Session = Depends(get_db),
):
    q = db.query(EvaluationQuestion).filter(EvaluationQuestion.dataset_id == dataset_id)
    if category:
        q = q.filter(EvaluationQuestion.category == category)
    if is_reviewed is not None:
        q = q.filter(EvaluationQuestion.is_reviewed == is_reviewed)
    questions = q.order_by(EvaluationQuestion.created_at).all()
    return [_format_question_response(item) for item in questions]


@router.post("/datasets/{dataset_id}/questions", response_model=QuestionResponse, status_code=201)
def add_question(
    dataset_id: uuid.UUID, req: QuestionCreateRequest, db: Session = Depends(get_db)
):
    dataset = db.query(EvaluationDataset).filter(EvaluationDataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Duplicate check within dataset
    existing = (
        db.query(EvaluationQuestion)
        .filter(
            EvaluationQuestion.dataset_id == dataset_id,
            EvaluationQuestion.question == req.question.strip(),
        )
        .first()
    )
    if existing:
        raise HTTPException(
            status_code=400,
            detail=f"Question already exists in this dataset with ID '{existing.id}'.",
        )

    # Initialize relevance labels map from relevant_chunk_ids (defaulting to 2 = highly relevant)
    initial_labels = {str(c): 2 for c in req.relevant_chunk_ids}
    initial_audit = [
        {
            "revision": 1,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "reviewer": "system_init",
            "notes": "Question created",
            "labels_count": len(initial_labels),
            "high_relevance_count": len(initial_labels),
            "partial_relevance_count": 0,
        }
    ]

    q = EvaluationQuestion(
        dataset_id=dataset_id,
        question=req.question.strip(),
        reference_answer=req.reference_answer.strip(),
        relevant_document_ids=[str(d) for d in req.relevant_document_ids],
        relevant_chunk_ids=[str(c) for c in req.relevant_chunk_ids],
        relevance_labels=initial_labels,
        audit_history=initial_audit,
        revision=1,
        key_evidence_spans=[],
        expected_evidence=req.expected_evidence,
        category=req.category,
        difficulty=req.difficulty,
        is_answerable=req.is_answerable,
        is_reviewed=req.is_reviewed,
        reviewer_notes=req.reviewer_notes,
        metadata_json=req.metadata_json or {},
    )
    db.add(q)
    db.commit()
    db.refresh(q)
    return _format_question_response(q)


@router.put("/questions/{question_id}", response_model=QuestionResponse)
def update_question(
    question_id: uuid.UUID, req: QuestionUpdateRequest, db: Session = Depends(get_db)
):
    q = db.query(EvaluationQuestion).filter(EvaluationQuestion.id == question_id).first()
    if not q:
        raise HTTPException(status_code=404, detail="Question not found")
    if req.question is not None:
        q.question = req.question.strip()
    if req.reference_answer is not None:
        q.reference_answer = req.reference_answer.strip()
    if req.relevant_document_ids is not None:
        q.relevant_document_ids = [str(d) for d in req.relevant_document_ids]
    if req.relevant_chunk_ids is not None:
        q.relevant_chunk_ids = [str(c) for c in req.relevant_chunk_ids]
    if req.expected_evidence is not None:
        q.expected_evidence = req.expected_evidence
    if req.category is not None:
        q.category = req.category
    if req.difficulty is not None:
        q.difficulty = req.difficulty
    if req.is_answerable is not None:
        q.is_answerable = req.is_answerable
    if req.is_reviewed is not None:
        q.is_reviewed = req.is_reviewed
    if req.reviewer_notes is not None:
        q.reviewer_notes = req.reviewer_notes
    if req.metadata_json is not None:
        q.metadata_json = req.metadata_json

    db.commit()
    db.refresh(q)
    return _format_question_response(q)


@router.post("/questions/{question_id}/review", response_model=QuestionResponse)
def review_question(
    question_id: uuid.UUID, req: QuestionReviewRequest, db: Session = Depends(get_db)
):
    q = db.query(EvaluationQuestion).filter(EvaluationQuestion.id == question_id).first()
    if not q:
        raise HTTPException(status_code=404, detail="Question not found")
    q.is_reviewed = req.is_reviewed
    if req.reviewer_notes is not None:
        q.reviewer_notes = req.reviewer_notes
    db.commit()
    db.refresh(q)
    return _format_question_response(q)


@router.post("/questions/{question_id}/labels", response_model=QuestionResponse)
def label_question_evidence(
    question_id: uuid.UUID, req: QuestionLabelRequest, db: Session = Depends(get_db)
):
    q = db.query(EvaluationQuestion).filter(EvaluationQuestion.id == question_id).first()
    if not q:
        raise HTTPException(status_code=404, detail="Question not found")

    history = list(q.audit_history or [])
    entry = {
        "revision": q.revision or 1,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "reviewer": req.reviewer or "human_evaluator",
        "notes": req.reviewer_notes,
        "labels_count": len(req.relevance_labels),
        "high_relevance_count": sum(1 for v in req.relevance_labels.values() if v == 2),
        "partial_relevance_count": sum(1 for v in req.relevance_labels.values() if v == 1),
        "prior_relevant_chunks_count": len(q.relevant_chunk_ids or []),
    }
    history.append(entry)

    q.relevance_labels = req.relevance_labels
    q.audit_history = history
    q.revision = (q.revision or 1) + 1
    q.is_reviewed = True

    if req.key_evidence_spans is not None:
        q.key_evidence_spans = req.key_evidence_spans
    if req.reviewer_notes is not None:
        q.reviewer_notes = req.reviewer_notes

    # Keep relevant_chunk_ids aligned with chunks having score >= 1
    updated_relevant_ids = [cid for cid, score in req.relevance_labels.items() if score >= 1]
    if updated_relevant_ids:
        q.relevant_chunk_ids = updated_relevant_ids

    db.commit()
    db.refresh(q)
    return _format_question_response(q)


@router.get("/questions/{question_id}/audit")
def get_question_audit(question_id: uuid.UUID, db: Session = Depends(get_db)):
    q = db.query(EvaluationQuestion).filter(EvaluationQuestion.id == question_id).first()
    if not q:
        raise HTTPException(status_code=404, detail="Question not found")
    return {
        "question_id": str(q.id),
        "revision": q.revision or 1,
        "is_reviewed": q.is_reviewed,
        "audit_history": q.audit_history or [],
        "relevance_labels": q.relevance_labels or {},
        "key_evidence_spans": q.key_evidence_spans or [],
    }


@router.delete("/questions/{question_id}")
def delete_question(question_id: uuid.UUID, db: Session = Depends(get_db)):
    q = db.query(EvaluationQuestion).filter(EvaluationQuestion.id == question_id).first()
    if not q:
        raise HTTPException(status_code=404, detail="Question not found")
    db.delete(q)
    db.commit()
    return {"status": "deleted", "id": str(question_id)}


# ==========================================
# Import / Export Endpoints
# ==========================================

@router.post("/datasets/{dataset_id}/import")
def import_questions(
    dataset_id: uuid.UUID,
    payload: Dict[str, Any],
    db: Session = Depends(get_db),
):
    dataset = db.query(EvaluationDataset).filter(EvaluationDataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    items = payload.get("questions") or payload.get("items") or []
    if not isinstance(items, list):
        raise HTTPException(status_code=400, detail="Invalid payload. Expected a list of questions.")

    added_count = 0
    skipped_count = 0

    for item in items:
        q_text = item.get("question", "").strip()
        ref_ans = item.get("reference_answer", "").strip()
        if not q_text or not ref_ans:
            skipped_count += 1
            continue

        existing = (
            db.query(EvaluationQuestion)
            .filter(
                EvaluationQuestion.dataset_id == dataset_id,
                EvaluationQuestion.question == q_text,
            )
            .first()
        )
        if existing:
            skipped_count += 1
            continue

        # Convert chunk & document IDs safely
        doc_ids = [str(d) for d in item.get("relevant_document_ids", [])]
        chunk_ids = [str(c) for c in item.get("relevant_chunk_ids", [])]

        question = EvaluationQuestion(
            dataset_id=dataset_id,
            question=q_text,
            reference_answer=ref_ans,
            relevant_document_ids=doc_ids,
            relevant_chunk_ids=chunk_ids,
            expected_evidence=item.get("expected_evidence"),
            category=item.get("category", "factual"),
            difficulty=item.get("difficulty", "medium"),
            is_answerable=item.get("is_answerable", True),
            is_reviewed=item.get("is_reviewed", False),
            reviewer_notes=item.get("reviewer_notes"),
            metadata_json=item.get("metadata", {}),
        )
        db.add(question)
        added_count += 1

    db.commit()
    return {
        "status": "success",
        "dataset_id": str(dataset_id),
        "imported_questions": added_count,
        "skipped_questions": skipped_count,
    }


@router.get("/datasets/{dataset_id}/export")
def export_dataset(dataset_id: uuid.UUID, db: Session = Depends(get_db)):
    d = db.query(EvaluationDataset).filter(EvaluationDataset.id == dataset_id).first()
    if not d:
        raise HTTPException(status_code=404, detail="Dataset not found")

    questions = (
        db.query(EvaluationQuestion)
        .filter(EvaluationQuestion.dataset_id == dataset_id)
        .order_by(EvaluationQuestion.created_at)
        .all()
    )

    export_payload = {
        "dataset_id": str(d.id),
        "name": d.name,
        "description": d.description,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "total_questions": len(questions),
        "questions": [
            {
                "id": str(q.id),
                "question": q.question,
                "reference_answer": q.reference_answer,
                "relevant_document_ids": q.relevant_document_ids or [],
                "relevant_chunk_ids": q.relevant_chunk_ids or [],
                "expected_evidence": q.expected_evidence,
                "category": q.category,
                "difficulty": q.difficulty,
                "is_answerable": q.is_answerable,
                "is_reviewed": q.is_reviewed,
                "reviewer_notes": q.reviewer_notes,
                "metadata": q.metadata_json or {},
            }
            for q in questions
        ],
    }
    return export_payload


# ==========================================
# Evaluation Runs Endpoints
# ==========================================

@router.post("/runs", response_model=EvaluationRunResponse, status_code=201)
def create_and_execute_run(
    req: EvaluationRunCreateRequest,
    background_tasks: BackgroundTasks,
    run_async: bool = Query(False, description="Whether to execute in background"),
    db: Session = Depends(get_db),
):
    dataset = db.query(EvaluationDataset).filter(EvaluationDataset.id == req.dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Evaluation dataset not found")

    q_count = db.query(EvaluationQuestion).filter(EvaluationQuestion.dataset_id == req.dataset_id).count()
    if q_count == 0:
        raise HTTPException(status_code=400, detail="Cannot run evaluation on empty dataset.")

    run = EvaluationRun(
        dataset_id=req.dataset_id,
        name=req.name,
        status="pending",
        run_type=req.run_type,
        config=req.model_dump(mode="json"),
        aggregate_metrics={},
        latency_summary={},
        token_summary={},
        cost_summary={},
    )
    db.add(run)
    db.commit()
    db.refresh(run)

    if run_async:
        # Background task execution
        background_tasks.add_task(evaluation_runner.run_evaluation, db, run.id)
    else:
        # Synchronous execution
        run = evaluation_runner.run_evaluation(db, run.id)

    return EvaluationRunResponse(
        id=run.id,
        dataset_id=run.dataset_id,
        dataset_name=dataset.name if dataset else None,
        name=run.name,
        status=run.status,
        run_type=run.run_type,
        config=run.config or {},
        aggregate_metrics=run.aggregate_metrics or {},
        latency_summary=run.latency_summary or {},
        token_summary=run.token_summary or {},
        cost_summary=run.cost_summary or {},
        error_message=run.error_message,
        created_at=run.created_at.isoformat(),
        completed_at=run.completed_at.isoformat() if run.completed_at else None,
    )


@router.get("/runs", response_model=List[EvaluationRunResponse])
def list_runs(
    dataset_id: Optional[uuid.UUID] = None,
    run_type: Optional[str] = None,
    db: Session = Depends(get_db),
):
    q = db.query(EvaluationRun)
    if dataset_id:
        q = q.filter(EvaluationRun.dataset_id == dataset_id)
    if run_type:
        q = q.filter(EvaluationRun.run_type == run_type)

    runs = q.order_by(desc(EvaluationRun.created_at)).all()
    results = []
    for r in runs:
        d_name = r.dataset.name if r.dataset else None
        results.append(
            EvaluationRunResponse(
                id=r.id,
                dataset_id=r.dataset_id,
                dataset_name=d_name,
                name=r.name,
                status=r.status,
                run_type=r.run_type,
                config=r.config or {},
                aggregate_metrics=r.aggregate_metrics or {},
                latency_summary=r.latency_summary or {},
                token_summary=r.token_summary or {},
                cost_summary=r.cost_summary or {},
                error_message=r.error_message,
                created_at=r.created_at.isoformat(),
                completed_at=r.completed_at.isoformat() if r.completed_at else None,
            )
        )
    return results


@router.get("/runs/{run_id}", response_model=EvaluationRunResponse)
def get_run(run_id: uuid.UUID, db: Session = Depends(get_db)):
    r = db.query(EvaluationRun).filter(EvaluationRun.id == run_id).first()
    if not r:
        raise HTTPException(status_code=404, detail="Evaluation run not found")
    d_name = r.dataset.name if r.dataset else None
    return EvaluationRunResponse(
        id=r.id,
        dataset_id=r.dataset_id,
        dataset_name=d_name,
        name=r.name,
        status=r.status,
        run_type=r.run_type,
        config=r.config or {},
        aggregate_metrics=r.aggregate_metrics or {},
        latency_summary=r.latency_summary or {},
        token_summary=r.token_summary or {},
        cost_summary=r.cost_summary or {},
        error_message=r.error_message,
        created_at=r.created_at.isoformat(),
        completed_at=r.completed_at.isoformat() if r.completed_at else None,
    )


@router.get("/runs/{run_id}/questions", response_model=List[EvaluationItemResultResponse])
def get_run_question_results(run_id: uuid.UUID, db: Session = Depends(get_db)):
    items = (
        db.query(EvaluationItemResult)
        .filter(EvaluationItemResult.run_id == run_id)
        .order_by(EvaluationItemResult.created_at)
        .all()
    )
    return [
        EvaluationItemResultResponse(
            id=item.id,
            run_id=item.run_id,
            question_id=item.question_id,
            question_text=item.question_text,
            reference_answer=item.reference_answer,
            generated_answer=item.generated_answer,
            retrieved_chunk_ids=[str(x) for x in (item.retrieved_chunk_ids or [])],
            relevant_chunk_ids=[str(x) for x in (item.relevant_chunk_ids or [])],
            retrieval_metrics=item.retrieval_metrics or {},
            faithfulness_score=item.faithfulness_score,
            answer_relevance_score=item.answer_relevance_score,
            citation_metrics=item.citation_metrics or {},
            claim_judgments=item.claim_judgments or [],
            latency_breakdown=item.latency_breakdown or {},
            token_breakdown=item.token_breakdown or {},
            estimated_cost_usd=item.estimated_cost_usd,
            status=item.status,
            error_message=item.error_message,
            created_at=item.created_at.isoformat(),
        )
        for item in items
    ]


@router.post("/runs/{run_id}/cancel")
def cancel_run(run_id: uuid.UUID, db: Session = Depends(get_db)):
    r = db.query(EvaluationRun).filter(EvaluationRun.id == run_id).first()
    if not r:
        raise HTTPException(status_code=404, detail="Run not found")
    evaluation_runner.cancel_run(run_id)
    r.status = "cancelled"
    db.commit()
    return {"status": "cancelled", "run_id": str(run_id)}


@router.post("/runs/{run_id}/retry", response_model=EvaluationRunResponse)
def retry_run(run_id: uuid.UUID, db: Session = Depends(get_db)):
    r = db.query(EvaluationRun).filter(EvaluationRun.id == run_id).first()
    if not r:
        raise HTTPException(status_code=404, detail="Run not found")
    # Clean previous item results and rerun
    db.query(EvaluationItemResult).filter(EvaluationItemResult.run_id == run_id).delete()
    r.status = "pending"
    db.commit()
    updated_run = evaluation_runner.run_evaluation(db, run_id)
    d_name = updated_run.dataset.name if updated_run.dataset else None
    return EvaluationRunResponse(
        id=updated_run.id,
        dataset_id=updated_run.dataset_id,
        dataset_name=d_name,
        name=updated_run.name,
        status=updated_run.status,
        run_type=updated_run.run_type,
        config=updated_run.config or {},
        aggregate_metrics=updated_run.aggregate_metrics or {},
        latency_summary=updated_run.latency_summary or {},
        token_summary=updated_run.token_summary or {},
        cost_summary=updated_run.cost_summary or {},
        error_message=updated_run.error_message,
        created_at=updated_run.created_at.isoformat(),
        completed_at=updated_run.completed_at.isoformat() if updated_run.completed_at else None,
    )


# ==========================================
# Run Comparison & Metrics Educational Metadata
# ==========================================

@router.get("/compare")
def compare_runs(
    run_a: uuid.UUID = Query(...),
    run_b: uuid.UUID = Query(...),
    db: Session = Depends(get_db),
):
    a = db.query(EvaluationRun).filter(EvaluationRun.id == run_a).first()
    b = db.query(EvaluationRun).filter(EvaluationRun.id == run_b).first()
    if not a or not b:
        raise HTTPException(status_code=404, detail="One or both evaluation runs not found")

    metrics_a = a.aggregate_metrics or {}
    metrics_b = b.aggregate_metrics or {}
    lat_a = a.latency_summary or {}
    lat_b = b.latency_summary or {}
    cost_a = a.cost_summary or {}
    cost_b = b.cost_summary or {}

    metric_keys = set(metrics_a.keys()).union(set(metrics_b.keys()))
    comparison_metrics = []

    for k in sorted(metric_keys):
        v_a = metrics_a.get(k)
        v_b = metrics_b.get(k)
        if isinstance(v_a, (int, float)) and isinstance(v_b, (int, float)):
            delta = round(v_b - v_a, 4)
            # For quality metrics, higher is better
            direction = "improved" if delta > 0 else ("regressed" if delta < 0 else "neutral")
            comparison_metrics.append(
                {
                    "metric": k,
                    "run_a_value": v_a,
                    "run_b_value": v_b,
                    "delta": delta,
                    "direction": direction,
                }
            )

    # Latency comparison (lower is better)
    p50_a = lat_a.get("p50_ms", 0.0)
    p50_b = lat_b.get("p50_ms", 0.0)
    lat_delta = round(p50_b - p50_a, 2)
    lat_direction = "improved" if lat_delta < 0 else ("regressed" if lat_delta > 0 else "neutral")

    # Cost comparison (lower is better)
    cost_val_a = cost_a.get("estimated_cost_usd", 0.0)
    cost_val_b = cost_b.get("estimated_cost_usd", 0.0)
    cost_delta = round(cost_val_b - cost_val_a, 6)
    cost_direction = "improved" if cost_delta < 0 else ("regressed" if cost_delta > 0 else "neutral")

    return {
        "run_a": {
            "id": str(a.id),
            "name": a.name,
            "run_type": a.run_type,
            "config": a.config,
        },
        "run_b": {
            "id": str(b.id),
            "name": b.name,
            "run_type": b.run_type,
            "config": b.config,
        },
        "metrics_comparison": comparison_metrics,
        "latency_comparison": {
            "p50_a_ms": p50_a,
            "p50_b_ms": p50_b,
            "delta_ms": lat_delta,
            "direction": lat_direction,
        },
        "cost_comparison": {
            "cost_a_usd": cost_val_a,
            "cost_b_usd": cost_val_b,
            "delta_usd": cost_delta,
            "direction": cost_direction,
        },
    }


@router.get("/metrics")
def get_metrics_educational_catalog():
    """
    Returns educational definitions, mathematical formulas, worked examples,
    strengths, and limitations for all evaluation metrics.
    """
    return [
        {
            "id": "precision_at_k",
            "name": "Precision@K",
            "category": "retrieval",
            "formula": "Precision@K = |Retrieved[:K] ∩ Relevant| / K",
            "description": "Measures the fraction of top-K retrieved chunks that are truly relevant evidence.",
            "interpretation": "High Precision@K means minimal noise and irrelevant distraction entering the LLM prompt context.",
            "strengths": "Directly evaluates context cleanliness and prompt noise efficiency.",
            "limitations": "Penalizes systems when the total number of relevant documents is smaller than K.",
            "common_pitfall": "Comparing Precision@10 on a query with only 2 relevant chunks, where the maximum possible precision is 0.20.",
        },
        {
            "id": "recall_at_k",
            "name": "Recall@K",
            "category": "retrieval",
            "formula": "Recall@K = |Retrieved[:K] ∩ Relevant| / |All Relevant Chunks|",
            "description": "Measures the fraction of all necessary evidence successfully captured in top-K.",
            "interpretation": "High Recall@K ensures the LLM has all pieces required to synthesize complete answers.",
            "strengths": "Essential for multi-hop questions requiring multiple chunks.",
            "limitations": "Can be trivially maximized by choosing a massive K (which inflates cost and noise).",
            "common_pitfall": "Ignoring the tradeoff between high recall and context window degradation ('lost-in-the-middle').",
        },
        {
            "id": "mrr",
            "name": "Mean Reciprocal Rank (MRR)",
            "category": "retrieval",
            "formula": "MRR = (1 / N) * sum(1 / rank_i)",
            "description": "Calculates the average reciprocal rank of the first relevant retrieved document.",
            "interpretation": "Reflects how quickly a user or downstream system finds their first relevant piece of evidence.",
            "strengths": "Simple, intuitive, and rewards placing key evidence at the very top.",
            "limitations": "Only considers the single highest-ranking relevant item, ignoring multi-evidence completeness.",
            "common_pitfall": "Relying solely on MRR for complex reasoning questions that demand multiple chunks.",
        },
        {
            "id": "ndcg_at_k",
            "name": "nDCG@K",
            "category": "retrieval",
            "formula": "nDCG@K = DCG@K / IDCG@K, where DCG@K = sum((2^rel - 1) / log2(i + 1))",
            "description": "Normalized Discounted Cumulative Gain assesses graded ranking quality with position-based logarithmic discounting.",
            "interpretation": "1.0 represents a mathematically optimal ranking where the most relevant items appear first.",
            "strengths": "Standard industrial metric supporting graded relevance and position sensitivity.",
            "limitations": "Requires careful calibration of relevance gain scales.",
            "common_pitfall": "Confusing binary relevance nDCG with graded multi-tier relevance evaluations.",
        },
        {
            "id": "faithfulness",
            "name": "Answer Faithfulness",
            "category": "generation",
            "formula": "Faithfulness = Supported Claims / Total Extracted Claims",
            "description": "Measures whether factual claims generated by the LLM are grounded in the retrieved context.",
            "interpretation": "Detects hallucinations and unsupported extrapolations beyond provided evidence.",
            "strengths": "Granular, claim-level explainability with explicit passage attribution.",
            "limitations": "Depends on the reliability of the claim extractor and judge evaluator.",
            "common_pitfall": "Assuming that having citations automatically implies factual faithfulness.",
        },
        {
            "id": "citation_support_precision",
            "name": "Citation Support Precision",
            "category": "generation",
            "formula": "Citation Support Precision = Supported Cited Claims / Total Cited Claims",
            "description": "Measures what fraction of citations actually support the factual proposition they are attached to.",
            "interpretation": "Exposes citation hallucination, where models attach valid tags [S1] to unrelated claims.",
            "strengths": "Distinguishes syntactic tag correctness from semantic truth entailment.",
            "limitations": "Requires semantic reasoning over the cited chunk excerpt.",
            "common_pitfall": "Checking only if [S1] exists in the retrieved context without checking if [S1] states the claim.",
        },
        {
            "id": "answer_relevance",
            "name": "Answer Relevance",
            "category": "generation",
            "formula": "Answer Relevance = 0.7 * Completeness + 0.3 * Question Intent Overlap",
            "description": "Evaluates whether the generated response directly answers what the user asked.",
            "interpretation": "Detects evasive, off-topic, or incomplete answers while rewarding correct refusal when evidence is missing.",
            "strengths": "Penalizes rambling while validating refusal accuracy on unanswerable questions.",
            "limitations": "Surface token overlap heuristics must be complemented with semantic judges.",
            "common_pitfall": "Penalizing an answer as irrelevant when it correctly refuses to answer an unanswerable question.",
        },
    ]


@router.post("/experiments/tradeoff")
def run_tradeoff_experiment(
    dataset_id: uuid.UUID,
    embedding_model_id: str = "sentence-transformers/all-MiniLM-L6-v2",
    top_k_values: List[int] = Query(default=[1, 3, 5, 10, 20]),
    algorithms: List[str] = Query(default=["exact", "hnsw"]),
    db: Session = Depends(get_db),
):
    """
    Sweeps across Top-K and retrieval algorithms on a dataset to generate a
    scientific quality-versus-latency tradeoff curve.
    """
    dataset = db.query(EvaluationDataset).filter(EvaluationDataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    questions = db.query(EvaluationQuestion).filter(EvaluationQuestion.dataset_id == dataset_id).all()
    if not questions:
        raise HTTPException(status_code=400, detail="Dataset has no questions")

    results = []

    for algo in algorithms:
        for k in top_k_values:
            p_vals = []
            r_vals = []
            lats = []

            for q in questions:
                t0 = time.perf_counter()
                s_req = SearchRequest(
                    query=q.question,
                    model_id=embedding_model_id,
                    search_algorithm=algo,
                    distance_metric="cosine",
                    top_k=k,
                    explain_plan=False,
                )
                try:
                    s_resp = retrieval_service.search(db, s_req)
                    query_lat = (time.perf_counter() - t0) * 1000.0

                    retrieved_ids = [str(r.chunk_id) for r in s_resp.results]
                    relevant_set = {str(c) for c in (q.relevant_chunk_ids or [])}

                    p = precision_at_k(retrieved_ids, relevant_set, k=k)
                    r = recall_at_k(retrieved_ids, relevant_set, k=k)

                    p_vals.append(p)
                    r_vals.append(r)
                    lats.append(query_lat)
                except Exception as e:
                    logger.warning(f"Tradeoff query failed: {e}")

            if p_vals:
                results.append(
                    {
                        "algorithm": algo,
                        "top_k": k,
                        "mean_precision": round(sum(p_vals) / len(p_vals), 4),
                        "mean_recall": round(sum(r_vals) / len(r_vals), 4),
                        "mean_latency_ms": round(sum(lats) / len(lats), 2),
                        "p50_latency_ms": _calculate_percentile(lats, 50),
                        "p95_latency_ms": _calculate_percentile(lats, 95),
                    }
                )

    return {
        "dataset_id": str(dataset_id),
        "total_questions_evaluated": len(questions),
        "tradeoff_curve": results,
    }
