from typing import List, Optional
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.rag.embeddings.service import EmbeddingService
from app.schemas.embeddings import (
    ModelInfoResponse,
    TokenizeRequest,
    TokenizeResponse,
    TokenizeCompareRequest,
    GenerateEmbeddingsRequest,
    GenerateEmbeddingsResponse,
    SimilarityCalculateRequest,
    SimilarityCalculateResponse,
    SimilarityMatrixRequest,
    SimilarityMatrixResponse,
    ProjectEmbeddingsRequest,
    ProjectEmbeddingsResponse,
    BenchmarkRunRequest,
    EmbeddingExperimentCreate,
    EmbeddingExperimentResponse,
    EmbeddingExperimentListResponse,
)

router = APIRouter()
embedding_service = EmbeddingService()


@router.get("/models", response_model=List[ModelInfoResponse])
def get_embedding_models():
    """List all registered embedding models, specifications, and availability."""
    return embedding_service.list_models()


@router.post("/tokenize", response_model=TokenizeResponse)
def tokenize_text(req: TokenizeRequest):
    """Inspect actual tokenizer tokens, IDs, token limits, and truncation warnings."""
    try:
        return embedding_service.tokenize_text(text=req.text, model_id=req.model_id)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Tokenization failed: {e}",
        )


@router.post("/tokenize/compare", response_model=List[TokenizeResponse])
def compare_tokenizers(req: TokenizeCompareRequest):
    """Compare tokenization behavior across multiple models on the same text."""
    results = []
    for model_id in req.model_ids:
        try:
            res = embedding_service.tokenize_text(text=req.text, model_id=model_id)
            results.append(res)
        except Exception as e:
            results.append({
                "model_id": model_id,
                "text": req.text,
                "char_count": len(req.text),
                "word_count": len(req.text.split()),
                "token_count": 0,
                "max_seq_length": 0,
                "tokens": [],
                "token_ids": [],
                "is_truncated": False,
                "truncated_tokens_count": 0,
                "warnings": [f"Failed to tokenize with '{model_id}': {e}"],
                "ratio_char_to_token": 0.0,
                "ratio_word_to_token": 0.0,
            })
    return results


@router.post("/generate", response_model=GenerateEmbeddingsResponse)
def generate_embeddings(req: GenerateEmbeddingsRequest):
    """Generate dense embeddings from custom text strings."""
    if not req.texts:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Field 'texts' must contain at least one text string.",
        )
    if len(req.texts) > 256:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Batch limit exceeded: maximum 256 texts per request.",
        )

    try:
        return embedding_service.generate_embeddings(
            texts=req.texts,
            model_id=req.model_id,
            batch_size=req.batch_size,
            normalize=req.normalize,
            is_query=req.is_query,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Embedding generation failed: {e}",
        )


@router.post("/similarity", response_model=SimilarityCalculateResponse)
def calculate_similarity(req: SimilarityCalculateRequest):
    """Calculate Cosine similarity, Dot product, and Euclidean distance between two sentences."""
    if not req.text_a.strip() or not req.text_b.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Both text_a and text_b must be non-empty strings.",
        )

    try:
        return embedding_service.calculate_similarity(
            text_a=req.text_a,
            text_b=req.text_b,
            model_id=req.model_id,
            normalize=req.normalize,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Similarity calculation failed: {e}",
        )


@router.post("/matrix", response_model=SimilarityMatrixResponse)
def calculate_similarity_matrix(req: SimilarityMatrixRequest):
    """Calculate pairwise similarity or distance matrix for multiple texts."""
    if not req.texts or len(req.texts) < 2:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Must provide at least 2 texts for pairwise matrix computation.",
        )
    if len(req.texts) > 50:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Matrix computation capped at 50 texts to preserve performance.",
        )

    try:
        return embedding_service.calculate_matrix(
            texts=req.texts,
            model_id=req.model_id,
            metric=req.metric,
            normalize=req.normalize,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Matrix calculation failed: {e}",
        )


@router.post("/project", response_model=ProjectEmbeddingsResponse)
def project_embeddings(
    req: ProjectEmbeddingsRequest,
    db: Session = Depends(get_db),
):
    """Project high-dimensional embeddings to 2D or 3D using PCA, t-SNE, or UMAP."""
    try:
        return embedding_service.project(db=db, req=req)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Projection failed: {e}",
        )


@router.post("/benchmark")
def run_benchmark(req: BenchmarkRunRequest):
    """Run reproducible evaluation benchmark measuring quality, latency, and throughput."""
    try:
        return embedding_service.run_benchmark(model_id=req.model_id)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Benchmark execution failed: {e}",
        )


@router.post("/experiments", response_model=EmbeddingExperimentResponse)
def create_embedding_experiment(
    req: EmbeddingExperimentCreate,
    db: Session = Depends(get_db),
):
    """Generate and persist embeddings for an existing chunking experiment or document."""
    try:
        return embedding_service.create_experiment(db=db, req=req)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Experiment creation failed: {e}",
        )


@router.get("/experiments", response_model=EmbeddingExperimentListResponse)
def list_embedding_experiments(
    document_id: Optional[UUID] = Query(None),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    """List saved embedding experiments."""
    total, items = embedding_service.list_experiments(
        db=db, document_id=document_id, limit=limit, offset=offset
    )
    return {"total": total, "items": items, "limit": limit, "offset": offset}


@router.get("/experiments/{experiment_id}")
def get_embedding_experiment(
    experiment_id: UUID,
    db: Session = Depends(get_db),
):
    """Get details of an embedding experiment including sample vectors."""
    res = embedding_service.get_experiment(db=db, experiment_id=experiment_id)
    if not res:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Experiment not found")
    return res


@router.delete("/experiments/{experiment_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_embedding_experiment(
    experiment_id: UUID,
    db: Session = Depends(get_db),
):
    """Delete an embedding experiment and its associated vectors."""
    deleted = embedding_service.delete_experiment(db=db, experiment_id=experiment_id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Experiment not found")
