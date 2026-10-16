from fastapi import APIRouter
from app.api.v1.endpoints import health, documents, chunking, embeddings, retrieval, rag, evaluation

api_router = APIRouter()

api_router.include_router(health.router, tags=["Health & Readiness"])
api_router.include_router(documents.router, prefix="/documents", tags=["Documents"])
api_router.include_router(chunking.router, prefix="/chunking", tags=["Chunking Lab"])
api_router.include_router(embeddings.router, prefix="/embeddings", tags=["Embeddings Lab"])
api_router.include_router(retrieval.router, prefix="/retrieval", tags=["Vector Search & Retrieval Lab"])
api_router.include_router(rag.router, prefix="/rag", tags=["RAG Chat & Generation"])
api_router.include_router(evaluation.router, prefix="/evaluation", tags=["Evaluation Lab & Benchmarking"])


