from fastapi import APIRouter
from app.api.v1.endpoints import (
    health,
    documents,
    chunking,
    embeddings,
    retrieval,
    rag,
    evaluation,
    conversations,
    agents,
    graph,
    multimodal,
)

api_router = APIRouter()

api_router.include_router(health.router, tags=["Health & Readiness"])
api_router.include_router(documents.router, prefix="/documents", tags=["Documents"])
api_router.include_router(chunking.router, prefix="/chunking", tags=["Chunking Lab"])
api_router.include_router(embeddings.router, prefix="/embeddings", tags=["Embeddings Lab"])
api_router.include_router(retrieval.router, prefix="/retrieval", tags=["Vector Search & Retrieval Lab"])
api_router.include_router(rag.router, prefix="/rag", tags=["RAG Chat & Generation"])
api_router.include_router(evaluation.router, prefix="/evaluation", tags=["Evaluation Lab & Benchmarking"])
api_router.include_router(conversations.router, prefix="/conversations", tags=["Conversational RAG & Memory"])
api_router.include_router(agents.router, prefix="/agents", tags=["Agentic RAG Lab"])
api_router.include_router(graph.router, prefix="/graph", tags=["Graph RAG & Knowledge Graph"])
api_router.include_router(multimodal.router, prefix="/multimodal", tags=["Multimodal RAG & Document Intelligence"])


