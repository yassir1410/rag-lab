from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.api import api_router
from app.core.config import settings
from app.core.logging import setup_logging, logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    setup_logging()
    logger.info(f"Starting {settings.PROJECT_NAME} v{settings.VERSION} [{settings.ENVIRONMENT}]")
    storage_dir = settings.ABSOLUTE_STORAGE_DIR
    logger.info(f"Upload storage directory configured at: {storage_dir}")
    yield
    # Shutdown
    logger.info("Shutting down application...")


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description=(
        "Comprehensive educational Retrieval-Augmented Generation (RAG) laboratory. "
        "Phase 01: Project Foundation & Architecture. "
        "Phase 02: Document Ingestion, Parsing, Normalization & Metadata Extraction. "
        "Phase 03: Chunking Strategies Lab. "
        "Phase 04: Embeddings Engineering Lab. "
        "Phase 05: Vector Database & Semantic Search. "
        "Phase 06: Naive RAG & Grounded LLM Generation."
    ),
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url=f"{settings.API_V1_STR}/docs",
    redoc_url=f"{settings.API_V1_STR}/redoc",
    lifespan=lifespan,
)

# CORS middleware
if settings.CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Include API routes
app.include_router(api_router, prefix="/api")


@app.get("/", tags=["Root"])
def root():
    return {
        "project": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "phase": "06 - Naive RAG & Grounded LLM Generation",
        "docs_url": f"{settings.API_V1_STR}/docs",
        "health_check": f"{settings.API_V1_STR}/health",
        "readiness_check": f"{settings.API_V1_STR}/health/ready",
    }
