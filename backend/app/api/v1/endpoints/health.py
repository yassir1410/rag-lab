import os
import tempfile
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, status, Response
from sqlalchemy import text
from sqlalchemy.orm import Session
from app.core.config import settings
from app.db.session import get_db
from app.schemas.health import HealthResponse, ReadinessResponse

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Liveness Health Check",
    description="Lightweight liveness probe returning application running status.",
)
def get_health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        version=settings.VERSION,
        environment=settings.ENVIRONMENT,
        timestamp=datetime.now(timezone.utc),
    )


@router.get(
    "/health/ready",
    response_model=ReadinessResponse,
    summary="Readiness Probe",
    description="Deep readiness probe verifying database connectivity, pgvector extension, and storage.",
)
def get_readiness(
    response: Response,
    db: Session = Depends(get_db),
) -> ReadinessResponse:
    db_connected = False
    pgvector_available = False
    pgvector_version = None
    storage_writable = False
    details = {}

    # 1. Database connectivity check
    try:
        db.execute(text("SELECT 1"))
        db_connected = True
        details["database"] = "Connected successfully"
    except Exception as e:
        details["database"] = f"Database connection error: {str(e)}"

    # 2. pgvector extension check
    if db_connected:
        try:
            result = db.execute(
                text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
            ).fetchone()
            if result:
                pgvector_available = True
                pgvector_version = result[0]
                details["pgvector"] = f"pgvector extension installed (v{pgvector_version})"
            else:
                details["pgvector"] = "pgvector extension is NOT installed in database"
        except Exception as e:
            details["pgvector"] = f"Error checking pgvector: {str(e)}"

    # 3. Storage filesystem check
    try:
        storage_dir = settings.ABSOLUTE_STORAGE_DIR
        test_file = storage_dir / ".healthcheck_tmp"
        test_file.write_text("health")
        test_file.unlink()
        storage_writable = True
        details["storage"] = f"Upload directory writable ({storage_dir})"
    except Exception as e:
        details["storage"] = f"Storage write error: {str(e)}"

    is_ready = db_connected and storage_writable

    if not is_ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return ReadinessResponse(
        status="ready" if is_ready else "unhealthy",
        database_connected=db_connected,
        pgvector_available=pgvector_available,
        pgvector_version=pgvector_version,
        storage_writable=storage_writable,
        details=details,
        timestamp=datetime.now(timezone.utc),
    )
