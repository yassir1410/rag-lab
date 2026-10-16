from datetime import datetime, timezone
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Lightweight liveness probe response."""
    status: str = Field(default="ok")
    version: str = Field(default="0.2.0")
    environment: str = Field(default="development")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ReadinessResponse(BaseModel):
    """Deep readiness probe response checking external dependencies."""
    status: str = Field(default="ready")
    database_connected: bool
    pgvector_available: bool
    pgvector_version: Optional[str] = None
    storage_writable: bool
    details: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
