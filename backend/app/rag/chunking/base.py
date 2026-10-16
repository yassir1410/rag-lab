from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any, Union
import uuid


@dataclass
class ChunkMetadata:
    """Detailed structural, offset, and hierarchical metadata for a chunk."""
    start_char: Optional[int] = None
    end_char: Optional[int] = None
    page_numbers: List[int] = field(default_factory=list)
    headings: List[str] = field(default_factory=list)
    source_section_ids: List[str] = field(default_factory=list)
    content_type: str = "text"  # text, table, code, heading
    parent_chunk_id: Optional[str] = None
    child_chunk_ids: List[str] = field(default_factory=list)
    breakpoint_distance: Optional[float] = None  # for semantic chunking
    custom: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    """Core chunk entity representing a retrievable or contextual segment."""
    id: uuid.UUID
    document_id: uuid.UUID
    sequence_index: int
    content: str
    char_count: int
    token_count: int
    metadata: ChunkMetadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ChunkingConfig:
    """Base configuration parameters for chunking algorithms."""
    chunk_size: int = 500
    chunk_overlap: int = 50
    measurement_unit: str = "character"  # "character" or "token"
    extra_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChunkingResult:
    """Result returned by any chunking strategy."""
    strategy: str
    chunks: List[Chunk]
    metrics: Dict[str, Any]
    config: Dict[str, Any]
    processing_duration_ms: float


class ChunkingStrategy(ABC):
    """Abstract interface that all concrete chunking algorithms must implement."""

    @property
    @abstractmethod
    def strategy_name(self) -> str:
        """Machine identifier (e.g. 'fixed_size', 'recursive')."""
        pass

    @property
    @abstractmethod
    def display_name(self) -> str:
        """Human-readable name."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Pedagogical description of the strategy."""
        pass

    @property
    @abstractmethod
    def default_config(self) -> Dict[str, Any]:
        """Default configuration parameters."""
        pass

    @abstractmethod
    def validate_config(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Validate configuration values and return normalized config dictionary."""
        pass

    @abstractmethod
    def split(
        self,
        text: str,
        document_id: uuid.UUID,
        config_dict: Dict[str, Any],
        document_context: Optional[Dict[str, Any]] = None,
    ) -> ChunkingResult:
        """Split document text into structured chunks with metrics."""
        pass
