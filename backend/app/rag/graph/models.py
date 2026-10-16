import uuid
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class ExtractedEntityCandidate:
    """Internal candidate representation produced during entity extraction."""
    canonical_name: str
    entity_type: str
    aliases: List[str] = field(default_factory=list)
    properties: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    evidence_text: str = ""
    chunk_id: Optional[uuid.UUID] = None
    page_reference: Optional[int] = None
    section_id: Optional[uuid.UUID] = None


@dataclass
class ExtractedRelationCandidate:
    """Internal candidate representation produced during relation extraction."""
    source_name: str
    target_name: str
    relation_type: str
    properties: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    evidence_text: str = ""
    chunk_id: Optional[uuid.UUID] = None
    page_reference: Optional[int] = None
    section_id: Optional[uuid.UUID] = None


@dataclass
class GraphExtractionResult:
    """Result aggregation produced by extraction strategies."""
    entities: List[ExtractedEntityCandidate] = field(default_factory=list)
    relations: List[ExtractedRelationCandidate] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    extraction_method: str = "rule"
    metrics: Dict[str, Any] = field(default_factory=dict)
