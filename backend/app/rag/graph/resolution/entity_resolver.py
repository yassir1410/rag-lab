import difflib
from typing import List, Optional, Tuple
from sqlalchemy.orm import Session

from app.models.graph import GraphEntity
from app.rag.graph.schemas import EntityMergeCandidate, EntityResponse
from app.rag.graph.resolution.entity_normalizer import entity_normalizer


class EntityResolver:
    """
    Entity Resolution and Disambiguation Engine.
    Identifies duplicate entity candidates, evaluates alias similarity,
    and flags ambiguous merge candidates for human review.
    """

    def __init__(self, fuzzy_threshold: float = 0.88):
        self.fuzzy_threshold = fuzzy_threshold

    def calculate_similarity(self, name_a: str, name_b: str) -> float:
        a_norm = entity_normalizer.canonicalize(name_a).lower()
        b_norm = entity_normalizer.canonicalize(name_b).lower()

        if a_norm == b_norm:
            return 1.0

        # Substring exact check
        if len(a_norm) > 4 and len(b_norm) > 4:
            if a_norm in b_norm or b_norm in a_norm:
                return 0.90

        # Sequence matcher ratio
        return difflib.SequenceMatcher(None, a_norm, b_norm).ratio()

    def find_merge_candidates(self, session: Session) -> List[EntityMergeCandidate]:
        """
        Scans all entities in the database and identifies candidate duplicates for review.
        """
        entities = session.query(GraphEntity).all()
        candidates: List[EntityMergeCandidate] = []
        n = len(entities)

        for i in range(n):
            for j in range(i + 1, n):
                e1 = entities[i]
                e2 = entities[j]

                # Only compare entities of the same type or compatible types
                if e1.entity_type != e2.entity_type:
                    continue

                sim = self.calculate_similarity(e1.canonical_name, e2.canonical_name)

                # Check alias overlap
                aliases_1 = set([a.lower() for a in (e1.aliases or [])])
                aliases_2 = set([a.lower() for a in (e2.aliases or [])])
                alias_overlap = bool(aliases_1.intersection(aliases_2))
                if alias_overlap:
                    sim = max(sim, 0.95)

                if sim >= self.fuzzy_threshold:
                    # Ambiguity guard: PERSON and ORGANIZATION require human review
                    requires_review = e1.entity_type in ("PERSON", "ORGANIZATION") or sim < 0.95
                    reason = (
                        "High string similarity"
                        if not alias_overlap
                        else "Shared alias match"
                    )

                    candidates.append(
                        EntityMergeCandidate(
                            source_entity=EntityResponse(
                                id=e1.id,
                                canonical_name=e1.canonical_name,
                                entity_type=e1.entity_type,
                                aliases=e1.aliases or [],
                                properties=e1.properties or {},
                                extraction_method=e1.extraction_method,
                                created_at=e1.created_at.isoformat() if e1.created_at else "",
                                updated_at=e1.updated_at.isoformat() if e1.updated_at else "",
                                evidence_count=len(e1.evidence_items or []),
                                degree=len(e1.outgoing_relations or []) + len(e1.incoming_relations or []),
                            ),
                            target_entity=EntityResponse(
                                id=e2.id,
                                canonical_name=e2.canonical_name,
                                entity_type=e2.entity_type,
                                aliases=e2.aliases or [],
                                properties=e2.properties or {},
                                extraction_method=e2.extraction_method,
                                created_at=e2.created_at.isoformat() if e2.created_at else "",
                                updated_at=e2.updated_at.isoformat() if e2.updated_at else "",
                                evidence_count=len(e2.evidence_items or []),
                                degree=len(e2.outgoing_relations or []) + len(e2.incoming_relations or []),
                            ),
                            similarity_score=round(sim, 3),
                            match_reason=reason,
                            requires_human_review=requires_review,
                        )
                    )

        return candidates


entity_resolver = EntityResolver()
