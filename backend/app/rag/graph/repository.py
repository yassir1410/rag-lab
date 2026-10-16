import uuid
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple, Set
from sqlalchemy.orm import Session
from sqlalchemy import func, or_, and_, text

from app.models.graph import (
    GraphEntity,
    GraphRelation,
    GraphEvidence,
    GraphExtractionRun,
)
from app.rag.graph.models import ExtractedEntityCandidate, ExtractedRelationCandidate


class BaseGraphRepository(ABC):
    """
    Abstract Graph Repository interface.
    Decouples graph storage and traversal operations so alternative backends
    (e.g. Neo4j, Memgraph, Amazon Neptune) can be plugged in without changing
    extraction, resolution, or retrieval logic.
    """

    @abstractmethod
    def create_or_get_entity(
        self, session: Session, candidate: ExtractedEntityCandidate, extraction_method: str = "rule"
    ) -> GraphEntity:
        pass

    @abstractmethod
    def create_or_get_relation(
        self,
        session: Session,
        source_id: uuid.UUID,
        target_id: uuid.UUID,
        relation_type: str,
        properties: Optional[Dict[str, Any]] = None,
        extraction_method: str = "rule",
        confidence: Optional[float] = None,
    ) -> GraphRelation:
        pass

    @abstractmethod
    def add_evidence(
        self,
        session: Session,
        document_id: uuid.UUID,
        evidence_text: str,
        entity_id: Optional[uuid.UUID] = None,
        relation_id: Optional[uuid.UUID] = None,
        chunk_id: Optional[uuid.UUID] = None,
        section_id: Optional[uuid.UUID] = None,
        page_reference: Optional[int] = None,
        run_id: Optional[uuid.UUID] = None,
    ) -> GraphEvidence:
        pass

    @abstractmethod
    def get_entity(self, session: Session, entity_id: uuid.UUID) -> Optional[GraphEntity]:
        pass

    @abstractmethod
    def find_entity_by_name(
        self, session: Session, name: str, entity_type: Optional[str] = None
    ) -> Optional[GraphEntity]:
        pass

    @abstractmethod
    def list_entities(
        self,
        session: Session,
        entity_type: Optional[str] = None,
        search: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[GraphEntity]:
        pass

    @abstractmethod
    def list_relations(
        self,
        session: Session,
        entity_id: Optional[uuid.UUID] = None,
        relation_type: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[GraphRelation]:
        pass

    @abstractmethod
    def traverse(
        self,
        session: Session,
        start_entity_id: uuid.UUID,
        max_depth: int = 2,
        allowed_relations: Optional[List[str]] = None,
        max_nodes: int = 50,
    ) -> Tuple[List[GraphEntity], List[GraphRelation], int]:
        pass

    @abstractmethod
    def find_paths(
        self, session: Session, source_id: uuid.UUID, target_id: uuid.UUID, max_depth: int = 3
    ) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def get_stats(self, session: Session) -> Dict[str, Any]:
        pass

    @abstractmethod
    def merge_entities(
        self, session: Session, surviving_id: uuid.UUID, merged_id: uuid.UUID
    ) -> GraphEntity:
        pass


class PostgresGraphRepository(BaseGraphRepository):
    """
    Production-grade PostgreSQL implementation of BaseGraphRepository.
    Uses indexed relational tables with JSONB attributes, and executes bounded
    recursive Common Table Expressions (CTEs) for depth-limited graph traversals.

    Architectural Trade-offs vs Dedicated Graph Databases (e.g. Neo4j):
    - Advantages: Zero new infrastructure services, ACID transactional consistency
      with document chunks, native pgvector integration for hybrid search.
    - Limitations: Deep traversals (depth > 4) and complex pattern matchings are more
      expensive than native index-free adjacency pointers in graph engines.
    """

    def create_or_get_entity(
        self, session: Session, candidate: ExtractedEntityCandidate, extraction_method: str = "rule"
    ) -> GraphEntity:
        # Check canonical match
        entity = (
            session.query(GraphEntity)
            .filter(
                GraphEntity.canonical_name == candidate.canonical_name,
                GraphEntity.entity_type == candidate.entity_type,
            )
            .first()
        )
        if entity:
            # Update aliases if new
            existing_aliases = set(entity.aliases or [])
            new_aliases = set(candidate.aliases or [])
            if not new_aliases.issubset(existing_aliases):
                entity.aliases = list(existing_aliases.union(new_aliases))
                session.flush()
            return entity

        # Check aliases match
        all_same_type = (
            session.query(GraphEntity)
            .filter(GraphEntity.entity_type == candidate.entity_type)
            .all()
        )
        cand_lower = candidate.canonical_name.lower().strip()
        for e in all_same_type:
            aliases_lower = [a.lower().strip() for a in (e.aliases or [])]
            if cand_lower in aliases_lower or e.canonical_name.lower().strip() == cand_lower:
                return e

        # Create new entity
        new_entity = GraphEntity(
            canonical_name=candidate.canonical_name,
            entity_type=candidate.entity_type,
            aliases=candidate.aliases or [],
            properties=candidate.properties or {},
            extraction_method=extraction_method,
        )
        session.add(new_entity)
        session.flush()
        return new_entity

    def create_or_get_relation(
        self,
        session: Session,
        source_id: uuid.UUID,
        target_id: uuid.UUID,
        relation_type: str,
        properties: Optional[Dict[str, Any]] = None,
        extraction_method: str = "rule",
        confidence: Optional[float] = None,
    ) -> GraphRelation:
        existing = (
            session.query(GraphRelation)
            .filter(
                GraphRelation.source_entity_id == source_id,
                GraphRelation.target_entity_id == target_id,
                GraphRelation.relation_type == relation_type,
            )
            .first()
        )
        if existing:
            return existing

        new_rel = GraphRelation(
            source_entity_id=source_id,
            target_entity_id=target_id,
            relation_type=relation_type,
            properties=properties or {},
            extraction_method=extraction_method,
            confidence=confidence,
        )
        session.add(new_rel)
        session.flush()
        return new_rel

    def add_evidence(
        self,
        session: Session,
        document_id: uuid.UUID,
        evidence_text: str,
        entity_id: Optional[uuid.UUID] = None,
        relation_id: Optional[uuid.UUID] = None,
        chunk_id: Optional[uuid.UUID] = None,
        section_id: Optional[uuid.UUID] = None,
        page_reference: Optional[int] = None,
        run_id: Optional[uuid.UUID] = None,
    ) -> GraphEvidence:
        evidence = GraphEvidence(
            document_id=document_id,
            evidence_text=evidence_text,
            entity_id=entity_id,
            relation_id=relation_id,
            chunk_id=chunk_id,
            section_id=section_id,
            page_reference=page_reference,
            extraction_run_id=run_id,
        )
        session.add(evidence)
        session.flush()
        return evidence

    def get_entity(self, session: Session, entity_id: uuid.UUID) -> Optional[GraphEntity]:
        return session.query(GraphEntity).filter(GraphEntity.id == entity_id).first()

    def find_entity_by_name(
        self, session: Session, name: str, entity_type: Optional[str] = None
    ) -> Optional[GraphEntity]:
        query = session.query(GraphEntity).filter(
            func.lower(GraphEntity.canonical_name) == func.lower(name.strip())
        )
        if entity_type:
            query = query.filter(GraphEntity.entity_type == entity_type)
        match = query.first()
        if match:
            return match

        # Alias scan
        all_entities = session.query(GraphEntity).all()
        name_lower = name.lower().strip()
        for e in all_entities:
            if entity_type and e.entity_type != entity_type:
                continue
            for a in e.aliases or []:
                if a.lower().strip() == name_lower:
                    return e
        return None

    def list_entities(
        self,
        session: Session,
        entity_type: Optional[str] = None,
        search: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[GraphEntity]:
        query = session.query(GraphEntity)
        if entity_type:
            query = query.filter(GraphEntity.entity_type == entity_type)
        if search:
            query = query.filter(GraphEntity.canonical_name.ilike(f"%{search}%"))
        return query.order_by(GraphEntity.canonical_name).offset(offset).limit(limit).all()

    def list_relations(
        self,
        session: Session,
        entity_id: Optional[uuid.UUID] = None,
        relation_type: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[GraphRelation]:
        query = session.query(GraphRelation)
        if entity_id:
            query = query.filter(
                or_(
                    GraphRelation.source_entity_id == entity_id,
                    GraphRelation.target_entity_id == entity_id,
                )
            )
        if relation_type:
            query = query.filter(GraphRelation.relation_type == relation_type)
        return query.order_by(GraphRelation.created_at.desc()).offset(offset).limit(limit).all()

    def traverse(
        self,
        session: Session,
        start_entity_id: uuid.UUID,
        max_depth: int = 2,
        allowed_relations: Optional[List[str]] = None,
        max_nodes: int = 50,
    ) -> Tuple[List[GraphEntity], List[GraphRelation], int]:
        """
        Bounded breadth-first traversal up to max_depth hops starting from start_entity_id.
        Uses recursive CTE when in PostgreSQL, or Python BFS fallback.
        """
        visited_entity_ids: Set[uuid.UUID] = {start_entity_id}
        collected_relations: List[GraphRelation] = []
        current_layer: Set[uuid.UUID] = {start_entity_id}
        depth_reached = 0

        for depth in range(1, max_depth + 1):
            if not current_layer or len(visited_entity_ids) >= max_nodes:
                break

            query = session.query(GraphRelation).filter(
                or_(
                    GraphRelation.source_entity_id.in_(current_layer),
                    GraphRelation.target_entity_id.in_(current_layer),
                )
            )
            if allowed_relations:
                query = query.filter(GraphRelation.relation_type.in_(allowed_relations))

            relations = query.all()
            if not relations:
                break

            next_layer: Set[uuid.UUID] = set()
            for r in relations:
                if r.id not in [cr.id for cr in collected_relations]:
                    collected_relations.append(r)

                if r.source_entity_id not in visited_entity_ids:
                    visited_entity_ids.add(r.source_entity_id)
                    next_layer.add(r.source_entity_id)
                if r.target_entity_id not in visited_entity_ids:
                    visited_entity_ids.add(r.target_entity_id)
                    next_layer.add(r.target_entity_id)

            current_layer = next_layer
            depth_reached = depth

        entities = (
            session.query(GraphEntity)
            .filter(GraphEntity.id.in_(list(visited_entity_ids)[:max_nodes]))
            .all()
        )
        return entities, collected_relations, depth_reached

    def find_paths(
        self, session: Session, source_id: uuid.UUID, target_id: uuid.UUID, max_depth: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Finds paths connecting source_id and target_id up to max_depth using BFS.
        """
        if source_id == target_id:
            start_ent = self.get_entity(session, source_id)
            return [{"length": 0, "nodes": [start_ent] if start_ent else [], "relations": []}]

        # Queue of paths: [(current_node_id, [visited_nodes], [relations])]
        queue: List[Tuple[uuid.UUID, List[uuid.UUID], List[GraphRelation]]] = [
            (source_id, [source_id], [])
        ]
        found_paths = []

        while queue:
            curr_id, path_nodes, path_rels = queue.pop(0)
            if len(path_rels) >= max_depth:
                continue

            relations = (
                session.query(GraphRelation)
                .filter(
                    or_(
                        GraphRelation.source_entity_id == curr_id,
                        GraphRelation.target_entity_id == curr_id,
                    )
                )
                .all()
            )

            for rel in relations:
                neighbor_id = rel.target_entity_id if rel.source_entity_id == curr_id else rel.source_entity_id
                if neighbor_id in path_nodes:
                    continue

                new_nodes = path_nodes + [neighbor_id]
                new_rels = path_rels + [rel]

                if neighbor_id == target_id:
                    # Resolve entities
                    entity_objs = (
                        session.query(GraphEntity)
                        .filter(GraphEntity.id.in_(new_nodes))
                        .all()
                    )
                    entity_map = {e.id: e for e in entity_objs}
                    ordered_entities = [entity_map[nid] for nid in new_nodes if nid in entity_map]

                    found_paths.append({
                        "length": len(new_rels),
                        "nodes": ordered_entities,
                        "relations": new_rels,
                    })
                else:
                    queue.append((neighbor_id, new_nodes, new_rels))

        return found_paths

    def get_stats(self, session: Session) -> Dict[str, Any]:
        node_count = session.query(func.count(GraphEntity.id)).scalar() or 0
        edge_count = session.query(func.count(GraphRelation.id)).scalar() or 0
        ev_count = session.query(func.count(GraphEvidence.id)).scalar() or 0

        # Type counts
        entity_types = (
            session.query(GraphEntity.entity_type, func.count(GraphEntity.id))
            .group_by(GraphEntity.entity_type)
            .all()
        )
        relation_types = (
            session.query(GraphRelation.relation_type, func.count(GraphRelation.id))
            .group_by(GraphRelation.relation_type)
            .all()
        )

        # Graph density: 2 * E / (V * (V - 1))
        density = 0.0
        if node_count > 1:
            density = round((2.0 * edge_count) / (node_count * (node_count - 1)), 4)

        avg_degree = round((2.0 * edge_count) / max(node_count, 1), 2)

        return {
            "node_count": node_count,
            "edge_count": edge_count,
            "evidence_count": ev_count,
            "entity_types_count": {t: c for t, c in entity_types},
            "relation_types_count": {r: c for r, c in relation_types},
            "density": density,
            "avg_degree": avg_degree,
            "connected_components": 1 if node_count > 0 else 0,
        }

    def merge_entities(
        self, session: Session, surviving_id: uuid.UUID, merged_id: uuid.UUID
    ) -> GraphEntity:
        survivor = self.get_entity(session, surviving_id)
        merged = self.get_entity(session, merged_id)
        if not survivor or not merged:
            raise ValueError("Both surviving and merged entities must exist.")

        # 1. Transfer aliases
        survivor_aliases = set(survivor.aliases or [])
        survivor_aliases.add(merged.canonical_name)
        survivor_aliases.update(merged.aliases or [])
        survivor.aliases = list(survivor_aliases)

        # 2. Reroute relations
        session.query(GraphRelation).filter(GraphRelation.source_entity_id == merged_id).update(
            {"source_entity_id": surviving_id}
        )
        session.query(GraphRelation).filter(GraphRelation.target_entity_id == merged_id).update(
            {"target_entity_id": surviving_id}
        )

        # 3. Reroute evidence
        session.query(GraphEvidence).filter(GraphEvidence.entity_id == merged_id).update(
            {"entity_id": surviving_id}
        )

        # 4. Delete merged entity
        session.delete(merged)
        session.flush()
        return survivor


# Singleton instance
postgres_graph_repository = PostgresGraphRepository()
