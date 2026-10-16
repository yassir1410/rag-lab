import uuid
from typing import List, Dict, Any, Optional, Tuple, Set
from sqlalchemy.orm import Session

from app.models.graph import GraphEntity, GraphRelation, GraphEvidence
from app.rag.graph.repository import BaseGraphRepository, postgres_graph_repository
from app.rag.graph.retrieval.entity_linker import entity_linker, EntityLinker
from app.rag.graph.retrieval.traversal import graph_traversal_engine, GraphTraversalEngine


class GraphRetriever:
    """
    Retrieves knowledge graph subgraphs, relation triples, and attached
    documentary evidence passages matching a natural language query.
    """

    def __init__(
        self,
        repository: Optional[BaseGraphRepository] = None,
        linker: Optional[EntityLinker] = None,
        traversal: Optional[GraphTraversalEngine] = None,
    ):
        self.repository = repository or postgres_graph_repository
        self.linker = linker or entity_linker
        self.traversal = traversal or graph_traversal_engine

    def retrieve(
        self,
        session: Session,
        query: str,
        max_hops: int = 2,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        # 1. Link entities in query
        linked_entities = self.linker.link_entities_in_query(session, query)

        # If no direct entity matched in query, fall back to top degree entities
        if not linked_entities:
            linked_entities = self.repository.list_entities(session, limit=2)

        collected_entities: Dict[uuid.UUID, GraphEntity] = {e.id: e for e in linked_entities}
        collected_relations: Dict[uuid.UUID, GraphRelation] = {}
        traversal_paths: List[List[str]] = []

        # 2. Traverse from linked entities
        for ent in linked_entities:
            nodes, edges, _ = self.traversal.traverse_neighborhood(
                session=session,
                start_entity_id=ent.id,
                max_depth=max_hops,
                max_nodes=15,
            )
            for n in nodes:
                collected_entities[n.id] = n
            for e in edges:
                collected_relations[e.id] = e

        # Formulate readable paths / triples
        all_node_map = {e.id: e for e in collected_entities.values()}
        for rel in collected_relations.values():
            s_name = all_node_map.get(rel.source_entity_id, None)
            t_name = all_node_map.get(rel.target_entity_id, None)
            if s_name and t_name:
                traversal_paths.append([
                    s_name.canonical_name,
                    rel.relation_type,
                    t_name.canonical_name,
                ])

        # 3. Retrieve attached evidence passages
        ent_ids = list(collected_entities.keys())
        rel_ids = list(collected_relations.keys())

        evidence_query = session.query(GraphEvidence)
        if ent_ids and rel_ids:
            evidence_query = evidence_query.filter(
                (GraphEvidence.entity_id.in_(ent_ids)) | (GraphEvidence.relation_id.in_(rel_ids))
            )
        elif ent_ids:
            evidence_query = evidence_query.filter(GraphEvidence.entity_id.in_(ent_ids))
        elif rel_ids:
            evidence_query = evidence_query.filter(GraphEvidence.relation_id.in_(rel_ids))

        evidence_items = evidence_query.limit(top_k * 2).all()

        # Deduplicate evidence text
        seen_texts = set()
        deduped_evidence: List[GraphEvidence] = []
        for ev in evidence_items:
            clean_snip = " ".join(ev.evidence_text.lower().split())[:80]
            if clean_snip not in seen_texts:
                seen_texts.add(clean_snip)
                deduped_evidence.append(ev)

        return {
            "entities": list(collected_entities.values()),
            "relations": list(collected_relations.values()),
            "evidence": deduped_evidence[:top_k],
            "traversal_paths": traversal_paths[:10],
            "linked_entities": linked_entities,
        }


graph_retriever = GraphRetriever()
