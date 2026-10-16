import uuid
from typing import List, Dict, Any, Optional, Tuple
from sqlalchemy.orm import Session
from app.models.graph import GraphEntity, GraphRelation
from app.rag.graph.repository import BaseGraphRepository, postgres_graph_repository


class GraphTraversalEngine:
    """
    Executes bounded graph traversals and finds paths between entities.
    Prevents cyclic runaway queries with depth and node caps.
    """

    def __init__(self, repository: Optional[BaseGraphRepository] = None):
        self.repository = repository or postgres_graph_repository

    def traverse_neighborhood(
        self,
        session: Session,
        start_entity_id: uuid.UUID,
        max_depth: int = 2,
        allowed_relations: Optional[List[str]] = None,
        max_nodes: int = 50,
    ) -> Tuple[List[GraphEntity], List[GraphRelation], int]:
        return self.repository.traverse(
            session=session,
            start_entity_id=start_entity_id,
            max_depth=max_depth,
            allowed_relations=allowed_relations,
            max_nodes=max_nodes,
        )

    def find_paths_between(
        self,
        session: Session,
        source_id: uuid.UUID,
        target_id: uuid.UUID,
        max_depth: int = 3,
    ) -> List[Dict[str, Any]]:
        return self.repository.find_paths(
            session=session,
            source_id=source_id,
            target_id=target_id,
            max_depth=max_depth,
        )


graph_traversal_engine = GraphTraversalEngine()
