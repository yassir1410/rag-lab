import re
from typing import List, Optional
from sqlalchemy.orm import Session
from app.models.graph import GraphEntity
from app.rag.graph.repository import BaseGraphRepository, postgres_graph_repository
from app.rag.graph.resolution.entity_normalizer import entity_normalizer


class EntityLinker:
    """
    Links entity mentions within natural language queries to canonical Knowledge Graph nodes.
    Supports canonical name matching, alias lookups, and normalized token matching.
    """

    def __init__(self, repository: Optional[BaseGraphRepository] = None):
        self.repository = repository or postgres_graph_repository

    def link_entities_in_query(
        self, session: Session, query: str
    ) -> List[GraphEntity]:
        all_entities = session.query(GraphEntity).all()
        linked: List[GraphEntity] = []
        seen_ids = set()

        q_lower = entity_normalizer.normalize_text(query).lower()

        for ent in all_entities:
            c_name = ent.canonical_name.lower().strip()
            # Word boundary check for canonical name
            pattern = r"\b" + re.escape(c_name) + r"\b"
            if re.search(pattern, q_lower):
                if ent.id not in seen_ids:
                    linked.append(ent)
                    seen_ids.add(ent.id)
                continue

            # Alias check
            for alias in (ent.aliases or []):
                a_name = alias.lower().strip()
                alias_pattern = r"\b" + re.escape(a_name) + r"\b"
                if re.search(alias_pattern, q_lower):
                    if ent.id not in seen_ids:
                        linked.append(ent)
                        seen_ids.add(ent.id)
                    break

        return linked


entity_linker = EntityLinker()
