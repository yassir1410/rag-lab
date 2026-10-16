from typing import List, Dict, Set, Any
from sqlalchemy.orm import Session
from app.models.graph import GraphEntity, GraphRelation
from app.rag.graph.schemas import CommunitySummary


class CommunityDetector:
    """
    Detects thematic entity communities in the knowledge graph.
    Uses connected components / label propagation clustering and generates
    grounded summaries for global corpus exploration.
    """

    def detect_communities(self, session: Session) -> List[CommunitySummary]:
        entities = session.query(GraphEntity).all()
        relations = session.query(GraphRelation).all()

        if not entities:
            return []

        # Build adjacency graph
        adj: Dict[str, Set[str]] = {str(e.id): set() for e in entities}
        ent_by_id = {str(e.id): e for e in entities}

        for r in relations:
            s_id = str(r.source_entity_id)
            t_id = str(r.target_entity_id)
            if s_id in adj and t_id in adj:
                adj[s_id].add(t_id)
                adj[t_id].add(s_id)

        # BFS Connected Components
        visited = set()
        components: List[List[str]] = []

        for e_id in adj:
            if e_id not in visited:
                comp = []
                queue = [e_id]
                visited.add(e_id)
                while queue:
                    curr = queue.pop(0)
                    comp.append(curr)
                    for nbr in adj[curr]:
                        if nbr not in visited:
                            visited.add(nbr)
                            queue.append(nbr)
                components.append(comp)

        # Build Community summaries
        summaries: List[CommunitySummary] = []
        for idx, comp_nodes in enumerate(components, start=1):
            member_entities = [ent_by_id[nid] for nid in comp_nodes if nid in ent_by_id]
            member_names = [e.canonical_name for e in member_entities]
            entity_types = set(e.entity_type for e in member_entities)

            # Find internal relations
            comp_node_set = set(comp_nodes)
            key_rels = []
            for r in relations:
                if str(r.source_entity_id) in comp_node_set and str(r.target_entity_id) in comp_node_set:
                    s_name = ent_by_id[str(r.source_entity_id)].canonical_name
                    t_name = ent_by_id[str(r.target_entity_id)].canonical_name
                    key_rels.append(f"({s_name}) -[{r.relation_type}]-> ({t_name})")

            # Thematic cluster naming
            if "TECHNOLOGY" in entity_types and "JOB_ROLE" in entity_types:
                cluster_name = f"Core Technical Expertise & Tooling (Cluster {idx})"
            elif "ORGANIZATION" in entity_types:
                cluster_name = f"Academic & Institutional Background (Cluster {idx})"
            else:
                cluster_name = f"Knowledge Community {idx}"

            summary_text = (
                f"Community contains {len(member_entities)} entities across types: {', '.join(entity_types)}. "
                f"Key member nodes include: {', '.join(member_names[:6])}. "
                f"Captures {len(key_rels)} verified relational paths in the documentary corpus."
            )

            summaries.append(
                CommunitySummary(
                    community_id=f"comm_{idx}",
                    name=cluster_name,
                    member_entities=member_names,
                    key_relations=key_rels[:8],
                    summary_text=summary_text,
                    size=len(member_names),
                )
            )

        return sorted(summaries, key=lambda c: c.size, reverse=True)


community_detector = CommunityDetector()
