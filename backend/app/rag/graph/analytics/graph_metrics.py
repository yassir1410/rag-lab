from typing import Dict, Any, List
from sqlalchemy.orm import Session
from app.models.graph import GraphEntity, GraphRelation


class GraphMetricsCalculator:
    """
    Computes structural graph analytics and topological metrics.
    """

    def compute_metrics(self, session: Session) -> Dict[str, Any]:
        entities = session.query(GraphEntity).all()
        relations = session.query(GraphRelation).all()

        v = len(entities)
        e = len(relations)

        if v == 0:
            return {
                "nodes": 0,
                "edges": 0,
                "density": 0.0,
                "avg_degree": 0.0,
                "isolated_nodes": 0,
                "max_degree_entity": None,
            }

        # Calculate degree per node
        degree_map: Dict[str, int] = {str(ent.id): 0 for ent in entities}
        for rel in relations:
            s_id = str(rel.source_entity_id)
            t_id = str(rel.target_entity_id)
            if s_id in degree_map:
                degree_map[s_id] += 1
            if t_id in degree_map:
                degree_map[t_id] += 1

        density = (2.0 * e) / (v * (v - 1)) if v > 1 else 0.0
        avg_degree = sum(degree_map.values()) / v
        isolated = sum(1 for deg in degree_map.values() if deg == 0)

        # Max degree node
        max_deg_id = max(degree_map, key=degree_map.get) if degree_map else None
        max_node = next((ent for ent in entities if str(ent.id) == max_deg_id), None) if max_deg_id else None

        return {
            "nodes": v,
            "edges": e,
            "density": round(density, 4),
            "avg_degree": round(avg_degree, 2),
            "isolated_nodes": isolated,
            "max_degree_entity": {
                "name": max_node.canonical_name if max_node else "N/A",
                "type": max_node.entity_type if max_node else "N/A",
                "degree": degree_map.get(max_deg_id, 0) if max_deg_id else 0,
            } if max_node else None,
        }


graph_metrics_calculator = GraphMetricsCalculator()
