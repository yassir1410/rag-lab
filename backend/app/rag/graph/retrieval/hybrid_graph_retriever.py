import uuid
from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session

from app.rag.graph.retrieval.graph_retriever import graph_retriever, GraphRetriever
from app.rag.retrieval.hybrid import HybridRetrievalEngine
from app.rag.retrieval.schemas import SearchRequest
from app.models.graph import GraphEvidence
from app.models.chunking import DocumentChunk


class HybridGraphRetriever:
    """
    Fuses Knowledge Graph traversal results with Phase 08 dense/hybrid vector retrieval.
    Applies Reciprocal Rank Fusion (RRF) to combine structured relational evidence
    with unstructured semantic similarity.
    """

    def __init__(
        self,
        graph_engine: Optional[GraphRetriever] = None,
        vector_engine: Optional[HybridRetrievalEngine] = None,
        rrf_k: int = 60,
    ):
        self.graph_engine = graph_engine or graph_retriever
        self.vector_engine = vector_engine or HybridRetrievalEngine()
        self.rrf_k = rrf_k

    def retrieve(
        self,
        session: Session,
        query: str,
        max_hops: int = 2,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        # 1. Graph retrieval
        graph_res = self.graph_engine.retrieve(session, query, max_hops=max_hops, top_k=top_k)
        graph_evidence = graph_res.get("evidence", [])

        # 2. Vector retrieval
        vector_req = SearchRequest(
            query=query,
            strategy="dense_only",
            top_k=top_k,
            explain_plan=False,
        )
        try:
            vector_res = self.vector_engine.execute(session, vector_req)
            vector_items = vector_res.results
        except Exception:
            vector_items = []

        # 3. Reciprocal Rank Fusion
        fused_scores: Dict[str, float] = {}
        chunk_map: Dict[str, Dict[str, Any]] = {}

        # Add graph evidence ranks
        for rank, ev in enumerate(graph_evidence, start=1):
            key = str(ev.chunk_id) if ev.chunk_id else f"ev_{ev.id}"
            score = 1.0 / (self.rrf_k + rank)
            fused_scores[key] = fused_scores.get(key, 0.0) + score
            chunk_map[key] = {
                "chunk_id": ev.chunk_id,
                "document_id": ev.document_id,
                "content": ev.evidence_text,
                "page_reference": ev.page_reference,
                "source_type": "graph",
            }

        # Add vector results ranks
        for rank, v_item in enumerate(vector_items, start=1):
            key = str(v_item.chunk_id)
            score = 1.0 / (self.rrf_k + rank)
            fused_scores[key] = fused_scores.get(key, 0.0) + score
            if key not in chunk_map:
                chunk_map[key] = {
                    "chunk_id": v_item.chunk_id,
                    "document_id": v_item.document_id,
                    "content": v_item.content,
                    "page_reference": v_item.sequence_index,
                    "source_type": "vector",
                }

        # Sort by fused score
        sorted_keys = sorted(fused_scores.keys(), key=lambda k: fused_scores[k], reverse=True)

        # Build fused evidence list
        fused_evidence: List[GraphEvidence] = []
        for k in sorted_keys[:top_k]:
            meta = chunk_map[k]
            fused_evidence.append(
                GraphEvidence(
                    id=uuid.uuid4(),
                    document_id=meta["document_id"],
                    chunk_id=meta["chunk_id"],
                    evidence_text=meta["content"],
                    page_reference=meta["page_reference"],
                )
            )

        return {
            "entities": graph_res.get("entities", []),
            "relations": graph_res.get("relations", []),
            "evidence": fused_evidence,
            "traversal_paths": graph_res.get("traversal_paths", []),
            "linked_entities": graph_res.get("linked_entities", []),
            "fused_count": len(fused_evidence),
        }


hybrid_graph_retriever = HybridGraphRetriever()
