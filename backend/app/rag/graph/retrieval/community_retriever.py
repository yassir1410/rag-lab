import uuid
from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session

from app.rag.graph.analytics.community_detection import community_detector, CommunityDetector
from app.models.graph import GraphEvidence
from app.models.document import Document


class CommunityRetriever:
    """
    Executes global thematic retrieval across graph communities.
    Matches queries to entity clusters and injects community summaries and key relation triples.
    """

    def __init__(self, detector: Optional[CommunityDetector] = None):
        self.detector = detector or community_detector

    def retrieve(
        self,
        session: Session,
        query: str,
        top_k: int = 3,
    ) -> Dict[str, Any]:
        communities = self.detector.detect_communities(session)
        if not communities:
            return {"entities": [], "relations": [], "evidence": [], "traversal_paths": []}

        # Score communities by keyword overlap
        q_terms = set(query.lower().split())
        scored_comms = []
        for comm in communities:
            comm_terms = set(comm.name.lower().split()).union(
                *[set(e.lower().split()) for e in comm.member_entities]
            )
            overlap = len(q_terms.intersection(comm_terms))
            scored_comms.append((overlap, comm))

        # Sort by overlap descending, then size descending
        scored_comms.sort(key=lambda x: (x[0], x[1].size), reverse=True)
        selected_comms = [c for _, c in scored_comms[:top_k]]

        # Turn community summaries into synthetic evidence items
        doc = session.query(Document).first()
        doc_id = doc.id if doc else uuid.uuid4()

        evidence_items: List[GraphEvidence] = []
        for comm in selected_comms:
            content = f"Community Summary: {comm.name}\n{comm.summary_text}\nKey Relations: {'; '.join(comm.key_relations)}"
            evidence_items.append(
                GraphEvidence(
                    id=uuid.uuid4(),
                    document_id=doc_id,
                    evidence_text=content,
                    page_reference=1,
                )
            )

        traversal_paths = []
        for comm in selected_comms:
            for rel_str in comm.key_relations[:4]:
                traversal_paths.append([rel_str])

        return {
            "entities": [],
            "relations": [],
            "evidence": evidence_items,
            "traversal_paths": traversal_paths,
            "communities": selected_comms,
        }


community_retriever = CommunityRetriever()
