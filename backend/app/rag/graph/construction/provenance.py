import uuid
from typing import Optional
from sqlalchemy.orm import Session
from app.models.graph import GraphEvidence
from app.rag.graph.repository import BaseGraphRepository


class ProvenanceTracker:
    """
    Guarantees strict provenance tracking for every Knowledge Graph entity and relation.
    Preserves exact document identifiers, chunk IDs, page numbers, and verbatim evidence snippets.
    """

    def __init__(self, repository: BaseGraphRepository):
        self.repository = repository

    def attach_entity_evidence(
        self,
        session: Session,
        entity_id: uuid.UUID,
        document_id: uuid.UUID,
        evidence_text: str,
        chunk_id: Optional[uuid.UUID] = None,
        section_id: Optional[uuid.UUID] = None,
        page_reference: Optional[int] = None,
        run_id: Optional[uuid.UUID] = None,
    ) -> GraphEvidence:
        return self.repository.add_evidence(
            session=session,
            document_id=document_id,
            evidence_text=evidence_text,
            entity_id=entity_id,
            chunk_id=chunk_id,
            section_id=section_id,
            page_reference=page_reference,
            run_id=run_id,
        )

    def attach_relation_evidence(
        self,
        session: Session,
        relation_id: uuid.UUID,
        document_id: uuid.UUID,
        evidence_text: str,
        chunk_id: Optional[uuid.UUID] = None,
        section_id: Optional[uuid.UUID] = None,
        page_reference: Optional[int] = None,
        run_id: Optional[uuid.UUID] = None,
    ) -> GraphEvidence:
        return self.repository.add_evidence(
            session=session,
            document_id=document_id,
            evidence_text=evidence_text,
            relation_id=relation_id,
            chunk_id=chunk_id,
            section_id=section_id,
            page_reference=page_reference,
            run_id=run_id,
        )
