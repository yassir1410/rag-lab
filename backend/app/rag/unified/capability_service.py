import uuid
from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import select, func

from app.models.document import Document
from app.models.chunking import DocumentChunk
from app.models.embeddings import ChunkEmbedding
from app.models.multimodal import DocumentPageAsset, DocumentElement, MultimodalEmbedding
from app.models.graph import GraphEvidence
from app.rag.unified.schemas import DocumentCapabilityReport


class DocumentCapabilityService:
    """
    Audits indexing state and capability readiness per document and globally.
    Prevents issuing queries to non-indexed representations.
    """

    @classmethod
    def get_document_capability(
        cls,
        session: Session,
        document_id: Optional[uuid.UUID] = None,
    ) -> DocumentCapabilityReport:
        if not document_id:
            return cls._get_global_capability(session)

        doc = session.get(Document, document_id)
        if not doc:
            return DocumentCapabilityReport(
                document_id=document_id,
                filename="Non-existent Document",
                recommended_actions=["Upload or select a valid document."],
            )

        # 1. Text Chunks
        chunk_count = session.scalar(
            select(func.count(DocumentChunk.id)).where(DocumentChunk.document_id == document_id)
        ) or 0

        # 2. Vector Embeddings
        embedding_count = session.scalar(
            select(func.count(ChunkEmbedding.id))
            .join(DocumentChunk, ChunkEmbedding.chunk_id == DocumentChunk.id)
            .where(DocumentChunk.document_id == document_id)
        ) or 0

        # 3. Multimodal Tables
        table_count = session.scalar(
            select(func.count(DocumentElement.id)).where(
                DocumentElement.document_id == document_id,
                DocumentElement.element_type == "table",
            )
        ) or 0

        # 4. Page Assets & Images
        page_count = session.scalar(
            select(func.count(DocumentPageAsset.id)).where(DocumentPageAsset.document_id == document_id)
        ) or 0

        # 5. CLIP Embeddings
        clip_count = session.scalar(
            select(func.count(MultimodalEmbedding.id)).where(MultimodalEmbedding.document_id == document_id)
        ) or 0

        # 6. Graph Evidence
        graph_count = session.scalar(
            select(func.count(GraphEvidence.id)).where(GraphEvidence.document_id == document_id)
        ) or 0

        has_chunks = chunk_count > 0
        has_embs = embedding_count > 0
        has_tbls = table_count > 0
        has_pages = page_count > 0
        has_clip = clip_count > 0
        has_graph = graph_count > 0

        supported_routes: List[str] = []
        if has_chunks and has_embs:
            supported_routes.extend(["text_dense", "text_hybrid", "agentic"])
        elif has_chunks:
            supported_routes.append("text_hybrid")  # lexical portion available

        if has_tbls:
            supported_routes.append("multimodal_table")

        if has_pages or has_clip:
            supported_routes.append("multimodal_visual")

        if has_graph:
            supported_routes.append("graph")

        recommended_actions: List[str] = []
        if not has_chunks:
            recommended_actions.append("Execute document chunking (Phase 03) to enable dense text search.")
        if has_chunks and not has_embs:
            recommended_actions.append("Generate vector embeddings (Phase 04) to enable vector search.")
        if not has_tbls and doc.mime_type == "application/pdf":
            recommended_actions.append("Run Table Extraction in Multimodal Lab (Phase 12) if document has tables.")
        if not has_graph:
            recommended_actions.append("Construct Knowledge Graph in Graph RAG Lab (Phase 11) for entity analysis.")

        return DocumentCapabilityReport(
            document_id=document_id,
            filename=doc.original_filename,
            mime_type=doc.mime_type,
            has_text_chunks=has_chunks,
            has_embeddings=has_embs,
            has_tables=has_tbls,
            has_page_images=has_pages,
            has_clip_embeddings=has_clip,
            has_knowledge_graph=has_graph,
            chunk_count=chunk_count,
            embedding_count=embedding_count,
            table_count=table_count,
            page_count=page_count,
            graph_evidence_count=graph_count,
            supported_routes=supported_routes,
            recommended_actions=recommended_actions,
        )

    @classmethod
    def _get_global_capability(cls, session: Session) -> DocumentCapabilityReport:
        chunk_count = session.scalar(select(func.count(DocumentChunk.id))) or 0
        embedding_count = session.scalar(select(func.count(ChunkEmbedding.id))) or 0
        table_count = session.scalar(
            select(func.count(DocumentElement.id)).where(DocumentElement.element_type == "table")
        ) or 0
        page_count = session.scalar(select(func.count(DocumentPageAsset.id))) or 0
        clip_count = session.scalar(select(func.count(MultimodalEmbedding.id))) or 0
        graph_count = session.scalar(select(func.count(GraphEvidence.id))) or 0

        supported_routes: List[str] = []
        if chunk_count > 0 and embedding_count > 0:
            supported_routes.extend(["text_dense", "text_hybrid", "agentic"])
        if table_count > 0:
            supported_routes.append("multimodal_table")
        if page_count > 0 or clip_count > 0:
            supported_routes.append("multimodal_visual")
        if graph_count > 0:
            supported_routes.append("graph")

        return DocumentCapabilityReport(
            document_id=None,
            filename="Global Repository",
            mime_type="all",
            has_text_chunks=chunk_count > 0,
            has_embeddings=embedding_count > 0,
            has_tables=table_count > 0,
            has_page_images=page_count > 0,
            has_clip_embeddings=clip_count > 0,
            has_knowledge_graph=graph_count > 0,
            chunk_count=chunk_count,
            embedding_count=embedding_count,
            table_count=table_count,
            page_count=page_count,
            graph_evidence_count=graph_count,
            supported_routes=supported_routes,
            recommended_actions=[],
        )


document_capability_service = DocumentCapabilityService()
