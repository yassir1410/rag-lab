import uuid
from typing import List, Optional, Tuple, Dict, Any
from sqlalchemy import select, func, desc
from sqlalchemy.orm import Session

from app.core.logging import logger
from app.models.document import Document, DocumentSection
from app.models.chunking import ChunkingExperiment, DocumentChunk
from app.rag.chunking.base import ChunkingResult, Chunk
from app.rag.chunking.registry import ChunkingStrategyRegistry, default_chunking_registry


class ChunkingService:
    def __init__(self, registry: Optional[ChunkingStrategyRegistry] = None):
        self.registry = registry or default_chunking_registry

    def _get_document_context(self, db: Session, document_id: uuid.UUID) -> Tuple[str, Dict[str, Any]]:
        """Fetch document full text and structural sections context."""
        doc = db.get(Document, document_id)
        if not doc:
            raise ValueError(f"Document with id {document_id} not found")

        sections = (
            db.execute(
                select(DocumentSection)
                .where(DocumentSection.document_id == document_id)
                .order_by(DocumentSection.section_index)
            )
            .scalars()
            .all()
        )

        # Full text: prefer normalized_content, fallback to raw content
        text_parts = [s.normalized_content or s.content for s in sections]
        full_text = "\n\n".join(text_parts) if text_parts else ""

        section_dicts = [
            {
                "id": str(s.id),
                "section_index": s.section_index,
                "content": s.normalized_content or s.content,
                "page_number": s.page_number,
                "heading": s.heading,
                "content_type": s.content_type,
                "metadata": s.metadata_json,
            }
            for s in sections
        ]

        context = {
            "document_id": str(doc.id),
            "original_filename": doc.original_filename,
            "page_count": doc.page_count,
            "sections": section_dicts,
        }

        return full_text, context

    def preview_chunking(
        self,
        db: Session,
        document_id: uuid.UUID,
        strategy_name: str,
        config: Dict[str, Any],
    ) -> ChunkingResult:
        """Run chunking strategy in-memory without database persistence."""
        full_text, context = self._get_document_context(db, document_id)
        strategy = self.registry.get(strategy_name)
        return strategy.split(
            text=full_text,
            document_id=document_id,
            config_dict=config,
            document_context=context,
        )

    def create_and_persist_experiment(
        self,
        db: Session,
        document_id: uuid.UUID,
        strategy_name: str,
        config: Dict[str, Any],
    ) -> ChunkingExperiment:
        """Run chunking strategy and persist the experiment and all generated chunks."""
        result = self.preview_chunking(db, document_id, strategy_name, config)

        exp_id = uuid.uuid4()
        experiment = ChunkingExperiment(
            id=exp_id,
            document_id=document_id,
            strategy=strategy_name,
            config=result.config,
            metrics=result.metrics,
        )
        db.add(experiment)

        # Map parent-child temporary IDs to newly created DB UUIDs if applicable
        id_map = {}
        for chunk in result.chunks:
            id_map[str(chunk.id)] = chunk.id

        for chunk in result.chunks:
            parent_id = None
            if chunk.metadata.parent_chunk_id:
                try:
                    parent_id = uuid.UUID(chunk.metadata.parent_chunk_id)
                except (ValueError, TypeError):
                    parent_id = None

            db_chunk = DocumentChunk(
                id=chunk.id,
                experiment_id=exp_id,
                document_id=document_id,
                sequence_index=chunk.sequence_index,
                content=chunk.content,
                token_count=chunk.token_count,
                char_count=chunk.char_count,
                parent_chunk_id=parent_id,
                metadata_json={
                    "start_char": chunk.metadata.start_char,
                    "end_char": chunk.metadata.end_char,
                    "page_numbers": chunk.metadata.page_numbers,
                    "headings": chunk.metadata.headings,
                    "source_section_ids": chunk.metadata.source_section_ids,
                    "content_type": chunk.metadata.content_type,
                    "child_chunk_ids": chunk.metadata.child_chunk_ids,
                    "breakpoint_distance": chunk.metadata.breakpoint_distance,
                    "custom": chunk.metadata.custom,
                },
            )
            db.add(db_chunk)

        db.commit()
        db.refresh(experiment)
        logger.info(
            f"Created chunking experiment {exp_id} for doc={document_id} using '{strategy_name}' ({len(result.chunks)} chunks)"
        )
        return experiment

    def list_experiments(
        self,
        db: Session,
        document_id: Optional[uuid.UUID] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Tuple[int, List[ChunkingExperiment]]:
        """List experiments with optional document filter."""
        query = select(ChunkingExperiment)
        count_query = select(func.count(ChunkingExperiment.id))

        if document_id:
            query = query.where(ChunkingExperiment.document_id == document_id)
            count_query = count_query.where(ChunkingExperiment.document_id == document_id)

        total = db.scalar(count_query) or 0
        experiments = (
            db.execute(
                query.order_by(desc(ChunkingExperiment.created_at))
                .limit(limit)
                .offset(offset)
            )
            .scalars()
            .all()
        )
        return total, list(experiments)

    def get_experiment(
        self, db: Session, experiment_id: uuid.UUID
    ) -> Optional[ChunkingExperiment]:
        """Fetch a specific experiment with its chunks."""
        return db.get(ChunkingExperiment, experiment_id)

    def delete_experiment(self, db: Session, experiment_id: uuid.UUID) -> bool:
        """Delete an experiment and its chunks without touching source documents."""
        exp = db.get(ChunkingExperiment, experiment_id)
        if not exp:
            return False
        db.delete(exp)
        db.commit()
        logger.info(f"Deleted chunking experiment {experiment_id}")
        return True

    def compare_strategies(
        self,
        db: Session,
        document_id: uuid.UUID,
        strategies_config: List[Dict[str, Any]],
    ) -> List[ChunkingResult]:
        """Execute multiple strategy configurations on the same document simultaneously."""
        results: List[ChunkingResult] = []
        for item in strategies_config:
            s_name = item.get("strategy")
            cfg = item.get("config", {})
            res = self.preview_chunking(db, document_id, s_name, cfg)
            results.append(res)
        return results
