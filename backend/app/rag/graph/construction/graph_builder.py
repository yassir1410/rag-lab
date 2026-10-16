import time
import uuid
from typing import Dict, Any, Optional, List
from sqlalchemy.orm import Session

from app.models.document import Document
from app.models.chunking import DocumentChunk
from app.models.graph import GraphExtractionRun
from app.rag.graph.repository import BaseGraphRepository, postgres_graph_repository
from app.rag.graph.extraction.entity_extractor import rule_entity_extractor
from app.rag.graph.extraction.relation_extractor import rule_relation_extractor
from app.rag.graph.extraction.structured_llm_extractor import StructuredLLMExtractor
from app.rag.graph.resolution.entity_normalizer import entity_normalizer
from app.rag.graph.construction.provenance import ProvenanceTracker


class KnowledgeGraphBuilder:
    """
    Coordinates end-to-end Knowledge Graph construction from ingested documents.
    Executes extraction, normalizes entities, builds relationship edges,
    and attaches documentary evidence.
    """

    def __init__(self, repository: Optional[BaseGraphRepository] = None):
        self.repository = repository or postgres_graph_repository
        self.provenance = ProvenanceTracker(self.repository)

    def build_from_document(
        self,
        session: Session,
        document_id: uuid.UUID,
        strategy: str = "rule",
        provider_id: str = "mock",
        model_id: str = "mock-instruct-v1",
        extraction_config: Optional[Dict[str, Any]] = None,
    ) -> GraphExtractionRun:
        t0 = time.perf_counter()

        doc = session.query(Document).filter(Document.id == document_id).first()
        if not doc:
            raise ValueError(f"Document with ID {document_id} not found.")

        # 1. Create extraction run record
        run_record = GraphExtractionRun(
            document_id=document_id,
            model_id=f"{provider_id}:{model_id}" if strategy == "llm" else "rule_based",
            extraction_config=extraction_config or {"strategy": strategy},
            status="running",
            metrics={},
        )
        session.add(run_record)
        session.flush()

        # 2. Fetch chunks
        chunks = (
            session.query(DocumentChunk)
            .filter(DocumentChunk.document_id == document_id)
            .order_by(DocumentChunk.sequence_index)
            .all()
        )

        llm_extractor = StructuredLLMExtractor(provider_id=provider_id, model_id=model_id) if strategy in ("llm", "hybrid") else None

        total_entities_extracted = 0
        total_relations_extracted = 0
        name_to_entity_id: Dict[str, uuid.UUID] = {}

        # 3. Process each chunk
        for chunk in chunks:
            raw_text = chunk.content

            # Extract entities
            chunk_entities = []
            if strategy in ("rule", "hybrid") or not llm_extractor:
                rule_ents = rule_entity_extractor.extract_from_chunk(
                    raw_text, chunk_id=chunk.id, page_reference=chunk.sequence_index
                )
                chunk_entities.extend(rule_ents)

            if strategy in ("llm", "hybrid") and llm_extractor:
                llm_res = llm_extractor.extract(
                    raw_text, chunk_id=chunk.id, page_reference=chunk.sequence_index
                )
                chunk_entities.extend(llm_res.entities)

            # Persist and resolve entities
            for ent_candidate in chunk_entities:
                # Canonicalize name
                ent_candidate.canonical_name = entity_normalizer.canonicalize(ent_candidate.canonical_name)
                entity = self.repository.create_or_get_entity(
                    session=session,
                    candidate=ent_candidate,
                    extraction_method=strategy,
                )
                name_to_entity_id[entity.canonical_name.lower().strip()] = entity.id
                for alias in (entity.aliases or []):
                    name_to_entity_id[alias.lower().strip()] = entity.id

                # Attach provenance
                self.provenance.attach_entity_evidence(
                    session=session,
                    entity_id=entity.id,
                    document_id=document_id,
                    evidence_text=ent_candidate.evidence_text or raw_text[:120],
                    chunk_id=chunk.id,
                    page_reference=chunk.sequence_index,
                    run_id=run_record.id,
                )
                total_entities_extracted += 1

            # Extract relations
            chunk_relations = []
            if strategy in ("rule", "hybrid") or not llm_extractor:
                rule_rels = rule_relation_extractor.extract_relations(
                    raw_text, chunk_entities, chunk_id=chunk.id, page_reference=chunk.sequence_index
                )
                chunk_relations.extend(rule_rels)

            if strategy in ("llm", "hybrid") and llm_extractor:
                chunk_relations.extend(llm_res.relations)

            # Persist relations
            for rel_candidate in chunk_relations:
                src_key = rel_candidate.source_name.lower().strip()
                tgt_key = rel_candidate.target_name.lower().strip()

                src_id = name_to_entity_id.get(src_key)
                tgt_id = name_to_entity_id.get(tgt_key)

                if src_id and tgt_id and src_id != tgt_id:
                    relation = self.repository.create_or_get_relation(
                        session=session,
                        source_id=src_id,
                        target_id=tgt_id,
                        relation_type=rel_candidate.relation_type,
                        properties=rel_candidate.properties or {},
                        extraction_method=strategy,
                        confidence=rel_candidate.confidence,
                    )

                    # Attach provenance
                    self.provenance.attach_relation_evidence(
                        session=session,
                        relation_id=relation.id,
                        document_id=document_id,
                        evidence_text=rel_candidate.evidence_text or raw_text[:120],
                        chunk_id=chunk.id,
                        page_reference=chunk.sequence_index,
                        run_id=run_record.id,
                    )
                    total_relations_extracted += 1

        session.commit()

        # Update extraction run metrics
        lat = (time.perf_counter() - t0) * 1000.0
        run_record.status = "completed"
        run_record.metrics = {
            "chunks_processed": len(chunks),
            "entities_extracted": total_entities_extracted,
            "unique_entities": len(set(name_to_entity_id.values())),
            "relations_extracted": total_relations_extracted,
            "latency_ms": round(lat, 2),
            "strategy": strategy,
        }
        session.commit()
        return run_record


knowledge_graph_builder = KnowledgeGraphBuilder()
