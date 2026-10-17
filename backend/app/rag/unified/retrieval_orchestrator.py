import time
import uuid
from typing import List, Dict, Any, Optional, Tuple
from sqlalchemy.orm import Session

from app.models.document import Document
from app.rag.retrieval.service import retrieval_service
from app.rag.retrieval.schemas import SearchRequest, RetrievalFilter
from app.rag.multimodal.retrieval.text_retriever import OCRTextRetriever
from app.rag.multimodal.service import MultimodalService
from app.rag.graph.retrieval.graph_retriever import graph_retriever
from app.rag.unified.schemas import UnifiedEvidenceItem


class UnifiedRetrievalOrchestrator:
    """
    Orchestrates multi-route retrieval across specialized engines, maps heterogeneous outputs
    into the Unified Evidence Model, and fuses candidate items via Reciprocal Rank Fusion (RRF).
    """

    def __init__(self):
        self.multimodal_service = MultimodalService()

    def retrieve(
        self,
        session: Session,
        query: str,
        routes: List[str],
        document_id: Optional[uuid.UUID] = None,
        page_number: Optional[int] = None,
        top_k: int = 5,
    ) -> Tuple[List[UnifiedEvidenceItem], Dict[str, float]]:
        per_route_latency: Dict[str, float] = {}
        route_results: Dict[str, List[UnifiedEvidenceItem]] = {}

        # Resolve document name if filter provided
        doc_name = "Global Corpus"
        if document_id:
            d = session.get(Document, document_id)
            if d:
                doc_name = d.original_filename

        for route in routes:
            t0 = time.perf_counter()
            items: List[UnifiedEvidenceItem] = []

            try:
                if route == "text_dense":
                    items = self._retrieve_text_dense(
                        session=session,
                        query=query,
                        document_id=document_id,
                        doc_name=doc_name,
                        page_number=page_number,
                        top_k=top_k * 2,
                    )
                elif route == "text_hybrid":
                    items = self._retrieve_text_hybrid(
                        session=session,
                        query=query,
                        document_id=document_id,
                        doc_name=doc_name,
                        page_number=page_number,
                        top_k=top_k * 2,
                    )
                elif route == "multimodal_table":
                    items = self._retrieve_multimodal_table(
                        session=session,
                        query=query,
                        document_id=document_id,
                        doc_name=doc_name,
                        page_number=page_number,
                        top_k=top_k * 2,
                    )
                elif route == "multimodal_visual":
                    items = self._retrieve_multimodal_visual(
                        session=session,
                        query=query,
                        document_id=document_id,
                        doc_name=doc_name,
                        page_number=page_number,
                        top_k=top_k * 2,
                    )
                elif route == "graph":
                    items = self._retrieve_graph(
                        session=session,
                        query=query,
                        document_id=document_id,
                        doc_name=doc_name,
                        top_k=top_k * 2,
                    )
                elif route == "agentic":
                    # Reuse text_hybrid with expanded search depth
                    items = self._retrieve_text_hybrid(
                        session=session,
                        query=query,
                        document_id=document_id,
                        doc_name=doc_name,
                        page_number=page_number,
                        top_k=top_k * 2,
                    )
                    for it in items:
                        it.engine = "agentic"
            except Exception as e:
                # Log error and preserve other routes
                items = []

            elapsed = round((time.perf_counter() - t0) * 1000.0, 2)
            per_route_latency[route] = elapsed
            route_results[route] = items

        # Fusion: If multiple routes, fuse via Reciprocal Rank Fusion (k=60)
        fused_items = self._fuse_results(route_results, top_k=top_k)

        # Assign unique stable citation identifiers: [S1], [S2]...
        for idx, item in enumerate(fused_items, start=1):
            item.citation_id = f"[S{idx}]"

        return fused_items, per_route_latency

    def _retrieve_text_dense(
        self,
        session: Session,
        query: str,
        document_id: Optional[uuid.UUID],
        doc_name: str,
        page_number: Optional[int],
        top_k: int,
    ) -> List[UnifiedEvidenceItem]:
        flt = RetrievalFilter(document_id=document_id, page_number=page_number) if (document_id or page_number) else None
        req = SearchRequest(
            query=query,
            strategy="dense_only",
            top_k=top_k,
            filters=flt,
            explain_plan=False,
        )
        resp = retrieval_service.search(session, req)
        items = []
        for r in resp.results:
            items.append(
                UnifiedEvidenceItem(
                    document_id=r.document_id,
                    document_name=r.document_name or doc_name,
                    evidence_id=str(r.chunk_id),
                    evidence_type="text",
                    page_number=r.page_number,
                    section_id=str(r.sequence_index),
                    engine="text_dense",
                    raw_score=r.similarity_score,
                    score_semantics="cosine_similarity",
                    calibrated_score=round(max(0.0, min(1.0, r.similarity_score)), 4),
                    content=r.content,
                    structured_data=None,
                    image_url=None,
                    citation_id="[TMP]",
                    origin_tag=f"[CHUNK_{r.sequence_index}]",
                    provenance={"chunk_id": str(r.chunk_id), "strategy": "dense_only"},
                )
            )
        return items

    def _retrieve_text_hybrid(
        self,
        session: Session,
        query: str,
        document_id: Optional[uuid.UUID],
        doc_name: str,
        page_number: Optional[int],
        top_k: int,
    ) -> List[UnifiedEvidenceItem]:
        flt = RetrievalFilter(document_id=document_id, page_number=page_number) if (document_id or page_number) else None
        req = SearchRequest(
            query=query,
            strategy="hybrid_rrf",
            top_k=top_k,
            filters=flt,
            rrf_k=60,
            explain_plan=False,
        )
        resp = retrieval_service.search(session, req)
        items = []
        for r in resp.results:
            items.append(
                UnifiedEvidenceItem(
                    document_id=r.document_id,
                    document_name=r.document_name or doc_name,
                    evidence_id=str(r.chunk_id),
                    evidence_type="text",
                    page_number=r.page_number,
                    section_id=str(r.sequence_index),
                    engine="text_hybrid",
                    raw_score=r.similarity_score,
                    score_semantics="rrf_score",
                    calibrated_score=round(max(0.0, min(1.0, r.similarity_score)), 4),
                    content=r.content,
                    structured_data=None,
                    image_url=None,
                    citation_id="[TMP]",
                    origin_tag=f"[CHUNK_{r.sequence_index}]",
                    provenance={"chunk_id": str(r.chunk_id), "strategy": "hybrid_rrf"},
                )
            )
        return items

    def _retrieve_multimodal_table(
        self,
        session: Session,
        query: str,
        document_id: Optional[uuid.UUID],
        doc_name: str,
        page_number: Optional[int],
        top_k: int,
    ) -> List[UnifiedEvidenceItem]:
        results = OCRTextRetriever.search(
            db=session,
            query=query,
            document_id=document_id,
            page_number=page_number,
            top_k=top_k,
        )
        items = []
        for r in results:
            struct = r.get("structured_data") or {}
            content_repr = struct.get("markdown") or struct.get("markdown_table") or r.get("content_text", "")

            item = UnifiedEvidenceItem(
                document_id=uuid.UUID(r["document_id"]) if r.get("document_id") else document_id,
                document_name=doc_name,
                evidence_id=r.get("id", str(uuid.uuid4())),
                evidence_type=r.get("element_type", "table"),
                page_number=r.get("page_number"),
                section_id=None,
                engine="multimodal_table",
                raw_score=r.get("score", 0.0),
                score_semantics="keyword_table_boost",
                calibrated_score=round(max(0.0, min(1.0, r.get("score", 0.0))), 4),
                content=content_repr,
                structured_data=struct,
                image_url=r.get("image_path"),
                citation_id="[TMP]",
                origin_tag=r.get("citation_tag", f"[TBL_P{r.get('page_number', 1)}]"),
                provenance={"element_id": r.get("id"), "element_type": r.get("element_type")},
            )
            item._local_image_path = r.get("_local_image_path")
            items.append(item)
        return items

    def _retrieve_multimodal_visual(
        self,
        session: Session,
        query: str,
        document_id: Optional[uuid.UUID],
        doc_name: str,
        page_number: Optional[int],
        top_k: int,
    ) -> List[UnifiedEvidenceItem]:
        results = self.multimodal_service.cross_modal_retriever.search_text_to_image(
            db=session,
            query=query,
            document_id=document_id,
            page_number=page_number,
            top_k=top_k,
        )
        items = []
        for r in results:
            item = UnifiedEvidenceItem(
                document_id=uuid.UUID(r["document_id"]) if r.get("document_id") else document_id,
                document_name=doc_name,
                evidence_id=r.get("id", str(uuid.uuid4())),
                evidence_type=r.get("element_type", "image"),
                page_number=r.get("page_number"),
                section_id=None,
                engine="multimodal_visual",
                raw_score=r.get("score", 0.0),
                score_semantics="clip_similarity",
                calibrated_score=round(max(0.0, min(1.0, r.get("score", 0.0))), 4),
                content=r.get("content_text") or f"[Visual figure asset on page {r.get('page_number')}]",
                structured_data=r.get("structured_data"),
                image_url=r.get("image_path"),
                citation_id="[TMP]",
                origin_tag=r.get("citation_tag", f"[IMG_P{r.get('page_number', 1)}]"),
                provenance={"model": "clip-ViT-B-32", "strategy": "cross_modal"},
            )
            item._local_image_path = r.get("_local_image_path")
            items.append(item)
        return items

    def _retrieve_graph(
        self,
        session: Session,
        query: str,
        document_id: Optional[uuid.UUID],
        doc_name: str,
        top_k: int,
    ) -> List[UnifiedEvidenceItem]:
        ret_data = graph_retriever.retrieve(
            session=session,
            query=query,
            max_hops=2,
            top_k=top_k,
        )
        evidence = ret_data.get("evidence", [])
        relations = ret_data.get("relations", [])
        items = []

        # Convert graph evidence snippets
        for idx, ev in enumerate(evidence):
            if document_id and ev.get("document_id") and str(ev["document_id"]) != str(document_id):
                continue
            items.append(
                UnifiedEvidenceItem(
                    document_id=uuid.UUID(str(ev["document_id"])) if ev.get("document_id") else document_id,
                    document_name=doc_name,
                    evidence_id=str(ev.get("id") or uuid.uuid4()),
                    evidence_type="graph",
                    page_number=ev.get("page_number"),
                    section_id=None,
                    engine="graph",
                    raw_score=0.85 - (idx * 0.05),
                    score_semantics="graph_evidence_rank",
                    calibrated_score=round(max(0.1, 0.85 - (idx * 0.05)), 4),
                    content=ev.get("snippet", ""),
                    structured_data={"entity_id": ev.get("entity_id"), "relation_id": ev.get("relation_id")},
                    image_url=None,
                    citation_id="[TMP]",
                    origin_tag=f"[GRAPH_EVID_{idx+1}]",
                    provenance={"graph_hops": 2},
                )
            )

        # If evidence was sparse, summarize relation triples
        if not items and relations:
            triple_texts = [f"({r.get('source_entity_name')} --[{r.get('relation_type')}]--> {r.get('target_entity_name')})" for r in relations[:5]]
            items.append(
                UnifiedEvidenceItem(
                    document_id=document_id,
                    document_name=doc_name,
                    evidence_id=str(uuid.uuid4()),
                    evidence_type="graph",
                    page_number=None,
                    section_id=None,
                    engine="graph",
                    raw_score=0.80,
                    score_semantics="graph_relation_triple",
                    calibrated_score=0.80,
                    content="Knowledge Graph Relations:\n" + "\n".join(triple_texts),
                    structured_data={"relation_count": len(relations)},
                    image_url=None,
                    citation_id="[TMP]",
                    origin_tag="[GRAPH_TRIPLES]",
                    provenance={"relations_extracted": len(relations)},
                )
            )

        return items

    def _fuse_results(
        self,
        route_results: Dict[str, List[UnifiedEvidenceItem]],
        top_k: int,
    ) -> List[UnifiedEvidenceItem]:
        """
        Reciprocal Rank Fusion (RRF) across heterogeneous retrieval engines.
        Formula: Score(d) = sum( 1 / (60 + rank_i) ) for each engine ranking.
        """
        rrf_scores: Dict[str, float] = {}
        items_by_key: Dict[str, UnifiedEvidenceItem] = {}

        # Engine weight multipliers (e.g. prioritize structured tables if retrieved)
        engine_weights = {
            "multimodal_table": 1.25,
            "multimodal_visual": 1.10,
            "graph": 1.15,
            "text_hybrid": 1.0,
            "text_dense": 0.95,
            "agentic": 1.05,
        }

        for engine, items in route_results.items():
            w = engine_weights.get(engine, 1.0)
            for rank, item in enumerate(items, start=1):
                # Unique key: chunk_id or element_id or content hash
                key = item.evidence_id or f"{item.document_id}_{item.content[:40]}"
                if key not in items_by_key:
                    items_by_key[key] = item

                rrf_add = w * (1.0 / (60.0 + rank))
                rrf_scores[key] = rrf_scores.get(key, 0.0) + rrf_add

        # Sort items by accumulated RRF score
        sorted_keys = sorted(rrf_scores.keys(), key=lambda k: rrf_scores[k], reverse=True)

        final_items: List[UnifiedEvidenceItem] = []
        for k in sorted_keys[:top_k]:
            it = items_by_key[k]
            # Normalize RRF score to [0.0, 1.0] for calibrated presentation
            it.calibrated_score = round(min(1.0, rrf_scores[k] * 25.0), 4)
            final_items.append(it)

        return final_items


unified_retrieval_orchestrator = UnifiedRetrievalOrchestrator()
