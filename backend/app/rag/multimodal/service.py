import os
import time
import uuid
from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import select

from app.models.document import Document
from app.models.multimodal import (
    DocumentPageAsset,
    DocumentElement,
    MultimodalEmbedding,
)
from app.rag.multimodal.ingestion.image_loader import SafeImageLoader
from app.rag.multimodal.ingestion.scanned_pdf import ScannedPDFProcessor
from app.rag.multimodal.ingestion.layout_analyzer import DocumentLayoutAnalyzer
from app.rag.multimodal.ingestion.table_extractor import TableExtractor
from app.rag.multimodal.ocr.tesseract_provider import (
    TesseractOCRProvider,
    EducationalOCRComparator,
)
from app.rag.multimodal.embeddings.image_embedding_provider import (
    MultimodalEmbeddingProvider,
)
from app.rag.multimodal.embeddings.multimodal_registry import (
    MultimodalModelRegistry,
)
from app.rag.multimodal.retrieval.text_retriever import OCRTextRetriever
from app.rag.multimodal.retrieval.cross_modal_retriever import CrossModalRetriever
from app.rag.multimodal.retrieval.image_retriever import ImageToImageRetriever
from app.rag.multimodal.retrieval.page_retriever import PageLevelRetriever
from app.rag.multimodal.retrieval.fusion import HybridMultimodalFusion
from app.rag.multimodal.generation.vision_provider import (
    MockVisionProvider,
    GeminiVisionProvider,
    VisionProviderError,
)
from app.rag.multimodal.generation.multimodal_context import MultimodalContextBuilder
from app.rag.multimodal.generation.citation_manager import MultimodalCitationManager
from app.rag.multimodal.evaluation.benchmark import MultimodalBenchmarkRunner


class MultimodalService:
    """End-to-end service orchestrating multimodal ingestion, OCR, layout analysis, retrieval, and vision Q&A."""

    def __init__(self):
        self.ocr_provider = TesseractOCRProvider()
        self.embedding_provider = MultimodalEmbeddingProvider()
        self.cross_modal_retriever = CrossModalRetriever(self.embedding_provider)
        self.image_retriever = ImageToImageRetriever(self.embedding_provider)
        self.mock_vision = MockVisionProvider()
        self.gemini_vision = GeminiVisionProvider()

    def _resolve_file_path(self, doc: Document) -> str:
        from app.core.config import settings
        if os.path.isabs(doc.storage_path) and os.path.exists(doc.storage_path):
            return doc.storage_path
        candidate = os.path.join(str(settings.ABSOLUTE_STORAGE_DIR), doc.storage_path)
        if os.path.exists(candidate):
            return candidate
        candidate2 = os.path.join("/app/data/uploads", doc.storage_path)
        if os.path.exists(candidate2):
            return candidate2
        return candidate

    def analyze_document(
        self,
        db: Session,
        document_id: uuid.UUID,
        dpi: int = 150,
        force_ocr: bool = False,
        ocr_language: str = "eng",
    ) -> Dict[str, Any]:
        """Perform document layout analysis, page rendering, OCR, and element extraction."""
        t0 = time.perf_counter()
        doc = db.get(Document, document_id)
        if not doc:
            raise ValueError(f"Document with ID {document_id} not found.")

        # Delete existing assets/elements for clean re-analysis
        db.query(DocumentElement).filter(DocumentElement.document_id == document_id).delete()
        db.query(DocumentPageAsset).filter(DocumentPageAsset.document_id == document_id).delete()
        db.query(MultimodalEmbedding).filter(MultimodalEmbedding.document_id == document_id).delete()
        db.commit()

        file_path = self._resolve_file_path(doc)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Underlying file not found at {file_path}")

        # 1. Process pages
        pages_info = ScannedPDFProcessor.process_pdf(file_path, dpi=dpi)
        scanned_count = sum(1 for p in pages_info if p["is_scanned"])

        page_asset_records: Dict[int, DocumentPageAsset] = {}
        for p in pages_info:
            p_num = p["page_number"]
            # Save page image
            img_path = SafeImageLoader.save_page_image(
                image_bytes=p["image_bytes"],
                document_id=document_id,
                page_number=p_num,
            )

            # Perform OCR if page is scanned or forced
            ocr_text = None
            ocr_conf = None
            if (p["is_scanned"] or force_ocr) and self.ocr_provider.is_available():
                ocr_res = self.ocr_provider.perform_ocr(p["image_bytes"], language=ocr_language)
                ocr_text = ocr_res.get("text")
                ocr_conf = ocr_res.get("confidence")
            elif not p["is_scanned"]:
                ocr_text = p["native_text"]
                ocr_conf = 98.0

            asset = DocumentPageAsset(
                document_id=document_id,
                page_number=p_num,
                width=p["width"],
                height=p["height"],
                has_native_text=p["has_native_text"],
                native_text_char_count=p["native_text_char_count"],
                is_scanned=p["is_scanned"],
                image_path=img_path,
                ocr_text=ocr_text,
                ocr_confidence=ocr_conf,
                ocr_language=ocr_language,
            )
            db.add(asset)
            db.flush()
            page_asset_records[p_num] = asset

            # Create page-level multimodal embedding
            try:
                page_vec = self.embedding_provider.encode_image(img_path)
                mm_emb = MultimodalEmbedding(
                    document_id=document_id,
                    page_asset_id=asset.id,
                    model_id="clip-ViT-B-32",
                    embedding_type="page",
                    vector_dimension=512,
                    vector=page_vec,
                )
                db.add(mm_emb)
            except Exception:
                pass

        # 2. Extract layout elements & tables
        import pymupdf
        pdf_doc = pymupdf.open(file_path)
        total_elements = 0
        elem_counts: Dict[str, int] = {}

        try:
            for p_num, asset in page_asset_records.items():
                pdf_page = pdf_doc[p_num - 1]
                # Scale from points to pixel dimensions of rendered image
                scale_x = asset.width / max(1.0, pdf_page.rect.width)
                scale_y = asset.height / max(1.0, pdf_page.rect.height)

                elements = DocumentLayoutAnalyzer.analyze_page_layout(
                    pdf_page,
                    page_number=p_num,
                    scale_x=scale_x,
                    scale_y=scale_y,
                )

                for elem in elements:
                    elem_type = elem["element_type"]
                    elem_counts[elem_type] = elem_counts.get(elem_type, 0) + 1
                    total_elements += 1

                    # Crop figure or table image if applicable
                    cropped_path = None
                    if elem_type in ("figure", "table"):
                        elem_id_preview = uuid.uuid4()
                        cropped_path = SafeImageLoader.crop_and_save_element(
                            page_image_path=asset.image_path,
                            bbox=elem["bounding_box"],
                            document_id=document_id,
                            element_id=elem_id_preview,
                        )

                    doc_elem = DocumentElement(
                        document_id=document_id,
                        page_asset_id=asset.id,
                        page_number=p_num,
                        element_type=elem_type,
                        reading_order=elem["reading_order"],
                        bounding_box=elem["bounding_box"],
                        content_text=elem["content_text"],
                        structured_data=elem["structured_data"],
                        asset_path=cropped_path,
                        confidence=elem["confidence"],
                    )
                    db.add(doc_elem)
                    db.flush()

                    # Embed visual figures into CLIP space
                    if cropped_path and os.path.exists(cropped_path):
                        try:
                            f_vec = self.embedding_provider.encode_image(cropped_path)
                            f_emb = MultimodalEmbedding(
                                document_id=document_id,
                                element_id=doc_elem.id,
                                model_id="clip-ViT-B-32",
                                embedding_type="image",
                                vector_dimension=512,
                                vector=f_vec,
                            )
                            db.add(f_emb)
                        except Exception:
                            pass
        finally:
            pdf_doc.close()

        db.commit()
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        return {
            "document_id": str(document_id),
            "total_pages": len(pages_info),
            "scanned_pages_count": scanned_count,
            "total_elements": total_elements,
            "elements_by_type": elem_counts,
            "processing_time_ms": latency_ms,
            "message": f"Successfully analyzed {len(pages_info)} pages and extracted {total_elements} layout elements.",
        }

    def get_document_elements(
        self,
        db: Session,
        document_id: uuid.UUID,
        element_type: Optional[str] = None,
        page_number: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        stmt = select(DocumentElement).where(DocumentElement.document_id == document_id)
        if element_type:
            stmt = stmt.where(DocumentElement.element_type == element_type)
        if page_number:
            stmt = stmt.where(DocumentElement.page_number == page_number)
        stmt = stmt.order_by(DocumentElement.page_number, DocumentElement.reading_order)
        elements = db.scalars(stmt).all()

        return [
            {
                "id": str(e.id),
                "document_id": str(e.document_id),
                "page_number": e.page_number,
                "element_type": e.element_type,
                "reading_order": e.reading_order,
                "bounding_box": e.bounding_box,
                "content_text": e.content_text,
                "structured_data": e.structured_data,
                "asset_path": e.asset_path,
                "confidence": e.confidence,
            }
            for e in elements
        ]

    def get_document_pages(
        self,
        db: Session,
        document_id: uuid.UUID,
    ) -> List[Dict[str, Any]]:
        stmt = (
            select(DocumentPageAsset)
            .where(DocumentPageAsset.document_id == document_id)
            .order_by(DocumentPageAsset.page_number)
        )
        pages = db.scalars(stmt).all()

        return [
            {
                "id": str(p.id),
                "document_id": str(p.document_id),
                "page_number": p.page_number,
                "width": p.width,
                "height": p.height,
                "has_native_text": p.has_native_text,
                "native_text_char_count": p.native_text_char_count,
                "is_scanned": p.is_scanned,
                "image_path": p.image_path,
                "ocr_text": p.ocr_text,
                "ocr_confidence": p.ocr_confidence,
                "ocr_language": p.ocr_language,
            }
            for p in pages
        ]

    def get_page_image_path(
        self,
        db: Session,
        document_id: uuid.UUID,
        page_number: int,
    ) -> Optional[str]:
        stmt = select(DocumentPageAsset).where(
            DocumentPageAsset.document_id == document_id,
            DocumentPageAsset.page_number == page_number,
        )
        asset = db.scalars(stmt).first()
        if asset and os.path.exists(asset.image_path):
            return asset.image_path
        return None

    def preview_ocr(
        self,
        db: Session,
        document_id: uuid.UUID,
        page_number: int,
        language: str = "eng",
        run_engine: str = "tesseract",
    ) -> Dict[str, Any]:
        """Compares native PDF text vs OCR text, computing CER, WER, and word confidence."""
        t0 = time.perf_counter()
        doc = db.get(Document, document_id)
        if not doc:
            raise FileNotFoundError("Document not found.")
        file_path = self._resolve_file_path(doc)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Underlying file not found at {file_path}")

        import pymupdf
        pdf = pymupdf.open(file_path)
        try:
            if page_number < 1 or page_number > len(pdf):
                raise ValueError(f"Page number {page_number} out of range (1-{len(pdf)})")
            page = pdf[page_number - 1]
            native_text = page.get_text("text").strip()
            pixmap = page.get_pixmap(dpi=150)
            image_bytes = pixmap.tobytes("png")
        finally:
            pdf.close()

        ocr_res = self.ocr_provider.perform_ocr(image_bytes, language=language)
        ocr_text = ocr_res.get("text", "")
        is_avail = ocr_res.get("is_available", False)
        ocr_conf = ocr_res.get("confidence", 0.0)
        warning = ocr_res.get("warning")

        # If tesseract unavailable, provide simulated educational OCR comparison
        if not is_avail:
            ocr_text = native_text
            ocr_conf = 88.5
            warning = "Tesseract engine not detected. Demonstrating baseline with native text."

        cer = EducationalOCRComparator.compute_cer(native_text, ocr_text)
        wer = EducationalOCRComparator.compute_wer(native_text, ocr_text)
        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        return {
            "document_id": str(document_id),
            "page_number": page_number,
            "language": language,
            "is_available": is_avail,
            "native_text": native_text,
            "ocr_text": ocr_text,
            "cer": cer,
            "wer": wer,
            "ocr_confidence": ocr_conf,
            "words_detected": len(ocr_text.split()),
            "execution_time_ms": elapsed_ms,
            "warning": warning,
        }

    def extract_tables(
        self,
        db: Session,
        document_id: uuid.UUID,
        page_number: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        doc = db.get(Document, document_id)
        if not doc:
            raise FileNotFoundError("Document not found.")
        file_path = self._resolve_file_path(doc)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Underlying file not found at {file_path}")
        
        tables = TableExtractor.extract_tables_from_doc(file_path, target_page=page_number)

        # Persist extracted tables if not already stored as DocumentElement
        try:
            for t in tables:
                p_num = t["page_number"]
                existing = db.query(DocumentElement).filter(
                    DocumentElement.document_id == document_id,
                    DocumentElement.page_number == p_num,
                    DocumentElement.element_type == "table",
                ).first()
                if not existing:
                    page_asset = db.query(DocumentPageAsset).filter(
                        DocumentPageAsset.document_id == document_id,
                        DocumentPageAsset.page_number == p_num,
                    ).first()
                    doc_elem = DocumentElement(
                        document_id=document_id,
                        page_asset_id=page_asset.id if page_asset else None,
                        page_number=p_num,
                        element_type="table",
                        reading_order=100,
                        bounding_box=t["bounding_box"],
                        content_text=t["markdown_table"],
                        structured_data={
                            "table_id": t["table_id"],
                            "caption": t["caption"],
                            "num_rows": t["num_rows"],
                            "num_cols": t["num_cols"],
                            "headers": t["headers"],
                            "rows": t["rows"],
                            "markdown": t["markdown_table"],
                            "raw_flattened_text": t.get("raw_flattened_text", ""),
                        },
                        confidence=0.95,
                    )
                    db.add(doc_elem)
            db.commit()
        except Exception as e:
            db.rollback()
            logger.warning(f"Failed to persist extracted tables to db: {e}")

        return tables

    def search(
        self,
        db: Session,
        query: str,
        strategy: str = "hybrid",
        document_id: Optional[uuid.UUID] = None,
        page_number: Optional[int] = None,
        element_ids: Optional[List[uuid.UUID]] = None,
        top_k: int = 5,
        ocr_weight: float = 0.5,
        visual_weight: float = 0.5,
    ) -> Dict[str, Any]:
        """Execute multimodal retrieval across OCR text, text-to-image, page-level, or hybrid RRF."""
        t0 = time.perf_counter()

        if strategy == "ocr_text":
            results = OCRTextRetriever.search(
                db,
                query=query,
                document_id=document_id,
                page_number=page_number,
                element_ids=element_ids,
                top_k=top_k,
            )
        elif strategy == "text_to_image":
            results = self.cross_modal_retriever.search_text_to_image(
                db,
                query=query,
                document_id=document_id,
                page_number=page_number,
                element_ids=element_ids,
                top_k=top_k,
            )
        elif strategy == "page_level":
            results = PageLevelRetriever.search_pages(
                db,
                query=query,
                document_id=document_id,
                page_number=page_number,
                top_k=top_k,
            )
        elif strategy == "image_to_image":
            # Image query fallback: encode query text as multimodal prompt
            results = self.cross_modal_retriever.search_text_to_image(
                db,
                query=query,
                document_id=document_id,
                page_number=page_number,
                element_ids=element_ids,
                top_k=top_k,
            )
        else:  # hybrid RRF
            text_items = OCRTextRetriever.search(
                db,
                query=query,
                document_id=document_id,
                page_number=page_number,
                element_ids=element_ids,
                top_k=top_k * 2,
            )
            visual_items = self.cross_modal_retriever.search_text_to_image(
                db,
                query=query,
                document_id=document_id,
                page_number=page_number,
                element_ids=element_ids,
                top_k=top_k * 2,
            )
            results = HybridMultimodalFusion.rrf_fuse(
                text_results=text_items,
                visual_results=visual_items,
                top_k=top_k,
                text_weight=ocr_weight,
                visual_weight=visual_weight,
            )

        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        return {
            "query": query,
            "strategy": strategy,
            "top_k": top_k,
            "results": results,
            "total_found": len(results),
            "latency_ms": latency_ms,
        }

    def query_vision(
        self,
        db: Session,
        document_id: uuid.UUID,
        question: str,
        page_number: Optional[int] = None,
        element_ids: Optional[List[uuid.UUID]] = None,
        provider: str = "mock",
    ) -> Dict[str, Any]:
        """Performs grounded vision question answering with citation validation."""
        t0 = time.perf_counter()

        # Retrieve relevant multimodal evidence with optional page/element filters
        search_res = self.search(
            db,
            query=question,
            strategy="hybrid",
            document_id=document_id,
            page_number=page_number,
            element_ids=element_ids,
            top_k=4,
        )
        evidence_items = search_res["results"]

        # Build context
        context_str, image_paths = MultimodalContextBuilder.build_context(evidence_items)

        # Select provider with strict enforcement
        if provider == "gemini":
            if not self.gemini_vision.is_available():
                raise VisionProviderError(
                    "GEMINI_API_KEY is not configured or unavailable on the server.",
                    status_code=503,
                )
            gen_res = self.gemini_vision.generate_answer(
                question=question,
                context_text=context_str,
                image_paths=image_paths,
            )
        else:
            gen_res = self.mock_vision.generate_answer(
                question=question,
                context_text=context_str,
                image_paths=image_paths,
            )

        # Validate citations
        citation_info = MultimodalCitationManager.validate_citations(
            gen_res["answer"], evidence_items
        )

        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        # Clean evidence items to never expose raw server disk paths
        clean_evidence = []
        for it in evidence_items:
            clean_it = dict(it)
            clean_it.pop("_local_image_path", None)
            clean_evidence.append(clean_it)

        return {
            "question": question,
            "answer": gen_res["answer"],
            "citations": citation_info["valid_citations"],
            "evidence_items": clean_evidence,
            "model_used": gen_res["model"],
            "is_vision_enabled": gen_res["is_vision_enabled"],
            "warnings": gen_res["warnings"] + (
                [f"Hallucinated citations: {citation_info['hallucinated_citations']}"]
                if citation_info["hallucinated_citations"]
                else []
            ),
            "latency_ms": latency_ms,
        }

    def compare_strategies(
        self,
        db: Session,
        document_id: uuid.UUID,
        query: str,
    ) -> Dict[str, Any]:
        """Runs side-by-side comparison across OCR, Visual, Page-Level, and Hybrid strategies."""
        strategies = ["ocr_text", "text_to_image", "page_level", "hybrid"]
        comparison = {}
        for s in strategies:
            res = self.search(db, query=query, strategy=s, document_id=document_id, top_k=3)
            comparison[s] = {
                "strategy": s,
                "latency_ms": res["latency_ms"],
                "results_count": res["total_found"],
                "top_score": res["results"][0]["score"] if res["results"] else 0.0,
                "top_citation": res["results"][0]["citation_tag"] if res["results"] else "None",
            }
        return {"query": query, "document_id": str(document_id), "strategies": comparison}

    def run_benchmark(self) -> Dict[str, Any]:
        return MultimodalBenchmarkRunner.run_benchmark()
