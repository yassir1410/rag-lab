from typing import List, Dict, Any, Optional
import uuid
from sqlalchemy.orm import Session
from sqlalchemy import select, or_
from app.models.multimodal import DocumentElement, DocumentPageAsset


class OCRTextRetriever:
    """Retrieves document elements and pages based on lexical/keyword or OCR text match."""

    @classmethod
    def search(
        cls,
        db: Session,
        query: str,
        document_id: Optional[uuid.UUID] = None,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        query_terms = [t.lower().strip() for t in query.split() if len(t.strip()) > 2]
        if not query_terms:
            query_terms = [query.lower().strip()]

        stmt = select(DocumentElement)
        if document_id:
            stmt = stmt.where(DocumentElement.document_id == document_id)

        # Match content_text against terms
        conditions = [DocumentElement.content_text.ilike(f"%{term}%") for term in query_terms]
        stmt = stmt.where(or_(*conditions)).limit(top_k * 2)

        results = db.scalars(stmt).all()

        scored_items = []
        for elem in results:
            text_lower = elem.content_text.lower()
            match_count = sum(1 for term in query_terms if term in text_lower)
            score = round(min(1.0, 0.4 + (0.6 * match_count / max(1, len(query_terms)))), 3)
            scored_items.append({
                "id": str(elem.id),
                "document_id": str(elem.document_id),
                "page_number": elem.page_number,
                "element_type": elem.element_type,
                "score": score,
                "content_text": elem.content_text[:400],
                "structured_data": elem.structured_data,
                "image_path": elem.asset_path,
                "bounding_box": elem.bounding_box,
                "citation_tag": f"[{elem.element_type.upper()}_P{elem.page_number}]",
                "strategy": "ocr_text",
            })

        scored_items.sort(key=lambda x: x["score"], reverse=True)
        return scored_items[:top_k]
