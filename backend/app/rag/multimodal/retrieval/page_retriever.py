import uuid
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from sqlalchemy import select
from app.models.multimodal import DocumentPageAsset


class PageLevelRetriever:
    """Retrieves whole document pages based on page-level content, OCR text, and visual identity."""

    @classmethod
    def search_pages(
        cls,
        db: Session,
        query: str,
        document_id: Optional[uuid.UUID] = None,
        page_number: Optional[int] = None,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        terms = [t.lower().strip() for t in query.split() if len(t.strip()) > 2]
        if not terms:
            terms = [query.lower().strip()]

        stmt = select(DocumentPageAsset)
        if document_id:
            stmt = stmt.where(DocumentPageAsset.document_id == document_id)
        if page_number:
            stmt = stmt.where(DocumentPageAsset.page_number == page_number)

        pages = db.scalars(stmt).all()
        scored_pages = []

        for p in pages:
            combined_text = (p.ocr_text or "")
            # Count term matches
            text_lower = combined_text.lower()
            matches = sum(1 for t in terms if t in text_lower)
            score = round(min(1.0, 0.3 + (0.7 * matches / max(1, len(terms)))), 3) if matches > 0 else 0.2

            asset_url = f"/api/v1/multimodal/documents/{p.document_id}/pages/{p.page_number}/image"

            scored_pages.append({
                "id": str(p.id),
                "document_id": str(p.document_id),
                "page_number": p.page_number,
                "element_type": "page",
                "score": score,
                "content_text": (combined_text[:300] + "...") if combined_text else f"Page {p.page_number} (Scanned: {p.is_scanned})",
                "structured_data": {
                    "width": p.width,
                    "height": p.height,
                    "is_scanned": p.is_scanned,
                    "native_char_count": p.native_text_char_count,
                },
                "image_path": asset_url,
                "_local_image_path": p.image_path,
                "bounding_box": None,
                "citation_tag": f"[PAGE_{p.page_number}]",
                "strategy": "page_level",
            })

        scored_pages.sort(key=lambda x: x["score"], reverse=True)
        return scored_pages[:top_k]
