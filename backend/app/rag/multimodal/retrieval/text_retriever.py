import os
import re
import uuid
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from sqlalchemy import select, or_
from app.models.multimodal import DocumentElement


STOPWORDS = {
    "the", "is", "are", "at", "which", "on", "a", "an", "and", "or", "to",
    "what", "how", "according", "cite", "exact", "source", "please", "show",
    "tell", "me", "in", "of", "for", "with", "by", "from", "its", "be",
    "do", "does", "did", "can", "could", "would", "should"
}

TABLE_INTENT_TERMS = {"table", "tableau", "tabular", "matrix", "grid"}


class OCRTextRetriever:
    """Retrieves document elements and pages based on normalized lexical/keyword matching with table awareness."""

    @classmethod
    def normalize_query_terms(cls, query: str) -> List[str]:
        """Extracts cleaned alphanumeric tokens, strips possessives and punctuation, and removes stopwords."""
        raw_tokens = re.findall(r"\b[a-zA-Z0-9_\u00c0-\u017f]+\b", query.lower())
        meaningful = []
        for t in raw_tokens:
            base = t[:-2] if t.endswith("'s") else (t[:-1] if t.endswith("s") and len(t) > 3 and not t.endswith("ss") else t)
            cand = base if len(base) > 1 else t
            if cand not in STOPWORDS and len(cand) > 1:
                meaningful.append(cand)
        # Deduplicate while preserving order
        deduped = list(dict.fromkeys(meaningful))
        return deduped if deduped else ([query.lower().strip()] if query.strip() else [])

    @classmethod
    def search(
        cls,
        db: Session,
        query: str,
        document_id: Optional[uuid.UUID] = None,
        page_number: Optional[int] = None,
        element_ids: Optional[List[uuid.UUID]] = None,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        query_terms = cls.normalize_query_terms(query)
        if not query_terms:
            return []

        stmt = select(DocumentElement)
        if document_id:
            stmt = stmt.where(DocumentElement.document_id == document_id)
        if page_number is not None:
            stmt = stmt.where(DocumentElement.page_number == page_number)
        if element_ids:
            stmt = stmt.where(DocumentElement.id.in_(element_ids))

        # Match content_text against query terms
        conditions = [DocumentElement.content_text.ilike(f"%{term}%") for term in query_terms]
        stmt = stmt.where(or_(*conditions))

        # Do NOT prematurely limit without ranking! Retrieve candidate pool.
        results = db.scalars(stmt).all()

        is_table_query = any(t in TABLE_INTENT_TERMS for t in query_terms)
        scored_items = []

        for elem in results:
            corpus = (elem.content_text + " " + str(elem.structured_data or {})).lower()
            match_count = sum(1 for term in query_terms if term in corpus)
            if match_count == 0:
                continue

            ratio = match_count / len(query_terms)
            # Table-aware boost for queries demonstrating tabular intent
            table_boost = 1.35 if (is_table_query and elem.element_type == "table") else 1.0
            score = round(min(1.0, (0.3 + 0.7 * ratio) * table_boost), 4)

            # Public authorized asset URL to avoid exposing internal container paths
            public_image_url = f"/api/v1/multimodal/documents/{elem.document_id}/pages/{elem.page_number}/image"

            local_img = elem.asset_path
            if not local_img or not os.path.exists(local_img):
                if elem.page_asset and elem.page_asset.image_path and os.path.exists(elem.page_asset.image_path):
                    local_img = elem.page_asset.image_path
                else:
                    candidate = f"/app/data/uploads/multimodal/{elem.document_id}/page_{elem.page_number:03d}.png"
                    local_img = candidate if os.path.exists(candidate) else None

            scored_items.append({
                "id": str(elem.id),
                "document_id": str(elem.document_id),
                "page_number": elem.page_number,
                "element_type": elem.element_type,
                "score": score,
                "match_count": match_count,
                "content_text": elem.content_text[:400],
                "structured_data": elem.structured_data,
                "image_path": public_image_url,
                "_local_image_path": local_img,
                "bounding_box": elem.bounding_box,
                "citation_tag": f"[{elem.element_type.upper()}_P{elem.page_number}]",
                "strategy": "ocr_text",
            })

        # Rank candidates before taking top_k
        scored_items.sort(key=lambda x: (x["score"], x["match_count"]), reverse=True)
        return scored_items[:top_k]
