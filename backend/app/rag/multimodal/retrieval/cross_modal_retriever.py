import uuid
import numpy as np
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from sqlalchemy import select
from app.models.multimodal import MultimodalEmbedding, DocumentElement, DocumentPageAsset
from app.rag.multimodal.embeddings.image_embedding_provider import MultimodalEmbeddingProvider


class CrossModalRetriever:
    """Retrieves document images and figures using text queries embedded into CLIP space."""

    def __init__(self, embedding_provider: Optional[MultimodalEmbeddingProvider] = None):
        self.provider = embedding_provider or MultimodalEmbeddingProvider()

    def search_text_to_image(
        self,
        db: Session,
        query: str,
        document_id: Optional[uuid.UUID] = None,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        query_vec = np.array(self.provider.encode_text(query), dtype=np.float32)

        stmt = select(MultimodalEmbedding).where(
            MultimodalEmbedding.embedding_type.in_(["image", "page"])
        )
        if document_id:
            stmt = stmt.where(MultimodalEmbedding.document_id == document_id)

        embeddings_records = db.scalars(stmt).all()
        scored = []

        for record in embeddings_records:
            if record.vector is None:
                continue
            rec_vec = np.array(record.vector, dtype=np.float32)
            # Cosine similarity (since vectors are L2-normalized)
            sim = float(np.dot(query_vec, rec_vec))

            elem = record.element
            page_asset = record.page_asset
            page_num = elem.page_number if elem else (page_asset.page_number if page_asset else 1)
            elem_type = elem.element_type if elem else "figure"
            img_path = (elem.asset_path if elem else None) or (page_asset.image_path if page_asset else None)

            scored.append({
                "id": str(record.id),
                "document_id": str(record.document_id),
                "page_number": page_num,
                "element_type": elem_type,
                "score": round(max(0.0, min(1.0, (sim + 1.0) / 2.0)), 4),
                "content_text": (elem.content_text if elem else "") or f"Visual element on page {page_num}",
                "structured_data": elem.structured_data if elem else {},
                "image_path": img_path,
                "bounding_box": elem.bounding_box if elem else None,
                "citation_tag": f"[IMG_P{page_num}]",
                "strategy": "text_to_image",
            })

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]
