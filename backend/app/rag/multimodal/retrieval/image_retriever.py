import uuid
import numpy as np
from typing import List, Dict, Any, Optional, Union
from PIL import Image
from sqlalchemy.orm import Session
from sqlalchemy import select
from app.models.multimodal import MultimodalEmbedding
from app.rag.multimodal.embeddings.image_embedding_provider import MultimodalEmbeddingProvider


class ImageToImageRetriever:
    """Retrieves document images that are visually similar to a query image."""

    def __init__(self, embedding_provider: Optional[MultimodalEmbeddingProvider] = None):
        self.provider = embedding_provider or MultimodalEmbeddingProvider()

    def search_similar_images(
        self,
        db: Session,
        query_image: Union[str, Image.Image],
        document_id: Optional[uuid.UUID] = None,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        query_vec = np.array(self.provider.encode_image(query_image), dtype=np.float32)

        stmt = select(MultimodalEmbedding).where(
            MultimodalEmbedding.embedding_type.in_(["image", "page"])
        )
        if document_id:
            stmt = stmt.where(MultimodalEmbedding.document_id == document_id)

        records = db.scalars(stmt).all()
        scored = []

        for record in records:
            if record.vector is None:
                continue
            rec_vec = np.array(record.vector, dtype=np.float32)
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
                "content_text": (elem.content_text if elem else "") or f"Visual figure on page {page_num}",
                "structured_data": elem.structured_data if elem else {},
                "image_path": img_path,
                "bounding_box": elem.bounding_box if elem else None,
                "citation_tag": f"[IMG_P{page_num}]",
                "strategy": "image_to_image",
            })

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]
