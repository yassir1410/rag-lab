import hashlib
import uuid
from typing import List, Set, Dict, Any, Optional
from sqlalchemy.orm import Session
from sqlalchemy import and_

from app.models.chunking import DocumentChunk
from app.rag.retrieval.schemas import SearchResultItem


def compute_content_hash(text: str) -> str:
    """Calculates SHA256 prefix hash of normalized text for exact duplicate detection."""
    normalized = "".join(text.lower().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def jaccard_similarity(text1: str, text2: str) -> float:
    """Calculates word-level Jaccard similarity between two passages."""
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    if not words1 or not words2:
        return 0.0
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    return float(intersection) / float(union) if union > 0 else 0.0


class ContextOptimizer:
    """
    Optimizes retrieved passages before injection into LLM prompt templates:
    1. Neighboring Chunk Expansion (+-1 contiguous chunks) to restore local document context.
    2. Redundancy Pruning to eliminate duplicate or near-identical passages.
    3. Adaptive Top-K Cutoff based on relative score degradation.
    """

    def expand_neighboring_chunks(
        self,
        session: Session,
        items: List[SearchResultItem],
        max_expansion: int = 1,
    ) -> List[SearchResultItem]:
        """
        Fetches adjacent contiguous chunks (sequence_index +- max_expansion)
        within the same document to restore context boundaries.
        """
        if not items:
            return []

        expanded_items: List[SearchResultItem] = []
        seen_chunk_ids = {item.chunk_id for item in items}

        for item in items:
            expanded_items.append(item)

            # Query database for contiguous neighbor chunks
            try:
                neighbors = (
                    session.query(DocumentChunk)
                    .filter(
                        and_(
                            DocumentChunk.document_id == item.document_id,
                            DocumentChunk.sequence_index.in_(
                                [item.sequence_index - 1, item.sequence_index + 1]
                            ),
                        )
                    )
                    .all()
                )

                for n_chunk in neighbors:
                    if n_chunk.id not in seen_chunk_ids:
                        seen_chunk_ids.add(n_chunk.id)
                        meta = n_chunk.metadata_json or {}
                        expanded_items.append(
                            SearchResultItem(
                                chunk_id=n_chunk.id,
                                document_id=item.document_id,
                                document_name=item.document_name,
                                mime_type=item.mime_type,
                                sequence_index=n_chunk.sequence_index,
                                token_count=n_chunk.token_count,
                                char_count=n_chunk.char_count,
                                content=n_chunk.content,
                                metadata=meta,
                                page_number=meta.get("page_number"),
                                raw_distance=item.raw_distance + 0.1,
                                similarity_score=item.similarity_score * 0.9,
                                distance_metric="neighbor_expansion",
                                distance_label="Neighbor",
                                similarity_label=f"Expanded (seq {n_chunk.sequence_index})",
                            )
                        )
            except Exception:
                # If neighbor lookup encounters any error, continue with primary items
                pass

        return expanded_items

    def prune_redundancy(
        self,
        items: List[SearchResultItem],
        similarity_threshold: float = 0.80,
    ) -> List[SearchResultItem]:
        """
        Removes near-duplicate chunks using Jaccard word-overlap and content hashing.
        """
        if not items:
            return []

        pruned: List[SearchResultItem] = []
        seen_hashes: Set[str] = set()

        for item in items:
            item_hash = compute_content_hash(item.content)
            if item_hash in seen_hashes:
                continue

            # Check semantic overlap against previously kept items
            is_redundant = False
            for kept in pruned:
                if item.document_id == kept.document_id:
                    sim = jaccard_similarity(item.content, kept.content)
                    if sim >= similarity_threshold:
                        is_redundant = True
                        break

            if not is_redundant:
                seen_hashes.add(item_hash)
                pruned.append(item)

        return pruned

    def apply_adaptive_cutoff(
        self,
        items: List[SearchResultItem],
        relative_threshold: float = 0.35,
        min_k: int = 2,
    ) -> List[SearchResultItem]:
        """
        Dynamically limits candidates if score falls steeply below top-1 score.
        """
        if len(items) <= min_k:
            return items

        top_score = items[0].similarity_score
        if top_score <= 0:
            return items

        filtered = [items[0]]
        for item in items[1:]:
            rel_ratio = item.similarity_score / top_score
            if len(filtered) < min_k or rel_ratio >= relative_threshold:
                filtered.append(item)
            else:
                break

        return filtered


context_optimizer = ContextOptimizer()
