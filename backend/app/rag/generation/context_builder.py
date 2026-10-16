import hashlib
from typing import List, Tuple
from app.rag.chunking.tokenizer import default_tokenizer
from app.rag.retrieval.schemas import SearchResultItem
from app.rag.generation.schemas import ContextSourceItem


class ContextBuilder:
    """
    Assembles, deduplicates, and formats retrieved chunks into structured context.
    Assigns sequential stable citation tags [S1], [S2], ..., [Sn].
    Applies token budgets, truncates oversized chunks, and drops lower-ranked chunks
    if context capacity is exceeded.
    """

    def __init__(self):
        self.tokenizer = default_tokenizer

    def build_context(
        self,
        retrieved_items: List[SearchResultItem],
        max_context_tokens: int = 2048,
    ) -> Tuple[str, List[ContextSourceItem], List[str]]:
        """
        Builds the formatted context string and context source list.
        Returns:
            (context_text, source_items, warnings)
        """
        warnings: List[str] = []
        if not retrieved_items:
            return "No relevant context passages were retrieved from the document database.", [], warnings

        source_items: List[ContextSourceItem] = []
        context_blocks: List[str] = []

        seen_hashes = set()
        accumulated_tokens = 0

        citation_index = 1
        for item in retrieved_items:
            # 1. Deduplicate by normalized content hash
            content_cleaned = item.content.strip()
            content_hash = hashlib.sha256(content_cleaned.encode("utf-8")).hexdigest()
            if content_hash in seen_hashes:
                continue
            seen_hashes.add(content_hash)

            # 2. Assign stable citation ID
            c_tag = f"[S{citation_index}]"

            # 3. Handle chunk content and token budget
            chunk_tokens = self.tokenizer.count_tokens(content_cleaned)

            # If adding this chunk exceeds max_context_tokens, check if we truncate or stop
            if accumulated_tokens + chunk_tokens > max_context_tokens:
                remaining_tokens = max_context_tokens - accumulated_tokens
                if remaining_tokens > 50:
                    # Truncate this chunk to fit remainder
                    spans = self.tokenizer.tokenize_spans(content_cleaned)
                    if len(spans) > remaining_tokens:
                        cutoff = spans[remaining_tokens - 1][1]
                        content_cleaned = content_cleaned[:cutoff] + " ... [TRUNCATED]"
                        chunk_tokens = self.tokenizer.count_tokens(content_cleaned)
                        warnings.append(
                            f"Source {c_tag} was truncated to fit the {max_context_tokens}-token context budget."
                        )
                    else:
                        break
                else:
                    warnings.append(
                        f"Context budget ({max_context_tokens} tokens) reached. Remaining {len(retrieved_items) - citation_index + 1} lower-ranked chunks were omitted."
                    )
                    break

            accumulated_tokens += chunk_tokens

            page_info = f", Page: {item.page_number}" if item.page_number is not None else ""
            block = (
                f"Source {c_tag} (Document: {item.document_name}{page_info}, Score: {item.similarity_score:.4f}):\n"
                f"{content_cleaned}"
            )
            context_blocks.append(block)

            source_item = ContextSourceItem(
                citation_id=c_tag,
                chunk_id=item.chunk_id,
                document_id=item.document_id,
                document_name=item.document_name,
                mime_type=item.mime_type,
                sequence_index=item.sequence_index,
                token_count=chunk_tokens,
                char_count=len(content_cleaned),
                content=content_cleaned,
                page_number=item.page_number,
                similarity_score=item.similarity_score,
                raw_distance=item.raw_distance,
                distance_metric=item.distance_metric,
                was_cited=False,
                metadata=item.metadata,
            )
            source_items.append(source_item)
            citation_index += 1

        formatted_context = "\n\n".join(context_blocks)
        return formatted_context, source_items, warnings


context_builder = ContextBuilder()
