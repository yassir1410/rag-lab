import statistics
from typing import List, Dict, Any, Optional
from app.rag.chunking.base import Chunk


class MetricsCalculator:
    """Calculates objective statistical and linguistic metrics on chunk collections."""

    @staticmethod
    def calculate(
        chunks: List[Chunk],
        source_text: str,
        nominal_overlap: int,
        target_size: int,
        duration_ms: float,
        semantic_similarities: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        total_chunks = len(chunks)
        if total_chunks == 0:
            return {
                "total_chunks": 0,
                "character_metrics": {
                    "min": 0, "max": 0, "mean": 0.0, "median": 0.0
                },
                "token_metrics": {
                    "min": 0, "max": 0, "mean": 0.0, "median": 0.0,
                    "percentiles": {"p10": 0, "p25": 0, "p50": 0, "p75": 0, "p90": 0}
                },
                "overlap_metrics": {
                    "nominal": nominal_overlap,
                    "mean_observed_chars": 0,
                    "max_observed_chars": 0
                },
                "duplicated_text_ratio": 0.0,
                "small_chunks_count": 0,
                "processing_duration_ms": duration_ms,
                "semantic_metrics": None,
            }

        char_counts = [c.char_count for c in chunks]
        token_counts = [c.token_count for c in chunks]

        # Calculate observed overlap between consecutive chunks
        observed_overlaps: List[int] = []
        for i in range(len(chunks) - 1):
            curr_text = chunks[i].content
            next_text = chunks[i + 1].content

            # Look for suffix of curr_text that matches prefix of next_text
            overlap_len = 0
            # Test overlaps from min(len(curr), len(next)) down to 1
            max_possible = min(len(curr_text), len(next_text))
            # Optimization: check from expected nominal overlap range
            for test_len in range(max_possible, 0, -1):
                if curr_text[-test_len:] == next_text[:test_len]:
                    overlap_len = test_len
                    break
            observed_overlaps.append(overlap_len)

        mean_observed_overlap = (
            round(statistics.mean(observed_overlaps), 1) if observed_overlaps else 0
        )
        max_observed_overlap = max(observed_overlaps) if observed_overlaps else 0

        # Percentiles helper
        sorted_tokens = sorted(token_counts)
        def get_percentile(data: List[int], p: float) -> int:
            if not data:
                return 0
            idx = int(round((len(data) - 1) * p))
            return data[idx]

        token_percentiles = {
            "p10": get_percentile(sorted_tokens, 0.10),
            "p25": get_percentile(sorted_tokens, 0.25),
            "p50": get_percentile(sorted_tokens, 0.50),
            "p75": get_percentile(sorted_tokens, 0.75),
            "p90": get_percentile(sorted_tokens, 0.90),
        }

        # Duplicated text ratio
        # Definition: Total characters across all chunks minus source text characters, divided by source text length
        total_chunk_chars = sum(char_counts)
        source_len = len(source_text) if source_text else 1
        dup_chars = max(0, total_chunk_chars - source_len)
        duplicated_text_ratio = round(dup_chars / source_len, 4)

        # Small chunks (under 25% of target size)
        small_threshold = max(10, int(target_size * 0.25))
        small_chunks_count = sum(1 for c in chunks if c.char_count < small_threshold)

        # Semantic metrics
        semantic_data = None
        if semantic_similarities:
            semantic_data = {
                "similarities": semantic_similarities,
                "mean_similarity": round(statistics.mean(semantic_similarities), 4),
                "min_similarity": round(min(semantic_similarities), 4),
                "max_similarity": round(max(semantic_similarities), 4),
            }

        return {
            "total_chunks": total_chunks,
            "character_metrics": {
                "min": min(char_counts),
                "max": max(char_counts),
                "mean": round(statistics.mean(char_counts), 1),
                "median": round(statistics.median(char_counts), 1),
            },
            "token_metrics": {
                "min": min(token_counts),
                "max": max(token_counts),
                "mean": round(statistics.mean(token_counts), 1),
                "median": round(statistics.median(token_counts), 1),
                "percentiles": token_percentiles,
            },
            "overlap_metrics": {
                "nominal": nominal_overlap,
                "mean_observed_chars": mean_observed_overlap,
                "max_observed_chars": max_observed_overlap,
            },
            "duplicated_text_ratio": duplicated_text_ratio,
            "small_chunks_count": small_chunks_count,
            "processing_duration_ms": duration_ms,
            "semantic_metrics": semantic_data,
        }
