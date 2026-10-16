import hashlib
import math
import time
import uuid
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from app.rag.chunking.base import ChunkingStrategy, Chunk, ChunkMetadata, ChunkingResult
from app.rag.chunking.metrics import MetricsCalculator
from app.rag.chunking.strategies.sentence import SentenceChunking
from app.rag.chunking.tokenizer import default_tokenizer


class BaseEmbeddingProvider(ABC):
    @abstractmethod
    def encode(self, texts: List[str]) -> List[List[float]]:
        pass

    @property
    @abstractmethod
    def is_available(self) -> bool:
        pass


class DeterministicMockEmbeddingProvider(BaseEmbeddingProvider):
    """
    Deterministic embedding provider using character n-gram hashing.
    Used for unit tests to provide reproducible semantic vectors without downloading large models.
    """
    def __init__(self, dim: int = 64):
        self.dim = dim

    @property
    def is_available(self) -> bool:
        return True

    def encode(self, texts: List[str]) -> List[List[float]]:
        vectors = []
        for text in texts:
            vec = [0.0] * self.dim
            words = text.lower().split()
            for w in words:
                h = int(hashlib.md5(w.encode("utf-8")).hexdigest(), 16)
                idx = h % self.dim
                vec[idx] += 1.0

            norm = math.sqrt(sum(x * x for x in vec))
            if norm > 0:
                vec = [x / norm for x in vec]
            vectors.append(vec)
        return vectors


class SentenceTransformersProvider(BaseEmbeddingProvider):
    """
    Local neural embedding provider using sentence-transformers (e.g. all-MiniLM-L6-v2).
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
        self._load_error = None

    @property
    def is_available(self) -> bool:
        if self._model is not None:
            return True
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            return True
        except Exception as e:
            self._load_error = str(e)
            return False

    def encode(self, texts: List[str]) -> List[List[float]]:
        if not self.is_available:
            raise RuntimeError(
                f"SentenceTransformer model '{self.model_name}' could not be loaded: {self._load_error}"
            )
        embeddings = self._model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return embeddings.tolist()


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Compute cosine similarity between two unit vectors."""
    dot = sum(a * b for a, b in zip(v1, v2))
    return max(-1.0, min(1.0, dot))


class SemanticChunking(ChunkingStrategy):
    """
    Semantic chunking based on sentence embeddings and cosine distance drops.
    Identifies natural conceptual transitions (breakpoints) where topic shifts occur.
    """

    def __init__(self, embedding_provider: Optional[BaseEmbeddingProvider] = None):
        self.embedding_provider = embedding_provider or DeterministicMockEmbeddingProvider()
        self.sentence_segmenter = SentenceChunking()

    @property
    def strategy_name(self) -> str:
        return "semantic"

    @property
    def display_name(self) -> str:
        return "Semantic Similarity"

    @property
    def description(self) -> str:
        return (
            "Computes dense embeddings for sentences and measures cosine similarity between "
            "adjacent sentences. Splits text when semantic distance crosses a threshold, "
            "preserving conceptual coherence rather than arbitrary lengths."
        )

    @property
    def default_config(self) -> Dict[str, Any]:
        return {
            "similarity_threshold": 0.70,  # Below this similarity, a split occurs
            "min_chunk_size": 100,        # Minimum characters before splitting
            "max_chunk_size": 1200,       # Maximum characters forcing a split
            "embedding_model": "deterministic_mock",  # or "all-MiniLM-L6-v2"
        }

    def validate_config(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        threshold = float(config_dict.get("similarity_threshold", 0.70))
        min_size = int(config_dict.get("min_chunk_size", 100))
        max_size = int(config_dict.get("max_chunk_size", 1200))
        model = str(config_dict.get("embedding_model", "deterministic_mock"))

        if not (0.0 <= threshold <= 1.0):
            raise ValueError(f"similarity_threshold must be between 0.0 and 1.0, got {threshold}")
        if min_size <= 0:
            raise ValueError(f"min_chunk_size must be positive, got {min_size}")
        if max_size <= min_size:
            raise ValueError(f"max_chunk_size ({max_size}) must be greater than min_chunk_size ({min_size})")

        return {
            "similarity_threshold": threshold,
            "min_chunk_size": min_size,
            "max_chunk_size": max_size,
            "embedding_model": model,
        }

    def split(
        self,
        text: str,
        document_id: uuid.UUID,
        config_dict: Dict[str, Any],
        document_context: Optional[Dict[str, Any]] = None,
    ) -> ChunkingResult:
        start_time = time.perf_counter()
        config = self.validate_config(config_dict)
        threshold = config["similarity_threshold"]
        min_size = config["min_chunk_size"]
        max_size = config["max_chunk_size"]

        if not text or not text.strip():
            duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
            metrics = MetricsCalculator.calculate(
                chunks=[], source_text="", nominal_overlap=0, target_size=max_size, duration_ms=duration_ms
            )
            return ChunkingResult(
                strategy=self.strategy_name,
                chunks=[],
                metrics=metrics,
                config=config,
                processing_duration_ms=duration_ms,
            )

        # 1. Segment into sentences
        sentence_spans = self.sentence_segmenter._split_into_sentences(text)
        if len(sentence_spans) <= 1:
            chunk = Chunk(
                id=uuid.uuid4(),
                document_id=document_id,
                sequence_index=0,
                content=text.strip(),
                char_count=len(text.strip()),
                token_count=default_tokenizer.count_tokens(text),
                metadata=ChunkMetadata(start_char=0, end_char=len(text)),
            )
            duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
            metrics = MetricsCalculator.calculate([chunk], text, 0, max_size, duration_ms)
            return ChunkingResult(
                strategy=self.strategy_name,
                chunks=[chunk],
                metrics=metrics,
                config=config,
                processing_duration_ms=duration_ms,
            )

        # 2. Compute embeddings for sentences
        sentence_texts = [s[2] for s in sentence_spans]
        embeddings = self.embedding_provider.encode(sentence_texts)

        # 3. Calculate adjacent similarities
        similarities: List[float] = []
        for idx in range(len(embeddings) - 1):
            sim = cosine_similarity(embeddings[idx], embeddings[idx + 1])
            similarities.append(round(sim, 4))

        # 4. Identify breakpoints and build chunks
        chunks: List[Chunk] = []
        curr_sentences: List[Tuple[int, int, str]] = [sentence_spans[0]]
        curr_len = len(sentence_spans[0][2])
        seq_idx = 0

        for idx in range(len(similarities)):
            sim = similarities[idx]
            next_sent = sentence_spans[idx + 1]
            next_len = len(next_sent[2])

            is_breakpoint = (
                (sim < threshold and curr_len >= min_size)
                or (curr_len + next_len > max_size)
            )

            if is_breakpoint:
                # Flush current sentences as chunk
                chunk_text = " ".join(s[2] for s in curr_sentences)
                chunk = Chunk(
                    id=uuid.uuid4(),
                    document_id=document_id,
                    sequence_index=seq_idx,
                    content=chunk_text,
                    char_count=len(chunk_text),
                    token_count=default_tokenizer.count_tokens(chunk_text),
                    metadata=ChunkMetadata(
                        start_char=curr_sentences[0][0],
                        end_char=curr_sentences[-1][1],
                        content_type="text",
                        breakpoint_distance=round(1.0 - sim, 4),
                        custom={"similarity_to_next": sim},
                    ),
                )
                chunks.append(chunk)
                seq_idx += 1
                curr_sentences = [next_sent]
                curr_len = next_len
            else:
                curr_sentences.append(next_sent)
                curr_len += next_len

        # Flush remaining sentences
        if curr_sentences:
            chunk_text = " ".join(s[2] for s in curr_sentences)
            chunk = Chunk(
                id=uuid.uuid4(),
                document_id=document_id,
                sequence_index=seq_idx,
                content=chunk_text,
                char_count=len(chunk_text),
                token_count=default_tokenizer.count_tokens(chunk_text),
                metadata=ChunkMetadata(
                    start_char=curr_sentences[0][0],
                    end_char=curr_sentences[-1][1],
                    content_type="text",
                ),
            )
            chunks.append(chunk)

        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
        metrics = MetricsCalculator.calculate(
            chunks=chunks,
            source_text=text,
            nominal_overlap=0,
            target_size=max_size,
            duration_ms=duration_ms,
            semantic_similarities=similarities,
        )

        return ChunkingResult(
            strategy=self.strategy_name,
            chunks=chunks,
            metrics=metrics,
            config=config,
            processing_duration_ms=duration_ms,
        )
