import re
import time
import uuid
from typing import List, Dict, Any, Optional, Tuple
from app.rag.chunking.base import ChunkingStrategy, Chunk, ChunkMetadata, ChunkingResult
from app.rag.chunking.metrics import MetricsCalculator
from app.rag.chunking.tokenizer import default_tokenizer


class SentenceChunking(ChunkingStrategy):
    """
    Sentence-based chunking strategy with robust French and English segmentation.
    Segments text into linguistic sentences, preserving grammar and boundary integrity,
    then combines sentences into chunks respecting the target budget and overlap.
    """

    # Common English & French abbreviations that should not trigger sentence breaks
    ABBREVIATIONS = {
        # English
        "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "vs", "etc", "inc", "ltd", "dept", "e.g", "i.e",
        # French
        "m", "mme", "mlle", "me", "av", "apr", "art", "bd", "c.-à-d", "cf", "ch", "chap", "col", "éd", "ex", "fasc", "fig", "hab", "n°", "p", "pp", "par", "r-v", "s", "st", "ste", "t", "vol",
    }

    @property
    def strategy_name(self) -> str:
        return "sentence"

    @property
    def display_name(self) -> str:
        return "Sentence-Based"

    @property
    def description(self) -> str:
        return (
            "Segments text into grammatical sentences respecting both English and French "
            "abbreviation conventions, then groups sentences into chunks up to the target size limit. "
            "Never breaks sentences in the middle."
        )

    @property
    def default_config(self) -> Dict[str, Any]:
        return {
            "chunk_size": 500,
            "sentence_overlap": 1,  # number of sentences to repeat between chunks
            "measurement_unit": "character",  # "character" or "token"
            "language": "auto",  # "auto", "fr", "en"
        }

    def validate_config(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        chunk_size = int(config_dict.get("chunk_size", 500))
        sentence_overlap = int(config_dict.get("sentence_overlap", 1))
        unit = str(config_dict.get("measurement_unit", "character")).lower()
        language = str(config_dict.get("language", "auto")).lower()

        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if sentence_overlap < 0:
            raise ValueError(f"sentence_overlap must be non-negative, got {sentence_overlap}")
        if unit not in ("character", "token"):
            raise ValueError(f"measurement_unit must be 'character' or 'token', got '{unit}'")

        return {
            "chunk_size": chunk_size,
            "sentence_overlap": sentence_overlap,
            "measurement_unit": unit,
            "language": language,
        }

    def _split_into_sentences(self, text: str) -> List[Tuple[int, int, str]]:
        """
        Segments text into sentences while tracking (start_char, end_char, text).
        Protects abbreviations, numbers with decimals (3.14 or 3,14), and quotes.
        """
        if not text:
            return []

        # Find potential sentence terminators: . ? ! followed by whitespace, quotes or end of string
        terminator_pattern = re.compile(r'([.?!]+[\'"»”]?)(?=\s+|$)', re.UNICODE)
        matches = list(terminator_pattern.finditer(text))

        if not matches:
            return [(0, len(text), text.strip())]

        sentences: List[Tuple[int, int, str]] = []
        current_start = 0

        for match in matches:
            punct_end = match.end()
            candidate_sentence = text[current_start:punct_end].strip()

            # Check if terminator is inside an abbreviation (e.g. "Dr.", "M.", "e.g.")
            # Get the word directly before the punctuation
            prefix_match = re.search(r'([a-zA-ZÀ-ÿ\.\-]+)$', text[current_start:match.start() + 1])
            if prefix_match:
                word = prefix_match.group(1).lower().rstrip('.')
                if word in self.ABBREVIATIONS:
                    # Skip break on abbreviation
                    continue

            # Check for numeric decimals: e.g. "version 3.14"
            if re.search(r'\d+\.\d*$', text[current_start:match.start() + 1]):
                continue

            if candidate_sentence:
                # Find exact start position after stripping
                actual_start = text.find(candidate_sentence, current_start)
                actual_end = actual_start + len(candidate_sentence)
                sentences.append((actual_start, actual_end, candidate_sentence))
                current_start = punct_end

        # Handle remaining trailing text
        trailing = text[current_start:].strip()
        if trailing:
            actual_start = text.find(trailing, current_start)
            sentences.append((actual_start, actual_start + len(trailing), trailing))

        return sentences

    def _get_length(self, text: str, unit: str) -> int:
        if unit == "token":
            return default_tokenizer.count_tokens(text)
        return len(text)

    def split(
        self,
        text: str,
        document_id: uuid.UUID,
        config_dict: Dict[str, Any],
        document_context: Optional[Dict[str, Any]] = None,
    ) -> ChunkingResult:
        start_time = time.perf_counter()
        config = self.validate_config(config_dict)
        chunk_size = config["chunk_size"]
        sentence_overlap = config["sentence_overlap"]
        unit = config["measurement_unit"]

        if not text or not text.strip():
            duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
            metrics = MetricsCalculator.calculate(
                chunks=[],
                source_text="",
                nominal_overlap=sentence_overlap,
                target_size=chunk_size,
                duration_ms=duration_ms,
            )
            return ChunkingResult(
                strategy=self.strategy_name,
                chunks=[],
                metrics=metrics,
                config=config,
                processing_duration_ms=duration_ms,
            )

        sentence_spans = self._split_into_sentences(text)
        chunks: List[Chunk] = []

        if not sentence_spans:
            duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
            return ChunkingResult(
                strategy=self.strategy_name,
                chunks=[],
                metrics=MetricsCalculator.calculate([], text, sentence_overlap, chunk_size, duration_ms),
                config=config,
                processing_duration_ms=duration_ms,
            )

        i = 0
        seq_idx = 0
        total_sentences = len(sentence_spans)

        while i < total_sentences:
            curr_sentences: List[Tuple[int, int, str]] = []
            curr_len = 0

            j = i
            while j < total_sentences:
                s_start, s_end, s_text = sentence_spans[j]
                s_len = self._get_length(s_text, unit)

                # If single sentence exceeds budget, add it alone
                if not curr_sentences and s_len >= chunk_size:
                    curr_sentences.append(sentence_spans[j])
                    j += 1
                    break

                if curr_len + s_len > chunk_size and curr_sentences:
                    break

                curr_sentences.append(sentence_spans[j])
                curr_len += s_len
                j += 1

            chunk_text = " ".join(s[2] for s in curr_sentences)
            start_char = curr_sentences[0][0]
            end_char = curr_sentences[-1][1]

            chunk = Chunk(
                id=uuid.uuid4(),
                document_id=document_id,
                sequence_index=seq_idx,
                content=chunk_text,
                char_count=len(chunk_text),
                token_count=default_tokenizer.count_tokens(chunk_text),
                metadata=ChunkMetadata(
                    start_char=start_char,
                    end_char=end_char,
                    content_type="text",
                    custom={"sentence_count": len(curr_sentences)},
                ),
            )
            chunks.append(chunk)
            seq_idx += 1

            # Advance index considering sentence_overlap
            if j >= total_sentences:
                break
            # Step forward: advance by count of sentences in current chunk minus overlap
            step = max(1, len(curr_sentences) - sentence_overlap)
            i += step

        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
        metrics = MetricsCalculator.calculate(
            chunks=chunks,
            source_text=text,
            nominal_overlap=sentence_overlap,
            target_size=chunk_size,
            duration_ms=duration_ms,
        )

        return ChunkingResult(
            strategy=self.strategy_name,
            chunks=chunks,
            metrics=metrics,
            config=config,
            processing_duration_ms=duration_ms,
        )
