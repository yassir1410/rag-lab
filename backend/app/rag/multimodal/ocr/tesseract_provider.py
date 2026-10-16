import io
import time
import shutil
from typing import Dict, Any, List, Optional
from PIL import Image

try:
    import pytesseract
    from pytesseract import Output
except ImportError:
    pytesseract = None
    Output = None

from app.rag.multimodal.ocr.base import BaseOCRProvider


class TesseractOCRProvider(BaseOCRProvider):
    """Tesseract local OCR provider with language pack discovery and educational fallback."""

    def __init__(self):
        self._tesseract_cmd = shutil.which("tesseract")
        self._available = False
        self._languages: List[str] = ["eng"]

        if self._tesseract_cmd and pytesseract is not None:
            try:
                # Test execution
                pytesseract.get_tesseract_version()
                self._available = True
                try:
                    langs = pytesseract.get_languages()
                    if langs:
                        self._languages = langs
                except Exception:
                    self._languages = ["eng", "fra"]
            except Exception:
                self._available = False

    def is_available(self) -> bool:
        return self._available

    def get_supported_languages(self) -> List[str]:
        return self._languages

    def perform_ocr(self, image_bytes: bytes, language: str = "eng") -> Dict[str, Any]:
        start_time = time.perf_counter()
        img = Image.open(io.BytesIO(image_bytes))

        # Check if requested language is available
        clean_lang = language.replace("+", " ").split()[0] if language else "eng"

        if not self._available:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return {
                "text": "",
                "confidence": 0.0,
                "words": [],
                "is_available": False,
                "execution_time_ms": round(elapsed_ms, 2),
                "warning": "Tesseract binary not found in system PATH. OCR engine is unavailable.",
            }

        try:
            # Perform OCR with detailed word data
            data = pytesseract.image_to_data(img, lang=language, output_type=Output.DICT)
            words = []
            confidences = []
            full_text_parts = []

            n_boxes = len(data["text"])
            for i in range(n_boxes):
                text_piece = data["text"][i].strip()
                conf = float(data["conf"][i])
                if text_piece and conf > 0:
                    words.append({
                        "word": text_piece,
                        "confidence": conf,
                        "bbox": {
                            "x0": data["left"][i],
                            "y0": data["top"][i],
                            "x1": data["left"][i] + data["width"][i],
                            "y1": data["top"][i] + data["height"][i],
                            "width": data["width"][i],
                            "height": data["height"][i],
                        },
                    })
                    confidences.append(conf)
                    full_text_parts.append(text_piece)

            avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
            full_text = pytesseract.image_to_string(img, lang=language).strip()
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0

            return {
                "text": full_text,
                "confidence": round(avg_conf, 1),
                "words": words,
                "is_available": True,
                "execution_time_ms": round(elapsed_ms, 2),
                "warning": None,
            }
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return {
                "text": "",
                "confidence": 0.0,
                "words": [],
                "is_available": False,
                "execution_time_ms": round(elapsed_ms, 2),
                "warning": f"Tesseract OCR processing failed: {str(e)}",
            }


class EducationalOCRComparator:
    """Computes Character Error Rate (CER) and Word Error Rate (WER) using Levenshtein distance."""

    @staticmethod
    def _levenshtein(seq1: List[Any], seq2: List[Any]) -> int:
        n, m = len(seq1), len(seq2)
        if n == 0:
            return m
        if m == 0:
            return n

        dp = [[0] * (m + 1) for _ in range(n + 1)]
        for i in range(n + 1):
            dp[i][0] = i
        for j in range(m + 1):
            dp[0][j] = j

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = 0 if seq1[i - 1] == seq2[j - 1] else 1
                dp[i][j] = min(
                    dp[i - 1][j] + 1,      # deletion
                    dp[i][j - 1] + 1,      # insertion
                    dp[i - 1][j - 1] + cost,  # substitution
                )
        return dp[n][m]

    @classmethod
    def compute_cer(cls, reference: str, hypothesis: str) -> float:
        """Character Error Rate = Levenshtein(chars_ref, chars_hyp) / len(chars_ref)."""
        ref_chars = list(reference.strip())
        hyp_chars = list(hypothesis.strip())
        if not ref_chars:
            return 0.0 if not hyp_chars else 1.0
        dist = cls._levenshtein(ref_chars, hyp_chars)
        return round(min(1.0, dist / len(ref_chars)), 4)

    @classmethod
    def compute_wer(cls, reference: str, hypothesis: str) -> float:
        """Word Error Rate = Levenshtein(words_ref, words_hyp) / len(words_ref)."""
        ref_words = reference.strip().split()
        hyp_words = hypothesis.strip().split()
        if not ref_words:
            return 0.0 if not hyp_words else 1.0
        dist = cls._levenshtein(ref_words, hyp_words)
        return round(min(1.0, dist / len(ref_words)), 4)
