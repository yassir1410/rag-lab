from abc import ABC, abstractmethod
from typing import Dict, Any, List


class BaseOCRProvider(ABC):
    """Abstract base provider for Optical Character Recognition (OCR)."""

    @abstractmethod
    def is_available(self) -> bool:
        """Returns True if the underlying OCR engine and dependencies are installed and operational."""
        pass

    @abstractmethod
    def get_supported_languages(self) -> List[str]:
        """Returns list of installed language pack codes (e.g., ['eng', 'fra'])."""
        pass

    @abstractmethod
    def perform_ocr(self, image_bytes: bytes, language: str = "eng") -> Dict[str, Any]:
        """Execute OCR on raw image bytes.
        
        Returns a dict containing:
        - text: str
        - confidence: float (0.0 to 100.0)
        - words: List[dict] (word, bbox, conf)
        - is_available: bool
        - execution_time_ms: float
        - warning: Optional[str]
        """
        pass
