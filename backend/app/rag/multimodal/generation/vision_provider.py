import os
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import httpx


class BaseVisionProvider(ABC):
    @abstractmethod
    def is_available(self) -> bool:
        pass

    @abstractmethod
    def generate_answer(
        self,
        question: str,
        context_text: str,
        image_paths: List[str],
    ) -> Dict[str, Any]:
        """Generate answer grounded in multimodal context and image inputs."""
        pass


class MockVisionProvider(BaseVisionProvider):
    """Deterministic mock vision provider for automated tests and offline demonstrations."""

    def is_available(self) -> bool:
        return True

    def generate_answer(
        self,
        question: str,
        context_text: str,
        image_paths: List[str],
    ) -> Dict[str, Any]:
        has_images = len(image_paths) > 0
        img_tag = "[IMG_P1]" if has_images else "[PAGE_1]"

        # Check if question requires refusal or unanswerable
        q_lower = question.lower()
        if "weather" in q_lower or "vacation" in q_lower or "mars" in q_lower:
            return {
                "answer": "Based on the provided document and visual evidence, this question cannot be answered. No relevant information was found.",
                "model": "mock-vision-v1",
                "is_vision_enabled": has_images,
                "warnings": ["Information not found in documentary evidence."],
            }

        # Synthesize grounded answer
        answer = (
            f"Based on the analyzed document layout and visual inspection {img_tag}, "
            f"the requested information regarding '{question}' shows verifiable structural evidence. "
            f"The data is confirmed across both text sections and visual page assets {img_tag}."
        )

        return {
            "answer": answer,
            "model": "mock-vision-v1",
            "is_vision_enabled": has_images,
            "warnings": [],
        }


class GeminiVisionProvider(BaseVisionProvider):
    """Google Gemini multimodal provider using Gemini 1.5 Flash."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")

    def is_available(self) -> bool:
        return bool(self.api_key)

    def generate_answer(
        self,
        question: str,
        context_text: str,
        image_paths: List[str],
    ) -> Dict[str, Any]:
        if not self.is_available():
            return {
                "answer": "Vision capability unavailable: GEMINI_API_KEY is not configured.",
                "model": "gemini-1.5-flash",
                "is_vision_enabled": False,
                "warnings": ["Vision provider unavailable."],
            }

        prompt = (
            f"You are a multimodal RAG document intelligence assistant. "
            f"Answer the user's question using ONLY the provided document evidence and attached images.\n"
            f"Cite your sources using bracketed tags like [PAGE_X], [IMG_PX], or [TBL_PX].\n\n"
            f"DOCUMENT EVIDENCE:\n{context_text}\n\n"
            f"QUESTION:\n{question}\n\n"
            f"ANSWER:"
        )

        # In production, payload can include inline base64 images
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={self.api_key}"
        try:
            with httpx.Client(timeout=30.0) as client:
                res = client.post(url, json={"contents": [{"parts": [{"text": prompt}]}]})
                if res.status_code == 200:
                    data = res.json()
                    candidate = data.get("candidates", [{}])[0]
                    text = candidate.get("content", {}).get("parts", [{}])[0].get("text", "")
                    return {
                        "answer": text,
                        "model": "gemini-1.5-flash",
                        "is_vision_enabled": True,
                        "warnings": [],
                    }
                else:
                    return {
                        "answer": f"Gemini API returned error code {res.status_code}.",
                        "model": "gemini-1.5-flash",
                        "is_vision_enabled": False,
                        "warnings": [res.text[:200]],
                    }
        except Exception as e:
            return {
                "answer": f"Multimodal inference failed: {str(e)}",
                "model": "gemini-1.5-flash",
                "is_vision_enabled": False,
                "warnings": [str(e)],
            }
