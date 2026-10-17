import os
import base64
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import httpx
from app.core.config import settings

logger = logging.getLogger(__name__)


class VisionProviderError(Exception):
    """Raised when multimodal vision inference fails upstream."""

    def __init__(self, message: str, status_code: int = 502):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


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
        if "weather" in q_lower or "vacation" in q_lower:
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
    """Google Gemini multimodal vision provider supporting inline base64 image parts."""

    ALLOWED_IMAGE_MIMES = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }
    MAX_FILE_BYTES = 15 * 1024 * 1024  # 15 MB per image
    MAX_IMAGES = 2

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.api_key = api_key if api_key is not None else (getattr(settings, "GEMINI_API_KEY", None) or os.getenv("GEMINI_API_KEY"))
        self.model = model or getattr(settings, "GEMINI_MODEL", None) or os.getenv("GEMINI_MODEL") or "gemini-3.8-flash"
        self._base_url = "https://generativelanguage.googleapis.com/v1beta"

    def is_available(self) -> bool:
        return bool(self.api_key and len(self.api_key.strip()) > 5)

    def generate_answer(
        self,
        question: str,
        context_text: str,
        image_paths: List[str],
    ) -> Dict[str, Any]:
        if not self.is_available():
            raise VisionProviderError(
                "Gemini vision capability unavailable: GEMINI_API_KEY is not configured.",
                status_code=503,
            )

        prompt = (
            f"You are a multimodal RAG document intelligence assistant. "
            f"Answer the user's question using ONLY the provided document evidence and attached images.\n"
            f"Cite your sources using bracketed tags like [PAGE_X], [IMG_PX], or [TBL_PX] matching the evidence provided.\n"
            f"If the answer cannot be determined from the evidence or images, explicitly state that the document does not contain sufficient information.\n\n"
            f"DOCUMENT EVIDENCE:\n{context_text}\n\n"
            f"QUESTION:\n{question}\n\n"
            f"ANSWER:"
        )

        parts: List[Dict[str, Any]] = [{"text": prompt}]
        images_sent_count = 0
        allowed_dir = os.path.realpath(str(getattr(settings, "ABSOLUTE_STORAGE_DIR", "/app/data/uploads")))

        # Validate and attach image bytes
        for p in image_paths[: self.MAX_IMAGES]:
            if not p:
                continue

            real_path = os.path.realpath(p)
            # Security: verify path is within authorized storage directory
            if not (real_path.startswith(allowed_dir) or real_path.startswith("/app/data")):
                logger.warning(f"Rejected attachment outside storage directory: {real_path}")
                continue

            if not os.path.exists(real_path) or not os.path.isfile(real_path):
                logger.warning(f"Image attachment path does not exist: {real_path}")
                continue

            # Verify file size
            if os.path.getsize(real_path) > self.MAX_FILE_BYTES:
                logger.warning(f"Image attachment exceeds max size: {real_path}")
                continue

            # Verify MIME type
            ext = os.path.splitext(real_path)[1].lower()
            mime_type = self.ALLOWED_IMAGE_MIMES.get(ext, "image/png")

            try:
                with open(real_path, "rb") as img_file:
                    encoded_bytes = base64.b64encode(img_file.read()).decode("utf-8")
                    parts.append({
                        "inline_data": {
                            "mime_type": mime_type,
                            "data": encoded_bytes,
                        }
                    })
                    images_sent_count += 1
            except Exception as read_err:
                logger.error(f"Failed to read image bytes for {real_path}: {read_err}")

        url = f"{self._base_url}/models/{self.model}:generateContent"
        headers = {
            "x-goog-api-key": self.api_key,
            "Content-Type": "application/json",
        }
        payload = {
            "contents": [{"parts": parts}],
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": 2048,
            },
        }

        try:
            with httpx.Client(timeout=45.0) as client:
                res = client.post(url, headers=headers, json=payload)
                if res.status_code == 200:
                    data = res.json()
                    candidates = data.get("candidates", [])
                    if not candidates:
                        raise VisionProviderError("Gemini returned empty candidates list.", status_code=502)
                    candidate = candidates[0]
                    text = candidate.get("content", {}).get("parts", [{}])[0].get("text", "")
                    if not text:
                        finish_reason = candidate.get("finishReason", "UNKNOWN")
                        raise VisionProviderError(f"Gemini returned empty text (finishReason: {finish_reason}).", status_code=502)

                    return {
                        "answer": text.strip(),
                        "model": self.model,
                        "is_vision_enabled": images_sent_count > 0,
                        "warnings": [],
                    }
                else:
                    try:
                        err_json = res.json().get("error", {})
                        err_msg = err_json.get("message", res.text)
                    except Exception:
                        err_msg = res.text

                    # Strict error classification - NEVER return API error as generated answer
                    if res.status_code == 404:
                        raise VisionProviderError(
                            f"Gemini model '{self.model}' not found for API version v1beta: {err_msg}",
                            status_code=404,
                        )
                    elif res.status_code in (401, 403):
                        raise VisionProviderError(
                            f"Gemini API authentication failed: {err_msg}",
                            status_code=401,
                        )
                    elif res.status_code == 429:
                        raise VisionProviderError(
                            f"Gemini API rate limit exceeded: {err_msg}",
                            status_code=429,
                        )
                    elif res.status_code >= 500:
                        raise VisionProviderError(
                            f"Gemini upstream service error ({res.status_code}): {err_msg}",
                            status_code=502,
                        )
                    else:
                        raise VisionProviderError(
                            f"Gemini API error ({res.status_code}): {err_msg}",
                            status_code=res.status_code if 400 <= res.status_code < 600 else 502,
                        )
        except VisionProviderError:
            raise
        except httpx.TimeoutException as te:
            raise VisionProviderError(f"Gemini API request timed out: {te}", status_code=504)
        except httpx.HTTPStatusError as hse:
            code = hse.response.status_code if hse.response else 502
            raise VisionProviderError(f"Gemini API HTTP error ({code}): {hse}", status_code=code)
        except Exception as e:
            raise VisionProviderError(f"Multimodal inference failed: {str(e)}", status_code=502)
