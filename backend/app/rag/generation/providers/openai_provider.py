import json
import time
from typing import List, Optional, AsyncIterator
import httpx

from app.core.config import settings
from app.rag.generation.base import (
    BaseLLMProvider,
    GenerationRequest,
    GenerationResponse,
    LLMModelInfo,
    StreamChunk,
)


SUPPORTED_OPENAI_MODELS = [
    LLMModelInfo(
        model_id="gpt-4o-mini",
        display_name="GPT-4o Mini (Cloud)",
        provider_id="openai",
        context_window=128000,
        parameter_count="Proprietary",
        quantization="FP16 / Server",
        supports_streaming=True,
        description="Fast, cost-effective OpenAI cloud model.",
    ),
    LLMModelInfo(
        model_id="gpt-4o",
        display_name="GPT-4o (Cloud High-Capacity)",
        provider_id="openai",
        context_window=128000,
        parameter_count="Proprietary",
        quantization="FP16 / Server",
        supports_streaming=True,
        description="Flagship OpenAI cloud multimodal model.",
    ),
]


class OpenAIProvider(BaseLLMProvider):
    """
    OpenAI API LLM Provider (Optional cloud provider).
    Requires OPENAI_API_KEY environment variable.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self._api_key = api_key or getattr(settings, "OPENAI_API_KEY", None)
        self._base_url = (base_url or getattr(settings, "OPENAI_API_BASE", "https://api.openai.com/v1")).rstrip("/")

    @property
    def provider_id(self) -> str:
        return "openai"

    @property
    def display_name(self) -> str:
        return "OpenAI API (Cloud)"

    def is_available(self) -> bool:
        return bool(self._api_key and len(self._api_key.strip()) > 5)

    def list_models(self) -> List[LLMModelInfo]:
        return SUPPORTED_OPENAI_MODELS

    def _build_messages(self, request: GenerationRequest) -> List[dict]:
        messages = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.prompt})
        return messages

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        if not self.is_available():
            raise ValueError("OpenAI API key is missing. Set OPENAI_API_KEY in your environment.")

        target_model = request.metadata.get("model") or "gpt-4o-mini"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": target_model,
            "messages": self._build_messages(request),
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "stream": False,
        }

        start_time = time.perf_counter()
        with httpx.Client(timeout=60.0) as client:
            res = client.post(f"{self._base_url}/chat/completions", headers=headers, json=payload)
            if res.status_code != 200:
                try:
                    err_json = res.json().get("error", {})
                    err_msg = err_json.get("message", res.text)
                    err_code = err_json.get("code") or res.status_code
                    raise ValueError(f"OpenAI API Error ({err_code}): {err_msg}")
                except (ValueError, Exception) as parse_err:
                    if isinstance(parse_err, ValueError):
                        raise
                    res.raise_for_status()
            data = res.json()

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        choice = data["choices"][0]
        content = choice["message"]["content"]
        usage = data.get("usage", {})

        return GenerationResponse(
            content=content,
            model_id=target_model,
            provider_id="openai",
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            total_tokens=usage.get("total_tokens"),
            latency_ms=round(latency_ms, 2),
            finish_reason=choice.get("finish_reason", "stop"),
        )

    async def generate_stream(self, request: GenerationRequest) -> AsyncIterator[StreamChunk]:
        if not self.is_available():
            raise ValueError("OpenAI API key is missing.")

        target_model = request.metadata.get("model") or "gpt-4o-mini"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": target_model,
            "messages": self._build_messages(request),
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "stream": True,
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream("POST", f"{self._base_url}/chat/completions", headers=headers, json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    data_str = line[6:].strip()
                    if data_str == "[DONE]":
                        yield StreamChunk(content="", is_final=True, finish_reason="stop")
                        break
                    chunk_json = json.loads(data_str)
                    delta = chunk_json["choices"][0].get("delta", {})
                    content_piece = delta.get("content", "")
                    if content_piece:
                        yield StreamChunk(content=content_piece, is_final=False)
