import json
import time
from typing import List, Dict, Any, Optional, AsyncIterator
import httpx

from app.core.config import settings
from app.rag.generation.base import (
    BaseLLMProvider,
    GenerationRequest,
    GenerationResponse,
    LLMModelInfo,
    StreamChunk,
)


DEFAULT_OLLAMA_MODEL = getattr(settings, "OLLAMA_DEFAULT_MODEL", "qwen2.5:3b")


class OllamaProvider(BaseLLMProvider):
    """
    Local-first LLM provider using Ollama HTTP API.
    Designed for local CPU/GPU inference without third-party API keys.
    Default recommended model: qwen2.5:3b (approx 2.0 GB RAM, fast CPU inference).
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: Optional[int] = None,
        default_model: Optional[str] = None,
    ):
        self._base_url = (base_url or getattr(settings, "OLLAMA_BASE_URL", "http://host.docker.internal:11434")).rstrip("/")
        self._timeout = timeout or getattr(settings, "OLLAMA_TIMEOUT", 120)
        self._default_model = default_model or DEFAULT_OLLAMA_MODEL

    @property
    def provider_id(self) -> str:
        return "ollama"

    @property
    def display_name(self) -> str:
        return "Ollama (Local CPU/GPU)"

    @property
    def base_url(self) -> str:
        return self._base_url

    def is_available(self) -> bool:
        """Checks if Ollama server is reachable."""
        try:
            with httpx.Client(timeout=3.0) as client:
                res = client.get(f"{self._base_url}/api/tags")
                return res.status_code == 200
        except Exception:
            return False

    def list_models(self) -> List[LLMModelInfo]:
        """Lists models currently installed and pulled in local Ollama."""
        try:
            with httpx.Client(timeout=5.0) as client:
                res = client.get(f"{self._base_url}/api/tags")
                if res.status_code != 200:
                    return []
                data = res.json()
                models = []
                for m in data.get("models", []):
                    name = m.get("name", "")
                    details = m.get("details", {}) or {}
                    param_size = details.get("parameter_size", "Unknown")
                    quant = details.get("quantization_level", "Unknown")
                    models.append(
                        LLMModelInfo(
                            model_id=name,
                            display_name=f"{name} ({param_size}, {quant})",
                            provider_id="ollama",
                            context_window=4096,
                            parameter_count=param_size,
                            quantization=quant,
                            supports_streaming=True,
                            description=f"Local Ollama model. Quantization: {quant}, Params: {param_size}.",
                            is_available=True,
                            metadata=m,
                        )
                    )
                return models
        except Exception:
            return []

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Executes a synchronous completion via Ollama."""
        target_model = request.metadata.get("model") or self._default_model
        payload = {
            "model": target_model,
            "prompt": request.prompt,
            "system": request.system_prompt or "",
            "stream": False,
            "options": {
                "temperature": request.temperature,
                "num_predict": request.max_tokens,
            },
        }

        start_time = time.perf_counter()
        try:
            with httpx.Client(timeout=float(self._timeout)) as client:
                res = client.post(f"{self._base_url}/api/generate", json=payload)
                if res.status_code == 404:
                    raise ValueError(
                        f"Ollama model '{target_model}' is not pulled. "
                        f"Run 'ollama pull {target_model}' in your terminal to install it."
                    )
                res.raise_for_status()
                data = res.json()
        except httpx.ConnectError:
            raise ConnectionError(
                f"Could not connect to Ollama at {self._base_url}. "
                "Ensure Ollama is installed and running (`ollama serve`). "
                "On Fedora host: run `curl -fsSL https://ollama.com/install.sh | sh && ollama serve`."
            )
        except httpx.TimeoutException:
            raise TimeoutError(
                f"Ollama generation timed out after {self._timeout}s. "
                "Consider reducing max_tokens or running on a smaller quantized model."
            )

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        content = data.get("response", "")
        prompt_tokens = data.get("prompt_eval_count")
        completion_tokens = data.get("eval_count")
        total_tokens = (prompt_tokens + completion_tokens) if (prompt_tokens is not None and completion_tokens is not None) else None

        # Durations in nanoseconds
        load_duration_ns = data.get("load_duration", 0)
        load_latency_ms = (load_duration_ns / 1_000_000.0) if load_duration_ns else None

        return GenerationResponse(
            content=content,
            model_id=target_model,
            provider_id="ollama",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_ms=round(latency_ms, 2),
            model_load_latency_ms=round(load_latency_ms, 2) if load_latency_ms else None,
            finish_reason="stop" if data.get("done") else "length",
            metadata={
                "eval_duration_ms": round(data.get("eval_duration", 0) / 1_000_000.0, 2),
                "prompt_eval_duration_ms": round(data.get("prompt_eval_duration", 0) / 1_000_000.0, 2),
            },
        )

    async def generate_stream(self, request: GenerationRequest) -> AsyncIterator[StreamChunk]:
        """Asynchronously streams completion tokens via Ollama NDJSON chunks."""
        target_model = request.metadata.get("model") or self._default_model
        payload = {
            "model": target_model,
            "prompt": request.prompt,
            "system": request.system_prompt or "",
            "stream": True,
            "options": {
                "temperature": request.temperature,
                "num_predict": request.max_tokens,
            },
        }

        try:
            async with httpx.AsyncClient(timeout=float(self._timeout)) as client:
                async with client.stream("POST", f"{self._base_url}/api/generate", json=payload) as response:
                    if response.status_code == 404:
                        raise ValueError(
                            f"Ollama model '{target_model}' not installed. Run 'ollama pull {target_model}'."
                        )
                    response.raise_for_status()

                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        chunk_data = json.loads(line)
                        text_piece = chunk_data.get("response", "")
                        is_done = chunk_data.get("done", False)

                        yield StreamChunk(
                            content=text_piece,
                            is_final=is_done,
                            token_count=chunk_data.get("eval_count"),
                            finish_reason="stop" if is_done else None,
                        )
        except httpx.ConnectError:
            raise ConnectionError(f"Cannot reach Ollama at {self._base_url}. Ensure `ollama serve` is active.")
