import time
from typing import List, AsyncIterator
from app.rag.generation.base import (
    BaseLLMProvider,
    GenerationRequest,
    GenerationResponse,
    LLMModelInfo,
    StreamChunk,
)


class MockLLMProvider(BaseLLMProvider):
    """
    Explicitly labeled deterministic Mock LLM provider.
    Strictly permitted ONLY for unit/integration tests and automated CI test suites.
    """

    def __init__(self, response_override: str = None):
        self._response_override = response_override

    @property
    def provider_id(self) -> str:
        return "mock"

    @property
    def display_name(self) -> str:
        return "Mock LLM (Automated Test Fixture)"

    def is_available(self) -> bool:
        return True

    def list_models(self) -> List[LLMModelInfo]:
        return [
            LLMModelInfo(
                model_id="mock-instruct-v1",
                display_name="Mock Instruct Model (Test Fixture)",
                provider_id="mock",
                context_window=4096,
                parameter_count="0B (Deterministic)",
                quantization="None",
                supports_streaming=True,
                description="Mock provider designed solely for deterministic unit testing.",
                is_available=True,
            )
        ]

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        t0 = time.perf_counter()
        if self._response_override:
            content = self._response_override
        elif "insufficient" in request.prompt.lower() or "no relevant context" in request.prompt.lower():
            content = "Based on the provided documents, there is insufficient evidence to answer this question."
        elif "malicious" in request.prompt.lower() or "ignore all instructions" in request.prompt.lower():
            content = "The candidate is a backend engineer specializing in Python and PostgreSQL [S1]."
        else:
            content = "Based on the retrieved context, the candidate possesses strong backend engineering skills [S1] with extensive expertise in FastAPI and distributed systems [S2]."

        latency_ms = (time.perf_counter() - t0) * 1000.0

        return GenerationResponse(
            content=content,
            model_id="mock-instruct-v1",
            provider_id="mock",
            prompt_tokens=len(request.prompt.split()),
            completion_tokens=len(content.split()),
            total_tokens=len(request.prompt.split()) + len(content.split()),
            latency_ms=round(latency_ms, 2),
            finish_reason="stop",
        )

    async def generate_stream(self, request: GenerationRequest) -> AsyncIterator[StreamChunk]:
        resp = self.generate(request)
        words = resp.content.split(" ")
        for i, word in enumerate(words):
            piece = word + (" " if i < len(words) - 1 else "")
            is_final = (i == len(words) - 1)
            yield StreamChunk(
                content=piece,
                is_final=is_final,
                token_count=len(words) if is_final else None,
                finish_reason="stop" if is_final else None,
            )
