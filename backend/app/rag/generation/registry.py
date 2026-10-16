from typing import Dict, List, Optional
from app.rag.generation.base import BaseLLMProvider
from app.rag.generation.schemas import ProviderInfoResponse, ModelInfoResponse
from app.rag.generation.providers.ollama_provider import OllamaProvider
from app.rag.generation.providers.openai_provider import OpenAIProvider
from app.rag.generation.providers.gemini_provider import GeminiProvider
from app.rag.generation.providers.mock_provider import MockLLMProvider


class LLMProviderRegistry:
    """
    Registry for LLM providers and models.
    Supports local Ollama, cloud OpenAI/Gemini, and deterministic test mocks.
    """

    def __init__(self):
        self._providers: Dict[str, BaseLLMProvider] = {}
        self._register_default_providers()

    def _register_default_providers(self):
        self.register(OllamaProvider())
        self.register(OpenAIProvider())
        self.register(GeminiProvider())
        self.register(MockLLMProvider())

    def register(self, provider: BaseLLMProvider) -> None:
        self._providers[provider.provider_id.lower()] = provider

    def get_provider(self, provider_id: str) -> BaseLLMProvider:
        pid = provider_id.lower()
        if pid not in self._providers:
            available = ", ".join(self._providers.keys())
            raise ValueError(f"Unknown LLM provider '{provider_id}'. Available providers: {available}")
        return self._providers[pid]

    def list_providers(self) -> List[ProviderInfoResponse]:
        results = []
        for pid, provider in self._providers.items():
            is_avail = provider.is_available()
            if pid == "ollama":
                status = "Ready (Ollama service reachable)" if is_avail else "Offline (Ollama server not responding at configured URL)"
                default_model = "qwen2.5:3b"
                models = [m.model_id for m in provider.list_models()]
            elif pid == "openai":
                status = "Configured (API key present)" if is_avail else "Not configured (Set OPENAI_API_KEY)"
                default_model = "gpt-4o-mini"
                models = ["gpt-4o-mini", "gpt-4o"]
            elif pid == "gemini":
                status = "Configured (API key present)" if is_avail else "Not configured (Set GEMINI_API_KEY)"
                default_model = "gemini-3.8-flash"
                models = ["gemini-3.8-flash", "gemini-1.5-flash"]
            elif pid == "mock":
                status = "Active (Automated test fixture)"
                default_model = "mock-instruct-v1"
                models = ["mock-instruct-v1"]
            else:
                status = "Unknown"
                default_model = ""
                models = []

            results.append(
                ProviderInfoResponse(
                    provider_id=pid,
                    display_name=provider.display_name,
                    is_available=is_avail,
                    status_message=status,
                    default_model=default_model,
                    supported_models=models,
                )
            )
        return results

    def list_all_models(self) -> List[ModelInfoResponse]:
        models = []
        for provider in self._providers.values():
            for m in provider.list_models():
                models.append(
                    ModelInfoResponse(
                        model_id=m.model_id,
                        display_name=m.display_name,
                        provider_id=m.provider_id,
                        context_window=m.context_window,
                        parameter_count=m.parameter_count,
                        quantization=m.quantization,
                        supports_streaming=m.supports_streaming,
                        is_available=m.is_available,
                        description=m.description,
                    )
                )
        return models


llm_registry = LLMProviderRegistry()
