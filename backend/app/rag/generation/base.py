from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, AsyncIterator


@dataclass
class LLMModelInfo:
    model_id: str
    display_name: str
    provider_id: str
    context_window: int = 4096
    parameter_count: str = "Unknown"
    quantization: str = "Unknown"
    supports_streaming: bool = True
    description: str = ""
    is_available: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationRequest:
    prompt: str
    system_prompt: Optional[str] = None
    temperature: float = 0.0
    max_tokens: int = 512
    stop_sequences: List[str] = field(default_factory=list)
    stream: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationResponse:
    content: str
    model_id: str
    provider_id: str
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    latency_ms: float = 0.0
    finish_reason: str = "stop"
    model_load_latency_ms: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StreamChunk:
    content: str
    is_final: bool = False
    token_count: Optional[int] = None
    finish_reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseLLMProvider(ABC):
    """
    Abstract Base Class for all LLM providers (Ollama, OpenAI, Gemini, Mock).
    Provides a standardized interface for capability inspection, synchronous generation,
    and asynchronous token streaming.
    """

    @property
    @abstractmethod
    def provider_id(self) -> str:
        """Unique identifier for this provider (e.g. 'ollama', 'openai', 'gemini')."""
        pass

    @property
    @abstractmethod
    def display_name(self) -> str:
        """Human-readable provider label."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Verifies network reachability and authorization."""
        pass

    @abstractmethod
    def list_models(self) -> List[LLMModelInfo]:
        """Lists available models from this provider."""
        pass

    @abstractmethod
    def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Executes a synchronous LLM generation."""
        pass

    @abstractmethod
    def generate_stream(self, request: GenerationRequest) -> AsyncIterator[StreamChunk]:
        """Streams generation tokens asynchronously as they become available."""
        pass
