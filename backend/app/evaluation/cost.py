from typing import Dict, Any, Optional
from pydantic import BaseModel, Field


class ModelPricingSpec(BaseModel):
    """
    Pricing specification per 1,000,000 tokens for a specific model with versioning.
    """
    model_id: str
    provider_id: str
    input_cost_per_1m_usd: float
    output_cost_per_1m_usd: float
    currency: str = "USD"
    pricing_version: str = "2026-Q4"
    effective_date: str = "2026-01-01"
    pricing_source: str
    is_free_local: bool = False
    notes: Optional[str] = None


class CostCalculationResult(BaseModel):
    """
    Detailed monetary and token cost breakdown for a RAG execution.
    """
    input_tokens: int = 0
    output_tokens: int = 0
    context_tokens: int = 0
    judge_tokens: int = 0
    total_tokens: int = 0
    estimated_input_cost_usd: float = 0.0
    estimated_output_cost_usd: float = 0.0
    estimated_judge_cost_usd: float = 0.0
    total_estimated_cost_usd: Optional[float] = 0.0
    cost_display: str = "$0.000000"
    currency: str = "USD"
    is_estimated: bool = True
    is_unknown: bool = False
    pricing_available: bool = True
    pricing_source: str = "registry"
    pricing_version: str = "2026-Q4"
    notes: Optional[str] = None


class ModelPricingRegistry:
    """
    Configurable, versioned registry of LLM pricing for estimating experimental RAG budgets.
    """

    def __init__(self, default_version: str = "2026-Q4"):
        self.default_version = default_version
        self._pricing: Dict[str, ModelPricingSpec] = {
            # OpenAI Models
            "gpt-4o-mini": ModelPricingSpec(
                model_id="gpt-4o-mini",
                provider_id="openai",
                input_cost_per_1m_usd=0.15,
                output_cost_per_1m_usd=0.60,
                pricing_version=self.default_version,
                pricing_source="OpenAI Official Pricing (2026)",
            ),
            "gpt-4o": ModelPricingSpec(
                model_id="gpt-4o",
                provider_id="openai",
                input_cost_per_1m_usd=2.50,
                output_cost_per_1m_usd=10.00,
                pricing_version=self.default_version,
                pricing_source="OpenAI Official Pricing (2026)",
            ),
            # Google Gemini Models
            "gemini-1.5-flash": ModelPricingSpec(
                model_id="gemini-1.5-flash",
                provider_id="gemini",
                input_cost_per_1m_usd=0.075,
                output_cost_per_1m_usd=0.30,
                pricing_version=self.default_version,
                pricing_source="Google Cloud AI Pricing (2026)",
            ),
            "gemini-2.0-flash": ModelPricingSpec(
                model_id="gemini-2.0-flash",
                provider_id="gemini",
                input_cost_per_1m_usd=0.10,
                output_cost_per_1m_usd=0.40,
                pricing_version=self.default_version,
                pricing_source="Google Cloud AI Pricing (2026)",
            ),
            "gemini-3.8-flash": ModelPricingSpec(
                model_id="gemini-3.8-flash",
                provider_id="gemini",
                input_cost_per_1m_usd=0.10,
                output_cost_per_1m_usd=0.40,
                pricing_version=self.default_version,
                pricing_source="Google Gemini Experimental Pricing (2026)",
            ),
            # Local Ollama (Free API, local electricity/compute cost noted)
            "qwen2.5:3b": ModelPricingSpec(
                model_id="qwen2.5:3b",
                provider_id="ollama",
                input_cost_per_1m_usd=0.0,
                output_cost_per_1m_usd=0.0,
                is_free_local=True,
                pricing_version=self.default_version,
                pricing_source="Local Inference (0 USD API cost; hardware/electricity compute applies)",
                notes="Self-hosted local compute",
            ),
            # Automated Mock Provider
            "mock-instruct-v1": ModelPricingSpec(
                model_id="mock-instruct-v1",
                provider_id="mock",
                input_cost_per_1m_usd=0.0,
                output_cost_per_1m_usd=0.0,
                is_free_local=True,
                pricing_version=self.default_version,
                pricing_source="In-Memory Test Mock (0 USD)",
            ),
        }

    def get_pricing(self, model_id: str) -> Optional[ModelPricingSpec]:
        mid = model_id.lower()
        if mid in self._pricing:
            return self._pricing[mid]
        # Match by prefix (e.g. "gpt-4o-mini-2024-07-18")
        for key, spec in self._pricing.items():
            if mid.startswith(key):
                return spec
        return None

    def register_pricing(self, spec: ModelPricingSpec) -> None:
        """Dynamically add or override a model pricing spec."""
        self._pricing[spec.model_id.lower()] = spec

    def calculate_cost(
        self,
        model_id: str,
        input_tokens: int,
        output_tokens: int,
        context_tokens: int = 0,
        judge_tokens: int = 0,
        judge_model_id: Optional[str] = None,
        is_provider_reported: bool = False,
    ) -> CostCalculationResult:
        total_tokens = input_tokens + output_tokens + judge_tokens
        spec = self.get_pricing(model_id)

        if not spec:
            return CostCalculationResult(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                context_tokens=context_tokens,
                judge_tokens=judge_tokens,
                total_tokens=total_tokens,
                total_estimated_cost_usd=None,
                cost_display="unknown",
                is_unknown=True,
                pricing_available=False,
                pricing_source="unknown_model",
                notes=f"Pricing rates not registered for model '{model_id}'.",
            )

        in_cost = (input_tokens / 1_000_000.0) * spec.input_cost_per_1m_usd
        out_cost = (output_tokens / 1_000_000.0) * spec.output_cost_per_1m_usd

        judge_cost = 0.0
        if judge_tokens > 0 and judge_model_id:
            judge_spec = self.get_pricing(judge_model_id)
            if judge_spec:
                # Assume judge tokens roughly 70% input, 30% output
                j_in = int(judge_tokens * 0.7)
                j_out = int(judge_tokens * 0.3)
                judge_cost = (
                    (j_in / 1_000_000.0) * judge_spec.input_cost_per_1m_usd
                    + (j_out / 1_000_000.0) * judge_spec.output_cost_per_1m_usd
                )

        total_cost = in_cost + out_cost + judge_cost
        cost_formatted = f"${total_cost:.6f}"

        return CostCalculationResult(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            context_tokens=context_tokens,
            judge_tokens=judge_tokens,
            total_tokens=total_tokens,
            estimated_input_cost_usd=round(in_cost, 6),
            estimated_output_cost_usd=round(out_cost, 6),
            estimated_judge_cost_usd=round(judge_cost, 6),
            total_estimated_cost_usd=round(total_cost, 6),
            cost_display=cost_formatted,
            currency=spec.currency,
            is_estimated=not is_provider_reported,
            is_unknown=False,
            pricing_available=True,
            pricing_source=spec.pricing_source,
            pricing_version=spec.pricing_version,
            notes=spec.notes,
        )


pricing_registry = ModelPricingRegistry()
