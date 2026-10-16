from app.rag.chunking.tokenizer import default_tokenizer
from app.rag.generation.schemas import TokenAllocationMetrics


class TokenBudgetManager:
    """
    Manages and allocates token capacity across prompt components:
    1. System instructions
    2. User question
    3. Retrieved document context
    4. Reserved output generation budget
    """

    def __init__(self):
        self.tokenizer = default_tokenizer

    def count_tokens(self, text: str) -> int:
        return self.tokenizer.count_tokens(text)

    def allocate(
        self,
        system_prompt: str,
        question: str,
        context_text: str,
        max_output_tokens: int = 512,
        model_context_window: int = 4096,
        requested_max_context_tokens: int = 2048,
    ) -> TokenAllocationMetrics:
        sys_tokens = self.count_tokens(system_prompt)
        q_tokens = self.count_tokens(question)
        ctx_tokens = self.count_tokens(context_text)

        total_prompt = sys_tokens + q_tokens + ctx_tokens
        remaining = model_context_window - (total_prompt + max_output_tokens)

        tokenizer_name = "tiktoken (cl100k_base)" if self.tokenizer.is_tiktoken_available else "regex_subword_heuristic"
        is_approx = not self.tokenizer.is_tiktoken_available

        return TokenAllocationMetrics(
            system_prompt_tokens=sys_tokens,
            user_question_tokens=q_tokens,
            context_tokens=ctx_tokens,
            reserved_output_tokens=max_output_tokens,
            total_prompt_tokens=total_prompt,
            model_context_window=model_context_window,
            remaining_capacity=max(0, remaining),
            tokenizer_name=tokenizer_name,
            is_approximate=is_approx,
        )


token_budget_manager = TokenBudgetManager()
