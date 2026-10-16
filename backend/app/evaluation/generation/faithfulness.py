import json
import re
from typing import List, Dict, Any, Optional, Sequence
from app.evaluation.generation.schemas import ClaimJudgment, FaithfulnessResult
from app.evaluation.generation.claim_extractor import claim_extractor
from app.rag.generation.registry import llm_registry
from app.rag.generation.base import GenerationRequest


JUDGE_SYSTEM_PROMPT = """You are a rigorous, unbiased factual verification judge in an educational RAG laboratory.
Your task is to evaluate individual factual claims made by a RAG system against the provided retrieved context.

For EACH claim, you must output a JSON object with:
1. "claim": The claim text being evaluated
2. "status": Strictly one of ["supported", "unsupported", "contradicted"]
   - "supported": The factual claim is explicitly stated or directly entailed by the context.
   - "unsupported": The claim is not mentioned, not provable, or extrapolated beyond the context.
   - "contradicted": The context explicitly states facts that directly conflict with or refute the claim.
3. "explanation": A concise, clear sentence justifying the verdict.
4. "supporting_passage_snippets": A brief excerpt (under 30 words) from the context supporting the claim if supported.

Return ONLY a valid JSON array of objects:
[
  {
    "claim": "...",
    "status": "supported",
    "explanation": "Context explicitly mentions ...",
    "supporting_passage_snippets": ["..."]
  }
]
"""


class FaithfulnessEvaluator:
    """
    Evaluates answer faithfulness via claim extraction and context entailment verification.
    Supports LLM-as-a-Judge (Gemini, OpenAI, Ollama) and deterministic rule-based verification for offline testing.
    """

    def __init__(self):
        self.extractor = claim_extractor

    def evaluate(
        self,
        generated_answer: str,
        context_passages: Sequence[str],
        judge_provider: Optional[str] = None,
        judge_model: Optional[str] = None,
    ) -> FaithfulnessResult:
        if not generated_answer or not generated_answer.strip():
            return FaithfulnessResult(
                score=0.0,
                total_claims=0,
                supported_claims=0,
                unsupported_claims=0,
                contradicted_claims=0,
                claims=[],
                evaluator_info={"evaluator": "empty_input", "mode": "none"},
            )

        # 1. Extract atomic claims
        raw_claims = self.extractor.extract_claims(generated_answer)

        # Handle explicit refusal ("insufficient evidence")
        if len(raw_claims) == 1 and raw_claims[0].get("is_refusal"):
            claim_text = raw_claims[0]["claim"]
            judgment = ClaimJudgment(
                claim=claim_text,
                citations=[],
                status="supported",
                explanation="Model correctly refused to answer due to insufficient or ungrounded evidence.",
                supporting_passage_snippets=[],
            )
            return FaithfulnessResult(
                score=1.0,
                total_claims=1,
                supported_claims=1,
                unsupported_claims=0,
                contradicted_claims=0,
                claims=[judgment],
                evaluator_info={"evaluator": "refusal_detector", "is_refusal": True},
            )

        if not raw_claims:
            return FaithfulnessResult(
                score=1.0,
                total_claims=0,
                supported_claims=0,
                unsupported_claims=0,
                contradicted_claims=0,
                claims=[],
                evaluator_info={"evaluator": "no_factual_claims", "policy": "neutral_pass"},
            )

        # 2. Evaluate claims via LLM Judge or Deterministic Rule Engine
        if judge_provider and judge_provider.lower() not in ("none", "heuristic", "rules"):
            try:
                return self._evaluate_with_llm_judge(
                    raw_claims=raw_claims,
                    context_passages=context_passages,
                    provider_id=judge_provider,
                    model_id=judge_model,
                )
            except Exception as e:
                # Graceful degradation to deterministic rule engine if LLM judge fails
                result = self._evaluate_with_rules(raw_claims, context_passages)
                result.evaluator_info["fallback_reason"] = f"LLM Judge error: {str(e)}"
                return result
        else:
            return self._evaluate_with_rules(raw_claims, context_passages)

    def _evaluate_with_rules(
        self,
        raw_claims: List[Dict[str, Any]],
        context_passages: Sequence[str],
    ) -> FaithfulnessResult:
        full_context_text = " ".join(context_passages).lower()
        judgments: List[ClaimJudgment] = []
        supported_count = 0
        unsupported_count = 0
        contradicted_count = 0

        # Stop words to filter out during token overlap computation
        stop_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "with",
            "is", "was", "are", "were", "been", "has", "have", "had", "this", "that",
            "these", "those", "it", "its", "of", "by", "as", "from", "during", "who", "which"
        }

        for item in raw_claims:
            claim_text = item["claim"]
            citations = item.get("citations", [])

            # Extract significant tokens
            words = [
                w.lower()
                for w in re.findall(r"\b[a-zA-Z0-9_\-\.]{3,}\b", claim_text)
                if w.lower() not in stop_words
            ]

            if not words:
                status = "supported"
                explanation = "Claim contains trivial non-factual wording."
                snippets = []
            else:
                matches = sum(1 for w in words if w in full_context_text)
                match_ratio = matches / len(words)

                # Find best matching passage snippet
                best_snippet = ""
                for p in context_passages:
                    p_lower = p.lower()
                    if sum(1 for w in words if w in p_lower) > 0:
                        best_snippet = p[:150] + "..." if len(p) > 150 else p
                        break

                if match_ratio >= 0.50:
                    status = "supported"
                    supported_count += 1
                    explanation = f"Supported: {int(match_ratio * 100)}% of key entities found in retrieved context."
                    snippets = [best_snippet] if best_snippet else []
                elif "never" in claim_text.lower() and "always" in full_context_text:
                    status = "contradicted"
                    contradicted_count += 1
                    explanation = "Contradicted: Claim directly inverts factual context."
                    snippets = [best_snippet] if best_snippet else []
                else:
                    status = "unsupported"
                    unsupported_count += 1
                    explanation = f"Unsupported: Only {int(match_ratio * 100)}% of entities verified in context."
                    snippets = []

            judgments.append(
                ClaimJudgment(
                    claim=claim_text,
                    citations=citations,
                    status=status,
                    explanation=explanation,
                    supporting_passage_snippets=snippets,
                )
            )

        total = len(judgments)
        score = supported_count / total if total > 0 else 1.0

        return FaithfulnessResult(
            score=round(score, 4),
            total_claims=total,
            supported_claims=supported_count,
            unsupported_claims=unsupported_count,
            contradicted_claims=contradicted_count,
            claims=judgments,
            evaluator_info={"evaluator": "deterministic_rule_engine", "mode": "token_overlap"},
        )

    def _evaluate_with_llm_judge(
        self,
        raw_claims: List[Dict[str, Any]],
        context_passages: Sequence[str],
        provider_id: str,
        model_id: Optional[str] = None,
    ) -> FaithfulnessResult:
        provider = llm_registry.get_provider(provider_id)
        if not provider.is_available():
            raise RuntimeError(f"Judge provider '{provider_id}' is currently unavailable.")

        claims_list_text = "\n".join(
            [f"- Claim {i+1}: {c['claim']}" for i, c in enumerate(raw_claims)]
        )
        context_formatted = "\n\n".join(
            [f"[Passage {i+1}]:\n{p}" for i, p in enumerate(context_passages)]
        )

        user_prompt = f"""Retrieved Context:
====================
{context_formatted}
====================

Claims to Evaluate:
{claims_list_text}

Respond strictly with a JSON array of judgments."""

        gen_req = GenerationRequest(
            prompt=user_prompt,
            system_prompt=JUDGE_SYSTEM_PROMPT,
            temperature=0.0,
            max_tokens=1024,
            metadata={"model": model_id} if model_id else {},
        )

        resp = provider.generate(gen_req)

        # Parse JSON output from LLM
        clean_json = resp.content.strip()
        if clean_json.startswith("```"):
            clean_json = re.sub(r"^```(?:json)?\s*", "", clean_json)
            clean_json = re.sub(r"\s*```$", "", clean_json)

        parsed_judgments = json.loads(clean_json)
        judgments: List[ClaimJudgment] = []
        supported_count = 0
        unsupported_count = 0
        contradicted_count = 0

        for i, item in enumerate(parsed_judgments):
            status = item.get("status", "unsupported").lower()
            if status == "supported":
                supported_count += 1
            elif status == "contradicted":
                contradicted_count += 1
            else:
                unsupported_count += 1
                status = "unsupported"

            orig_citations = raw_claims[i].get("citations", []) if i < len(raw_claims) else []

            judgments.append(
                ClaimJudgment(
                    claim=item.get("claim", raw_claims[i]["claim"] if i < len(raw_claims) else ""),
                    citations=orig_citations,
                    status=status,
                    explanation=item.get("explanation", "Evaluated by LLM Judge"),
                    supporting_passage_snippets=item.get("supporting_passage_snippets", []),
                )
            )

        total = len(judgments)
        score = supported_count / total if total > 0 else 1.0

        return FaithfulnessResult(
            score=round(score, 4),
            total_claims=total,
            supported_claims=supported_count,
            unsupported_claims=unsupported_count,
            contradicted_claims=contradicted_count,
            claims=judgments,
            evaluator_info={
                "evaluator": f"llm_judge:{provider_id}",
                "model": resp.model_id,
                "latency_ms": resp.generation_latency_ms,
            },
        )


faithfulness_evaluator = FaithfulnessEvaluator()
