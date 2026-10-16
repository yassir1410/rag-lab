import re
from typing import Optional, Dict, Any
from app.evaluation.generation.schemas import AnswerRelevanceResult
from app.evaluation.generation.claim_extractor import claim_extractor


class AnswerRelevanceEvaluator:
    """
    Evaluates whether the generated response directly answers the user's question,
    assesses semantic completeness relative to reference answers, and validates
    proper refusal handling for unanswerable questions.
    """

    def __init__(self):
        self.extractor = claim_extractor

    def evaluate(
        self,
        question: str,
        generated_answer: str,
        reference_answer: Optional[str] = None,
        is_answerable: bool = True,
    ) -> AnswerRelevanceResult:
        if not generated_answer or not generated_answer.strip():
            return AnswerRelevanceResult(
                score=0.0,
                is_refusal=False,
                is_correct_refusal=False,
                completeness_score=0.0,
                relevance_score=0.0,
                explanation="Generated answer is completely empty.",
            )

        is_refusal = self.extractor.is_refusal(generated_answer)

        # Case 1: Unanswerable Question
        if not is_answerable:
            if is_refusal:
                return AnswerRelevanceResult(
                    score=1.0,
                    is_refusal=True,
                    is_correct_refusal=True,
                    completeness_score=1.0,
                    relevance_score=1.0,
                    explanation="Correct refusal: Model appropriately stated insufficient evidence for an unanswerable question.",
                )
            else:
                return AnswerRelevanceResult(
                    score=0.1,
                    is_refusal=False,
                    is_correct_refusal=False,
                    completeness_score=0.0,
                    relevance_score=0.2,
                    explanation="Hallucination/Failure: Model attempted to answer an unanswerable question without sufficient grounded evidence.",
                )

        # Case 2: Answerable Question, but model refused
        if is_refusal:
            return AnswerRelevanceResult(
                score=0.3,
                is_refusal=True,
                is_correct_refusal=False,
                completeness_score=0.0,
                relevance_score=0.4,
                explanation="False refusal: Model refused to answer even though the question was answerable from evidence.",
            )

        # Case 3: Standard Answerable Question
        stop_words = {
            "the", "a", "an", "and", "or", "in", "on", "at", "to", "for", "with",
            "is", "was", "are", "what", "which", "who", "where", "how", "does", "did"
        }

        # 1. Question terms overlap
        q_tokens = [
            w.lower()
            for w in re.findall(r"\b[a-zA-Z0-9_\-\.]{3,}\b", question)
            if w.lower() not in stop_words
        ]
        ans_lower = generated_answer.lower()
        q_matches = sum(1 for w in q_tokens if w in ans_lower) if q_tokens else 1
        relevance_score = min(1.0, q_matches / len(q_tokens)) if q_tokens else 0.8

        # 2. Reference answer completeness
        completeness_score = 0.8
        if reference_answer and reference_answer.strip():
            ref_tokens = [
                w.lower()
                for w in re.findall(r"\b[a-zA-Z0-9_\-\.]{3,}\b", reference_answer)
                if w.lower() not in stop_words
            ]
            if ref_tokens:
                ref_matches = sum(1 for w in ref_tokens if w in ans_lower)
                completeness_score = min(1.0, ref_matches / len(ref_tokens))

        if reference_answer and reference_answer.strip():
            overall_score = round(0.7 * completeness_score + 0.3 * max(relevance_score, completeness_score * 0.8), 4)
        else:
            overall_score = round(relevance_score, 4)

        return AnswerRelevanceResult(
            score=overall_score,
            is_refusal=False,
            is_correct_refusal=False,
            completeness_score=round(completeness_score, 4),
            relevance_score=round(relevance_score, 4),
            explanation=f"Relevance: {int(relevance_score * 100)}% question entity overlap, {int(completeness_score * 100)}% reference completeness.",
        )


answer_relevance_evaluator = AnswerRelevanceEvaluator()
