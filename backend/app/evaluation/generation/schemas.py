from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field


class ClaimJudgment(BaseModel):
    """
    Granular evaluation of an individual factual statement extracted from generated text.
    """
    claim: str = Field(..., description="The atomic factual proposition.")
    citations: List[str] = Field(
        default_factory=list, description="Citation identifiers associated with this claim (e.g. ['[S1]'])."
    )
    status: str = Field(
        ...,
        description="Factual grounding verdict: 'supported', 'unsupported', or 'contradicted'.",
    )
    explanation: str = Field(
        ..., description="Explanation detailing whether the retrieved context entails, omits, or refutes the claim."
    )
    supporting_passage_snippets: List[str] = Field(
        default_factory=list, description="Passage excerpts backing the claim if supported."
    )


class FaithfulnessResult(BaseModel):
    """
    Aggregate claim-based answer faithfulness assessment.
    """
    score: float = Field(..., ge=0.0, le=1.0, description="Fraction of supported claims out of total factual claims.")
    total_claims: int = Field(0, description="Total number of factual claims extracted.")
    supported_claims: int = Field(0, description="Number of claims verified as supported by retrieved context.")
    unsupported_claims: int = Field(0, description="Number of claims not established by context.")
    contradicted_claims: int = Field(0, description="Number of claims directly contradicted by context.")
    claims: List[ClaimJudgment] = Field(default_factory=list, description="List of per-claim verdicts.")
    evaluator_info: Dict[str, Any] = Field(default_factory=dict, description="Metadata on the judge model or heuristic.")


class AnswerRelevanceResult(BaseModel):
    """
    Evaluation of how well the generated answer addresses the original query and aligns with reference expectations.
    """
    score: float = Field(..., ge=0.0, le=1.0, description="Overall answer relevance score.")
    is_refusal: bool = Field(False, description="Whether the answer refused to answer due to lack of evidence.")
    is_correct_refusal: bool = Field(False, description="Whether refusal was appropriate given unanswerable context.")
    completeness_score: float = Field(..., ge=0.0, le=1.0, description="Completeness relative to reference answer.")
    relevance_score: float = Field(..., ge=0.0, le=1.0, description="Direct relevance to the query question.")
    explanation: str = Field(..., description="Justification of the score.")
    evaluator_info: Dict[str, Any] = Field(default_factory=dict)


class CitationEvaluationReport(BaseModel):
    """
    Comprehensive audit of citation quality and factual support.
    """
    citation_id_validity: float = Field(
        1.0, ge=0.0, le=1.0, description="Fraction of cited identifiers that exist in retrieved context."
    )
    citation_support_precision: float = Field(
        1.0, ge=0.0, le=1.0, description="Fraction of cited claims where citation actually supports the claim."
    )
    citation_coverage: float = Field(
        1.0, ge=0.0, le=1.0, description="Fraction of factual claims that have at least one supporting citation."
    )
    total_citations_made: int = Field(0)
    valid_citation_ids: int = Field(0)
    invalid_citation_ids: List[str] = Field(default_factory=list)
    unsupported_citations: List[str] = Field(default_factory=list)
    missing_citation_claims: List[str] = Field(default_factory=list)
    contradictory_citations: List[str] = Field(default_factory=list)
    claims: List[ClaimJudgment] = Field(default_factory=list)
