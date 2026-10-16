from typing import List, Dict, Any, Optional, Sequence
from app.evaluation.generation.schemas import (
    ClaimJudgment,
    CitationEvaluationReport,
    FaithfulnessResult,
)
from app.rag.generation.citation_manager import citation_manager
from app.rag.generation.schemas import ContextSourceItem


class CitationEvaluator:
    """
    Comprehensive Citation Quality Evaluator.
    Measures syntactic ID validity, citation support precision, citation coverage,
    and detects unsupported, missing, or contradictory citations.
    """

    def evaluate(
        self,
        generated_answer: str,
        sources: Sequence[ContextSourceItem],
        faithfulness_result: FaithfulnessResult,
    ) -> CitationEvaluationReport:
        if not generated_answer or not generated_answer.strip():
            return CitationEvaluationReport(
                citation_id_validity=1.0,
                citation_support_precision=1.0,
                citation_coverage=1.0,
                total_citations_made=0,
                valid_citation_ids=0,
                invalid_citation_ids=[],
                unsupported_citations=[],
                missing_citation_claims=[],
                contradictory_citations=[],
                claims=[],
            )

        # 1. Parse all citations from the generated answer
        raw_parsed_citations = citation_manager.parse_citations(generated_answer)
        valid_source_ids = {s.citation_id.upper().strip("[]") for s in sources}

        invalid_citations: List[str] = []
        valid_citations_count = 0
        for c in raw_parsed_citations:
            norm = c.upper().strip("[]")
            if norm in valid_source_ids:
                valid_citations_count += 1
            else:
                invalid_citations.append(f"[{norm}]")

        total_citations = len(raw_parsed_citations)
        id_validity = (
            float(valid_citations_count) / float(total_citations)
            if total_citations > 0
            else 1.0
        )

        # 2. Analyze claim-level citations from faithfulness judgments
        claims = faithfulness_result.claims
        total_claims = len(claims)

        claims_with_citations = [c for c in claims if c.citations]
        claims_without_citations = [c for c in claims if not c.citations]

        # Missing citations: factual claims that made assertions without any citation
        missing_citation_claims = [c.claim for c in claims_without_citations]

        # Citation support precision: among claims with citations, what fraction are actually supported?
        supported_cited_claims = [c for c in claims_with_citations if c.status == "supported"]
        unsupported_cited_claims = [c for c in claims_with_citations if c.status == "unsupported"]
        contradicted_cited_claims = [c for c in claims_with_citations if c.status == "contradicted"]

        citation_support_precision = (
            float(len(supported_cited_claims)) / float(len(claims_with_citations))
            if claims_with_citations
            else 1.0
        )

        # Citation coverage: fraction of total factual claims that have at least one supporting citation
        citation_coverage = (
            float(len(supported_cited_claims)) / float(total_claims)
            if total_claims > 0
            else 1.0
        )

        unsupported_citations = [
            f"{c.claim} (Cited: {', '.join(c.citations)})" for c in unsupported_cited_claims
        ]
        contradictory_citations = [
            f"{c.claim} (Cited: {', '.join(c.citations)})" for c in contradicted_cited_claims
        ]

        return CitationEvaluationReport(
            citation_id_validity=round(id_validity, 4),
            citation_support_precision=round(citation_support_precision, 4),
            citation_coverage=round(citation_coverage, 4),
            total_citations_made=total_citations,
            valid_citation_ids=valid_citations_count,
            invalid_citation_ids=invalid_citations,
            unsupported_citations=unsupported_citations,
            missing_citation_claims=missing_citation_claims,
            contradictory_citations=contradictory_citations,
            claims=claims,
        )


citation_evaluator = CitationEvaluator()
