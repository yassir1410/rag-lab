import re
from typing import List, Dict, Any, Set


class MultimodalCitationManager:
    """Extracts and validates multimodal citation tags in generated responses."""

    CITATION_REGEX = re.compile(r"\[([A-Z]+_P\d+|PAGE_\d+|TBL_P\d+|IMG_P\d+)\]", re.IGNORECASE)

    @classmethod
    def extract_citations(cls, text: str) -> List[str]:
        """Finds all bracketed citation tags."""
        matches = cls.CITATION_REGEX.findall(text)
        return [f"[{m.upper()}]" for m in matches]

    @classmethod
    def validate_citations(
        cls,
        generated_answer: str,
        evidence_items: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Validates that cited tags exist in evidence items and calculates citation integrity."""
        found_citations = cls.extract_citations(generated_answer)
        unique_found = list(set(found_citations))

        valid_evidence_tags: Set[str] = set()
        for item in evidence_items:
            tag = item.get("citation_tag", "").upper()
            if tag:
                valid_evidence_tags.add(tag)

        valid_citations = [c for c in unique_found if c in valid_evidence_tags]
        hallucinated = [c for c in unique_found if c not in valid_evidence_tags]

        precision = len(valid_citations) / len(unique_found) if unique_found else 1.0
        coverage = len(valid_citations) / len(valid_evidence_tags) if valid_evidence_tags else 0.0

        return {
            "all_citations": unique_found,
            "valid_citations": valid_citations,
            "hallucinated_citations": hallucinated,
            "is_grounded": len(hallucinated) == 0 and len(valid_citations) > 0,
            "citation_precision": round(precision, 3),
            "evidence_coverage": round(coverage, 3),
        }
