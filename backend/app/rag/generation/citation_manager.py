import re
from typing import List, Dict, Set
from app.rag.generation.schemas import ContextSourceItem, CitationDetail


BRACKETED_GROUP_REGEX = re.compile(r"\[([^\]]+)\]")
CITATION_TOKEN_REGEX = re.compile(r"\b(S\d+)\b", re.IGNORECASE)


class CitationManager:
    """
    Manages source attribution, citation identifier parsing, and validation.
    Extracts citation references [S1], [S2], [S1, S2, S3] from generated text and verifies
    whether each referenced source actually exists in the retrieved context.
    """

    def parse_citations(self, text: str) -> List[str]:
        """Finds all unique citation markers (e.g. ['S1', 'S2']) in generated text in appearance order."""
        if not text:
            return []
        seen: Set[str] = set()
        ordered_citations: List[str] = []
        for bracket_content in BRACKETED_GROUP_REGEX.findall(text):
            for token in CITATION_TOKEN_REGEX.findall(bracket_content):
                norm = token.upper()
                if norm not in seen:
                    seen.add(norm)
                    ordered_citations.append(norm)
        return ordered_citations

    def validate_citations(
        self,
        generated_text: str,
        sources: List[ContextSourceItem],
    ) -> tuple[List[CitationDetail], List[str], List[str]]:
        """
        Validates citations extracted from generated text against provided context sources.
        Returns:
            (citation_details, invalid_citation_ids, uncited_source_ids)
        """
        parsed_ids = self.parse_citations(generated_text)
        sources_by_citation_id: Dict[str, ContextSourceItem] = {
            s.citation_id.upper().strip("[]"): s for s in sources
        }

        citation_details: List[CitationDetail] = []
        invalid_citations: List[str] = []

        # Track which sources were cited
        cited_ids_set = set(parsed_ids)
        for s in sources:
            norm_id = s.citation_id.upper().strip("[]")
            s.was_cited = norm_id in cited_ids_set

        for c_id in parsed_ids:
            formatted_tag = f"[{c_id}]"
            if c_id in sources_by_citation_id:
                src = sources_by_citation_id[c_id]
                citation_details.append(
                    CitationDetail(
                        citation_id=formatted_tag,
                        is_valid=True,
                        document_name=src.document_name,
                        page_number=src.page_number,
                        chunk_id=src.chunk_id,
                        supporting_passage=src.content[:300] + "..." if len(src.content) > 300 else src.content,
                        similarity_score=src.similarity_score,
                        caveat="Citation match indicates identifier reference, not proven semantic entailment.",
                    )
                )
            else:
                invalid_citations.append(formatted_tag)
                citation_details.append(
                    CitationDetail(
                        citation_id=formatted_tag,
                        is_valid=False,
                        document_name=None,
                        page_number=None,
                        chunk_id=None,
                        supporting_passage=None,
                        similarity_score=None,
                        caveat="Hallucinated citation: This identifier was not present in the retrieved context.",
                    )
                )

        # Calculate uncited sources
        uncited_sources: List[str] = [
            s.citation_id
            for s in sources
            if s.citation_id.upper().strip("[]") not in cited_ids_set
        ]

        return citation_details, invalid_citations, uncited_sources


citation_manager = CitationManager()
