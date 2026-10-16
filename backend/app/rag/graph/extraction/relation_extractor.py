import re
import uuid
from typing import List, Optional
from app.rag.graph.models import ExtractedEntityCandidate, ExtractedRelationCandidate


class RuleBasedRelationExtractor:
    """
    Extracts relationship triples between co-occurring entities within text chunks.
    Preserves exact evidence snippets for verifiable citation grounding.
    """

    def extract_relations(
        self,
        text: str,
        entities: List[ExtractedEntityCandidate],
        chunk_id: Optional[uuid.UUID] = None,
        page_reference: Optional[int] = None,
        section_id: Optional[uuid.UUID] = None,
    ) -> List[ExtractedRelationCandidate]:
        relations: List[ExtractedRelationCandidate] = []
        if not entities or len(entities) < 2:
            return relations

        # Group entities by type
        persons = [e for e in entities if e.entity_type == "PERSON"]
        orgs = [e for e in entities if e.entity_type == "ORGANIZATION"]
        techs = [e for e in entities if e.entity_type == "TECHNOLOGY"]
        roles = [e for e in entities if e.entity_type == "JOB_ROLE"]
        locs = [e for e in entities if e.entity_type == "LOCATION"]

        # 1. PERSON -> STUDIED_AT -> ORGANIZATION
        for p in persons:
            for o in orgs:
                if any(kw in text.lower() for kw in ["graduate", "degree", "studied", "student", "diplôme", "insea"]):
                    snippet = self._find_connecting_sentence(text, o.canonical_name)
                    relations.append(
                        ExtractedRelationCandidate(
                            source_name=p.canonical_name,
                            target_name=o.canonical_name,
                            relation_type="STUDIED_AT",
                            confidence=0.95,
                            evidence_text=snippet or p.evidence_text,
                            chunk_id=chunk_id,
                            page_reference=page_reference,
                            section_id=section_id,
                        )
                    )

        # 2. PERSON -> HAS_SKILL -> TECHNOLOGY
        for p in persons:
            for t in techs:
                snippet = self._find_connecting_sentence(text, t.canonical_name)
                relations.append(
                    ExtractedRelationCandidate(
                        source_name=p.canonical_name,
                        target_name=t.canonical_name,
                        relation_type="HAS_SKILL",
                        confidence=0.92,
                        evidence_text=snippet or t.evidence_text,
                        chunk_id=chunk_id,
                        page_reference=page_reference,
                        section_id=section_id,
                    )
                )

        # 3. PERSON -> HOLDS_ROLE -> JOB_ROLE
        for p in persons:
            for r in roles:
                snippet = self._find_connecting_sentence(text, r.canonical_name)
                relations.append(
                    ExtractedRelationCandidate(
                        source_name=p.canonical_name,
                        target_name=r.canonical_name,
                        relation_type="HOLDS_ROLE",
                        confidence=0.94,
                        evidence_text=snippet or r.evidence_text,
                        chunk_id=chunk_id,
                        page_reference=page_reference,
                        section_id=section_id,
                    )
                )

        # 4. PERSON -> LOCATED_IN -> LOCATION
        for p in persons:
            for l in locs:
                snippet = self._find_connecting_sentence(text, l.canonical_name)
                relations.append(
                    ExtractedRelationCandidate(
                        source_name=p.canonical_name,
                        target_name=l.canonical_name,
                        relation_type="LOCATED_IN",
                        confidence=0.90,
                        evidence_text=snippet or l.evidence_text,
                        chunk_id=chunk_id,
                        page_reference=page_reference,
                        section_id=section_id,
                    )
                )

        # 5. JOB_ROLE -> USES -> TECHNOLOGY (if no PERSON detected)
        if not persons and roles and techs:
            for r in roles:
                for t in techs:
                    snippet = self._find_connecting_sentence(text, t.canonical_name)
                    relations.append(
                        ExtractedRelationCandidate(
                            source_name=r.canonical_name,
                            target_name=t.canonical_name,
                            relation_type="USES",
                            confidence=0.88,
                            evidence_text=snippet or t.evidence_text,
                            chunk_id=chunk_id,
                            page_reference=page_reference,
                            section_id=section_id,
                        )
                    )

        return relations

    def _find_connecting_sentence(self, text: str, term: str) -> str:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        for s in sentences:
            if term.lower() in s.lower():
                return s.strip()
        return text[:140].strip()


rule_relation_extractor = RuleBasedRelationExtractor()
