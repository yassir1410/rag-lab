import json
import re
import uuid
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from app.rag.generation.base import GenerationRequest
from app.rag.generation.registry import llm_registry
from app.rag.graph.models import ExtractedEntityCandidate, ExtractedRelationCandidate, GraphExtractionResult
from app.rag.graph.extraction.schema_validator import schema_validator
from app.rag.graph.extraction.entity_extractor import rule_entity_extractor
from app.rag.graph.extraction.relation_extractor import rule_relation_extractor


class LLMEntitySchema(BaseModel):
    name: str
    type: str
    aliases: List[str] = Field(default_factory=list)
    evidence: str = ""


class LLMRelationSchema(BaseModel):
    source: str
    target: str
    relation: str
    evidence: str = ""
    confidence: float = 0.9


class LLMExtractionPayload(BaseModel):
    entities: List[LLMEntitySchema] = Field(default_factory=list)
    relations: List[LLMRelationSchema] = Field(default_factory=list)


class StructuredLLMExtractor:
    """
    LLM-powered knowledge graph extractor using strict schema-validated structured output.
    Gracefully falls back to rule-based extraction for deterministic mock tests.
    """

    def __init__(self, provider_id: str = "mock", model_id: str = "mock-instruct-v1"):
        self.provider_id = provider_id
        self.model_id = model_id

    def extract(
        self,
        text: str,
        chunk_id: Optional[uuid.UUID] = None,
        page_reference: Optional[int] = None,
        section_id: Optional[uuid.UUID] = None,
    ) -> GraphExtractionResult:
        if self.provider_id == "mock":
            # Deterministic extraction via rule extractor
            rule_entities = rule_entity_extractor.extract_from_chunk(
                text, chunk_id=chunk_id, page_reference=page_reference, section_id=section_id
            )
            rule_relations = rule_relation_extractor.extract_relations(
                text, rule_entities, chunk_id=chunk_id, page_reference=page_reference, section_id=section_id
            )
            return GraphExtractionResult(
                entities=rule_entities,
                relations=rule_relations,
                extraction_method="mock_rule_hybrid",
            )

        provider = llm_registry.get_provider(self.provider_id)
        prompt = f"""Extract knowledge graph entities and relationship triples from the text passage.

Allowed Entity Types: PERSON, ORGANIZATION, PROJECT, TECHNOLOGY, JOB_ROLE, LOCATION, DATE, CERTIFICATION
Allowed Relation Types: WORKED_AT, DEVELOPED, USES, HAS_SKILL, STUDIED_AT, BELONGS_TO, LOCATED_IN, HOLDS_ROLE, CERTIFIED_IN

Output strictly a JSON object conforming to:
{{
  "entities": [
    {{"name": "Python", "type": "TECHNOLOGY", "aliases": ["py"], "evidence": "text snippet"}},
    {{"name": "INSEA", "type": "ORGANIZATION", "aliases": [], "evidence": "text snippet"}}
  ],
  "relations": [
    {{"source": "Candidate", "target": "Python", "relation": "HAS_SKILL", "evidence": "text snippet"}}
  ]
}}

Text to process:
\"\"\"{text}\"\"\"
"""
        try:
            req = GenerationRequest(
                prompt=prompt,
                system_prompt="You are a Knowledge Graph information extraction expert. Output strictly valid JSON.",
                temperature=0.0,
                max_tokens=600,
            )
            resp = provider.generate(req)
            parsed = self._parse_json(resp.content)
            if parsed:
                validated = LLMExtractionPayload(**parsed)
                entities = []
                for e in validated.entities:
                    e_type = schema_validator.validate_entity_type(e.type)
                    if e_type:
                        clean_name = schema_validator.sanitize_name(e.name)
                        if clean_name:
                            entities.append(
                                ExtractedEntityCandidate(
                                    canonical_name=clean_name,
                                    entity_type=e_type,
                                    aliases=e.aliases,
                                    evidence_text=e.evidence or text[:120],
                                    chunk_id=chunk_id,
                                    page_reference=page_reference,
                                    section_id=section_id,
                                )
                            )

                relations = []
                for r in validated.relations:
                    rel_type = schema_validator.validate_relation_type(r.relation)
                    if rel_type:
                        s_name = schema_validator.sanitize_name(r.source)
                        t_name = schema_validator.sanitize_name(r.target)
                        if s_name and t_name:
                            relations.append(
                                ExtractedRelationCandidate(
                                    source_name=s_name,
                                    target_name=t_name,
                                    relation_type=rel_type,
                                    confidence=r.confidence,
                                    evidence_text=r.evidence or text[:120],
                                    chunk_id=chunk_id,
                                    page_reference=page_reference,
                                    section_id=section_id,
                                )
                            )

                return GraphExtractionResult(
                    entities=entities,
                    relations=relations,
                    extraction_method="llm_structured",
                )
        except Exception:
            pass

        # Fallback to rules on error
        rule_entities = rule_entity_extractor.extract_from_chunk(
            text, chunk_id=chunk_id, page_reference=page_reference, section_id=section_id
        )
        rule_relations = rule_relation_extractor.extract_relations(
            text, rule_entities, chunk_id=chunk_id, page_reference=page_reference, section_id=section_id
        )
        return GraphExtractionResult(
            entities=rule_entities,
            relations=rule_relations,
            extraction_method="rule_fallback",
        )

    def _parse_json(self, text: str) -> Optional[Dict[str, Any]]:
        text = text.strip()
        try:
            return json.loads(text)
        except Exception:
            pass
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except Exception:
                pass
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except Exception:
                pass
        return None
