import re
from typing import Optional, Tuple
from app.rag.graph.schemas import EntityType, RelationType


SUPPORTED_ENTITY_TYPES = {e.value for e in EntityType}
SUPPORTED_RELATION_TYPES = {r.value for r in RelationType}


class SchemaValidator:
    """
    Validates and normalizes candidate entity and relation types
    against the project ontology. Rejects ungrounded or unsupported structures.
    """

    def validate_entity_type(self, raw_type: str) -> Optional[str]:
        cleaned = raw_type.upper().strip()
        if cleaned in SUPPORTED_ENTITY_TYPES:
            return cleaned

        # Common alias mappings
        type_mapping = {
            "COMPANY": "ORGANIZATION",
            "UNIVERSITY": "ORGANIZATION",
            "SCHOOL": "ORGANIZATION",
            "LANG": "TECHNOLOGY",
            "FRAMEWORK": "TECHNOLOGY",
            "DATABASE": "TECHNOLOGY",
            "TOOL": "TECHNOLOGY",
            "ROLE": "JOB_ROLE",
            "TITLE": "JOB_ROLE",
            "CITY": "LOCATION",
            "COUNTRY": "LOCATION",
        }
        return type_mapping.get(cleaned)

    def validate_relation_type(self, raw_type: str) -> Optional[str]:
        cleaned = raw_type.upper().strip().replace(" ", "_").replace("-", "_")
        if cleaned in SUPPORTED_RELATION_TYPES:
            return cleaned

        # Common relation mapping
        rel_mapping = {
            "WORKS_AT": "WORKED_AT",
            "EMPLOYED_AT": "WORKED_AT",
            "BUILT": "DEVELOPED",
            "CREATED": "DEVELOPED",
            "KNOWS": "HAS_SKILL",
            "SKILLED_IN": "HAS_SKILL",
            "PROFICIENT_IN": "HAS_SKILL",
            "GRADUATED_FROM": "STUDIED_AT",
            "ATTENDED": "STUDIED_AT",
            "LOCATED_AT": "LOCATED_IN",
            "LIVES_IN": "LOCATED_IN",
        }
        return rel_mapping.get(cleaned)

    def sanitize_name(self, name: str) -> str:
        """Strips markdown asterisks, surrounding punctuation, quotes."""
        if not name:
            return ""
        cleaned = re.sub(r"^[\*\"\'\s\-\•]+|[\*\"\'\s\-\.\,]+$", "", name)
        return cleaned.strip()


schema_validator = SchemaValidator()
