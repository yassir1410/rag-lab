import re
import uuid
from typing import List, Optional, Dict, Any, Tuple
from app.rag.graph.models import ExtractedEntityCandidate
from app.rag.graph.extraction.schema_validator import schema_validator


# Common technology dictionary with aliases
TECH_DICTIONARY: Dict[str, Dict[str, Any]] = {
    "Python": {"aliases": ["python3", "py"], "category": "Programming Language"},
    "Java": {"aliases": ["java8", "java17"], "category": "Programming Language"},
    "JavaScript": {"aliases": ["js", "ecmascript"], "category": "Programming Language"},
    "TypeScript": {"aliases": ["ts"], "category": "Programming Language"},
    "SQL": {"aliases": ["ansi sql"], "category": "Query Language"},
    "FastAPI": {"aliases": ["fast-api"], "category": "Backend Framework"},
    "Flask": {"aliases": [], "category": "Backend Framework"},
    "Node.js": {"aliases": ["nodejs", "node"], "category": "Runtime / Framework"},
    "Express": {"aliases": ["expressjs"], "category": "Backend Framework"},
    "Spring Boot": {"aliases": ["springboot", "spring"], "category": "Backend Framework"},
    "PostgreSQL": {"aliases": ["postgres", "pgsql"], "category": "Relational Database"},
    "Redis": {"aliases": ["redis-server"], "category": "In-Memory Database / Cache"},
    "MySQL": {"aliases": [], "category": "Relational Database"},
    "MongoDB": {"aliases": ["mongo"], "category": "NoSQL Database"},
    "Docker": {"aliases": ["docker-compose"], "category": "DevOps & Containers"},
    "pgvector": {"aliases": ["pg_vector"], "category": "Vector Extension"},
    "Git": {"aliases": ["github", "gitlab"], "category": "Version Control"},
    "CI/CD": {"aliases": ["cicd", "continuous integration"], "category": "DevOps Practice"},
    "Socket.io": {"aliases": ["socketio", "websockets"], "category": "Real-time Messaging"},
}

ORGANIZATION_NAMES = [
    "INSEA", "National Institute of Statistics and Applied Economics",
    "Mohammed V University", "OCP", "Capgemini", "Orange", "SQLI",
]

JOB_ROLES = [
    "Software Engineer", "Backend Engineer", "Full Stack Engineer",
    "Data Engineer", "DevOps Engineer", "Backend Developer",
]

LOCATIONS = [
    "Morocco", "Casablanca", "Rabat", "Tangier", "Marrakech", "France",
]


class RuleBasedEntityExtractor:
    """
    Deterministic rule-based and dictionary entity extractor.
    High throughput, 100% reproducible, zero external inference dependencies.
    """

    def extract_from_chunk(
        self,
        text: str,
        chunk_id: Optional[uuid.UUID] = None,
        page_reference: Optional[int] = None,
        section_id: Optional[uuid.UUID] = None,
    ) -> List[ExtractedEntityCandidate]:
        candidates: List[ExtractedEntityCandidate] = []
        seen_names = set()

        # 1. Extract Technologies
        for tech_name, meta in TECH_DICTIONARY.items():
            pattern = r"\b" + re.escape(tech_name) + r"\b"
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                evidence_snippet = self._extract_snippet(text, match.start(), match.end())
                candidates.append(
                    ExtractedEntityCandidate(
                        canonical_name=tech_name,
                        entity_type="TECHNOLOGY",
                        aliases=meta["aliases"],
                        properties={"category": meta["category"]},
                        confidence=0.98,
                        evidence_text=evidence_snippet,
                        chunk_id=chunk_id,
                        page_reference=page_reference,
                        section_id=section_id,
                    )
                )
                seen_names.add(tech_name.lower())

        # 2. Extract Organizations
        for org in ORGANIZATION_NAMES:
            pattern = r"\b" + re.escape(org) + r"\b"
            match = re.search(pattern, text, re.IGNORECASE)
            if match and org.lower() not in seen_names:
                evidence_snippet = self._extract_snippet(text, match.start(), match.end())
                candidates.append(
                    ExtractedEntityCandidate(
                        canonical_name=org,
                        entity_type="ORGANIZATION",
                        aliases=[],
                        properties={"type": "Academic" if "INSEA" in org or "University" in org else "Corporate"},
                        confidence=0.95,
                        evidence_text=evidence_snippet,
                        chunk_id=chunk_id,
                        page_reference=page_reference,
                        section_id=section_id,
                    )
                )
                seen_names.add(org.lower())

        # 3. Extract Job Roles
        for role in JOB_ROLES:
            pattern = r"\b" + re.escape(role) + r"\b"
            match = re.search(pattern, text, re.IGNORECASE)
            if match and role.lower() not in seen_names:
                evidence_snippet = self._extract_snippet(text, match.start(), match.end())
                candidates.append(
                    ExtractedEntityCandidate(
                        canonical_name=role,
                        entity_type="JOB_ROLE",
                        aliases=[],
                        properties={},
                        confidence=0.92,
                        evidence_text=evidence_snippet,
                        chunk_id=chunk_id,
                        page_reference=page_reference,
                        section_id=section_id,
                    )
                )
                seen_names.add(role.lower())

        # 4. Extract Locations
        for loc in LOCATIONS:
            pattern = r"\b" + re.escape(loc) + r"\b"
            match = re.search(pattern, text, re.IGNORECASE)
            if match and loc.lower() not in seen_names:
                evidence_snippet = self._extract_snippet(text, match.start(), match.end())
                candidates.append(
                    ExtractedEntityCandidate(
                        canonical_name=loc,
                        entity_type="LOCATION",
                        aliases=[],
                        properties={},
                        confidence=0.90,
                        evidence_text=evidence_snippet,
                        chunk_id=chunk_id,
                        page_reference=page_reference,
                        section_id=section_id,
                    )
                )
                seen_names.add(loc.lower())

        # 5. Extract Person Name (e.g. from resume header)
        person_match = re.search(r"^([A-Z][a-z]+(?:\s+[A-Z][a-zA-Z\-]+)+)", text.strip())
        if person_match and person_match.group(1).lower() not in seen_names:
            name = schema_validator.sanitize_name(person_match.group(1))
            if len(name.split()) in (2, 3) and not any(r.lower() in name.lower() for r in JOB_ROLES):
                evidence_snippet = text[:120].strip()
                candidates.append(
                    ExtractedEntityCandidate(
                        canonical_name=name,
                        entity_type="PERSON",
                        aliases=[],
                        properties={"role": "Candidate / Author"},
                        confidence=0.96,
                        evidence_text=evidence_snippet,
                        chunk_id=chunk_id,
                        page_reference=page_reference,
                        section_id=section_id,
                    )
                )
                seen_names.add(name.lower())

        return candidates

    def _extract_snippet(self, full_text: str, start: int, end: int, window: int = 60) -> str:
        s = max(0, start - window)
        e = min(len(full_text), end + window)
        snippet = full_text[s:e].replace("\n", " ").strip()
        return f"...{snippet}..." if (s > 0 or e < len(full_text)) else snippet


rule_entity_extractor = RuleBasedEntityExtractor()
