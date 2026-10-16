import unicodedata
import re
from typing import Dict


ABBREVIATION_MAP: Dict[str, str] = {
    "postgres": "PostgreSQL",
    "pgsql": "PostgreSQL",
    "py": "Python",
    "python3": "Python",
    "js": "JavaScript",
    "ts": "TypeScript",
    "k8s": "Kubernetes",
    "mongo": "MongoDB",
    "fast-api": "FastAPI",
    "node": "Node.js",
    "nodejs": "Node.js",
    "cicd": "CI/CD",
}


class EntityNormalizer:
    """
    Normalizes candidate entity mentions for consistent graph matching.
    Handles accent stripping, casing, whitespace normalization, and tech abbreviations.
    """

    def normalize_text(self, text: str) -> str:
        if not text:
            return ""
        # Strip accents
        nfkd = unicodedata.normalize("NFKD", text)
        no_accents = "".join([c for c in nfkd if not unicodedata.combining(c)])
        # Normalize whitespace
        cleaned = re.sub(r"\s+", " ", no_accents).strip()
        return cleaned

    def resolve_abbreviation(self, name: str) -> str:
        key = name.lower().strip()
        return ABBREVIATION_MAP.get(key, name)

    def canonicalize(self, name: str) -> str:
        norm = self.normalize_text(name)
        resolved = self.resolve_abbreviation(norm)
        return resolved


entity_normalizer = EntityNormalizer()
