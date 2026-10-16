import re
from typing import List, Tuple, Dict, Any, Optional

BRACKET_CITATION_REGEX = re.compile(r"\[(S\d+(?:,\s*S\d+)*)\]", re.IGNORECASE)
SINGLE_CITATION_REGEX = re.compile(r"\b(S\d+)\b", re.IGNORECASE)
SENTENCE_SPLIT_REGEX = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")

BOILERPLATE_PREFIX_REGEX = re.compile(
    r"^(?:"
    r"based on the (?:provided|retrieved|given)?\s*(?:documents?|context|sources?|passages?|information),?\s*|"
    r"according to the (?:provided|retrieved|given)?\s*(?:documents?|context|sources?|passages?|information),?\s*|"
    r"in the (?:provided|retrieved|given)?\s*(?:documents?|context|sources?|passages?),?\s*|"
    r"sure,?\s*(?:here is the answer|here is what i found|here is the information)?:?\s*|"
    r"d'après les documents? (?:fournis?|disponibles?)?,?\s*|"
    r"selon les (?:documents?|sources?|informations?),?\s*|"
    r"as mentioned in the (?:documents?|context),?\s*"
    r")+",
    re.IGNORECASE,
)

CONVERSATIONAL_FILLER_PATTERNS = [
    re.compile(r"^(?:i hope this helps|let me know if you (?:have questions|need anything else)|feel free to ask).*$", re.IGNORECASE),
    re.compile(r"^(?:n'hésitez pas à demander|j'espère que cela vous aide).*$", re.IGNORECASE),
]

REFUSAL_PATTERNS = [
    "insufficient evidence",
    "cannot answer",
    "not enough information",
    "does not mention",
    "no information",
    "aucun élément",
    "preuves insuffisantes",
]


class ClaimExtractor:
    """
    Extracts atomic factual claims from generated text, mapping each claim to its associated citation tags.
    """

    def is_refusal(self, text: str) -> bool:
        lowered = text.lower()
        return any(pat in lowered for pat in REFUSAL_PATTERNS)

    def extract_claims(self, text: str) -> List[Dict[str, Any]]:
        """
        Parses sentences/clauses into discrete propositions with attached citations.
        Returns a list of dicts: {"claim": str, "citations": List[str]}
        """
        if not text or not text.strip():
            return []

        cleaned_text = text.strip()
        if self.is_refusal(cleaned_text):
            # If the entire response is a standard refusal, it is a single non-factual refusal statement
            return [
                {
                    "claim": cleaned_text,
                    "citations": [],
                    "is_refusal": True,
                }
            ]

        # Break into sentences
        paragraphs = [p.strip() for p in cleaned_text.split("\n") if p.strip()]
        claims: List[Dict[str, Any]] = []

        for para in paragraphs:
            sentences = SENTENCE_SPLIT_REGEX.split(para)
            for sentence in sentences:
                s_clean = sentence.strip()
                if not s_clean or len(s_clean) < 10:
                    continue

                # Filter conversational fillers
                if any(pat.match(s_clean) for pat in CONVERSATIONAL_FILLER_PATTERNS):
                    continue

                # Find all citations within this sentence
                citations_found = []
                for bracket_match in BRACKET_CITATION_REGEX.findall(s_clean):
                    for token in SINGLE_CITATION_REGEX.findall(bracket_match):
                        tag = f"[{token.upper()}]"
                        if tag not in citations_found:
                            citations_found.append(tag)

                # Clean citation tags from the claim statement for readability
                claim_text = BRACKET_CITATION_REGEX.sub("", s_clean).strip()
                # Strip conversational/introductory boilerplate clauses
                claim_text = BOILERPLATE_PREFIX_REGEX.sub("", claim_text).strip()
                # Capitalize first character if lowercase after prefix removal
                if claim_text and claim_text[0].islower():
                    claim_text = claim_text[0].upper() + claim_text[1:]
                # Clean multiple spaces
                claim_text = re.sub(r"\s+", " ", claim_text)

                if claim_text and len(claim_text) > 8:
                    claims.append(
                        {
                            "claim": claim_text,
                            "citations": citations_found,
                            "is_refusal": False,
                        }
                    )

        return claims


claim_extractor = ClaimExtractor()
