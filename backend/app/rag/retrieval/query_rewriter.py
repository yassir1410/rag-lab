import re
from typing import List, Dict, Any, Optional

COMMON_TECH_ACRONYMS = {
    "rag": "Retrieval-Augmented Generation RAG",
    "knn": "K-Nearest Neighbors KNN",
    "ann": "Approximate Nearest Neighbor ANN",
    "hnsw": "Hierarchical Navigable Small World HNSW",
    "ivf": "Inverted File Index IVFFlat",
    "fts": "Full-Text Search FTS tsvector",
    "db": "database PostgreSQL",
    "api": "Application Programming Interface REST API",
    "llm": "Large Language Model LLM",
}

STOPWORDS = {
    "a", "an", "the", "and", "or", "in", "on", "at", "to", "for", "with", "by",
    "from", "about", "into", "through", "during", "is", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "do", "does", "did", "can",
    "could", "should", "would", "what", "which", "who", "whom", "this", "that",
    "these", "those", "how", "why", "when", "where", "explain", "describe",
    "tell", "me", "show", "list", "give", "de", "la", "le", "les", "des", "du",
    "un", "une", "et", "ou", "pour", "dans", "sur", "avec", "par", "est", "sont",
}


class QueryProcessor:
    """
    Advanced query pre-processing engine providing:
    1. Keyword & Entity Extraction for high-precision lexical filtering.
    2. Query Rewriting and Acronym Expansion.
    3. Multi-Query Expansion decomposing complex inquiries into sub-queries.
    """

    def extract_keywords(self, query: str) -> List[str]:
        """Extracts significant alphanumeric tokens and domain keywords."""
        tokens = re.findall(r"\b[a-zA-Z0-9_\-+#.]{2,}\b", query)
        keywords = [t for t in tokens if t.lower() not in STOPWORDS]
        # Preserve uniqueness while maintaining order
        seen = set()
        deduped = []
        for kw in keywords:
            kw_lower = kw.lower()
            if kw_lower not in seen:
                seen.add(kw_lower)
                deduped.append(kw)
        return deduped

    def rewrite_query(self, query: str) -> str:
        """Expands known abbreviations and domain terms to boost retrieval recall."""
        tokens = query.split()
        rewritten_tokens = []
        for tok in tokens:
            cleaned = re.sub(r"[^\w]", "", tok).lower()
            if cleaned in COMMON_TECH_ACRONYMS:
                rewritten_tokens.append(COMMON_TECH_ACRONYMS[cleaned])
            else:
                rewritten_tokens.append(tok)
        return " ".join(rewritten_tokens)

    def expand_multi_query(self, query: str, max_queries: int = 3) -> List[str]:
        """
        Decomposes or generates multiple perspectives of the user query:
        1. Original user query
        2. Keyword-focused formulation for lexical retrieval
        3. Conceptual / definitional expansion
        """
        queries = [query.strip()]
        keywords = self.extract_keywords(query)

        # 2. Keyword-dense variation
        if keywords and len(keywords) >= 2:
            kw_variation = " ".join(keywords)
            if kw_variation.lower() != query.lower():
                queries.append(kw_variation)

        # 3. Rewritten / expanded acronym variation
        rewritten = self.rewrite_query(query)
        if rewritten.lower() != query.lower() and rewritten not in queries:
            queries.append(rewritten)

        # 4. Contextual detail variation
        if len(queries) < max_queries and keywords:
            detail_query = f"{' '.join(keywords[:4])} technical details and architecture"
            queries.append(detail_query)

        return queries[:max_queries]


query_processor = QueryProcessor()
