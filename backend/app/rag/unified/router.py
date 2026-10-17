import re
import uuid
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session

from app.rag.unified.schemas import RoutingDecision, DocumentCapabilityReport
from app.rag.unified.capability_service import document_capability_service


WORD_TO_PAGE_NUM = {
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "fifth": 5,
    "sixth": 6,
    "seventh": 7,
    "eighth": 8,
    "ninth": 9,
    "tenth": 10,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}


def extract_explicit_page_number(query: str, request_page_number: Optional[int] = None) -> Optional[int]:
    """
    Extracts an unambiguous page number reference from query text if request_page_number is not provided.
    Explicit request_page_number always takes precedence.
    Conservative parsing: Returns None for ambiguous queries (e.g. multi-page comparisons or generic mentions).
    """
    if request_page_number is not None:
        return request_page_number

    if not query or not query.strip():
        return None

    pages_found: List[int] = []

    # 1. Matches "page 1", "on page 1", "page 001", "page: 1"
    digit_matches = re.findall(r"\b(?:on\s+)?page\s*[:#]?\s*(\d+)\b", query, re.IGNORECASE)
    for dm in digit_matches:
        try:
            val = int(dm)
            if val > 0 and val not in pages_found:
                pages_found.append(val)
        except ValueError:
            pass

    # 2. Matches "on the first page", "first page", "second page"
    ordinal_matches = re.findall(
        r"\b(?:on\s+the\s+|in\s+the\s+)?(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\s+page\b",
        query,
        re.IGNORECASE,
    )
    for om in ordinal_matches:
        val = WORD_TO_PAGE_NUM.get(om.lower())
        if val and val not in pages_found:
            pages_found.append(val)

    # 3. Matches "page one", "page two"
    word_matches = re.findall(
        r"\bpage\s+(one|two|three|four|five|six|seven|eight|nine|ten)\b",
        query,
        re.IGNORECASE,
    )
    for wm in word_matches:
        val = WORD_TO_PAGE_NUM.get(wm.lower())
        if val and val not in pages_found:
            pages_found.append(val)

    # If exactly one unambiguous page was referenced, return it
    if len(pages_found) == 1:
        return pages_found[0]

    # If multiple distinct pages were mentioned (e.g., "compare page 1 and page 2") or none, return None
    return None


class IntelligentQueryRouter:
    """
    Intelligent Query Router for Unified RAG.
    Selects optimal retrieval engine(s) based on query semantics, document capabilities, and routing mode.
    Exposes transparent, explainable decision trails.
    """

    TABLE_PATTERNS = [
        r"\btable\b", r"\btableau\b", r"\btabular\b", r"\bmatrix\b", r"\bgrid\b",
        r"\brow\b", r"\bcolumn\b", r"\bgravity\b", r"\bmass\b", r"\bdiameter\b",
        r"\bdensity\b", r"\borbital\b", r"\bmetric\b", r"\bnumbers?\b", r"\bvalues?\b",
        r"\bparameters?\b", r"\bmeasurements?\b", r"\bcompare (the )?data\b"
    ]

    VISUAL_PATTERNS = [
        r"\bfigure\b", r"\bimage\b", r"\bdiagram\b", r"\bchart\b", r"\bphoto(graph)?\b",
        r"\bpicture\b", r"\bvisual\b", r"\billustration\b", r"\blook like\b",
        r"\bcolor\b", r"\bschematic\b", r"\bplot\b", r"\bvisible\b"
    ]

    GRAPH_PATTERNS = [
        r"\brelationship\b", r"\bconnected to\b", r"\bconnection\b", r"\bworked with\b",
        r"\bcollaborat(e|ed|ion)\b", r"\bnetwork\b", r"\bknows\b", r"\bassociated with\b",
        r"\bbetween .+ and .+\b", r"\bentity\b", r"\bwho worked\b"
    ]

    AGENTIC_PATTERNS = [
        r"\bmulti-step\b", r"\bdeep investigation\b", r"\bplan and execute\b",
        r"\bfind evidence across multiple sections\b", r"\bcomprehensive audit\b"
    ]

    ALL_KNOWN_ROUTES = {
        "text_dense",
        "text_hybrid",
        "multimodal_table",
        "multimodal_visual",
        "graph",
        "agentic",
    }

    def route_query(
        self,
        session: Session,
        question: str,
        document_id: Optional[uuid.UUID] = None,
        page_number: Optional[int] = None,
        mode: str = "auto",
        manual_routes: Optional[List[str]] = None,
    ) -> RoutingDecision:
        cap_report: DocumentCapabilityReport = document_capability_service.get_document_capability(
            session=session, document_id=document_id
        )

        effective_page = extract_explicit_page_number(question, page_number)

        # 1. Manual Routing Mode
        if mode == "manual" and manual_routes:
            valid_routes = [r for r in manual_routes if r in self.ALL_KNOWN_ROUTES]
            warnings: List[str] = []
            for r in valid_routes:
                if r not in cap_report.supported_routes and cap_report.document_id:
                    warnings.append(
                        f"Route '{r}' was manually selected but document '{cap_report.filename}' lacks the required index."
                    )
            return RoutingDecision(
                mode="manual",
                selected_routes=valid_routes or ["text_hybrid"],
                routing_reasons=["Explicit user manual route selection."],
                intent_detected=["manual_override"],
                capabilities_checked=cap_report.model_dump(),
                warnings=warnings,
                fallback_decisions=[],
                effective_page_number=effective_page,
            )

        # 2. Comparative Routing Mode
        if mode == "comparative":
            # For comparative evaluation, select up to 3 candidate routes that have indexes
            comp_routes = [r for r in ["text_hybrid", "multimodal_table", "multimodal_visual", "graph"] if r in cap_report.supported_routes]
            if not comp_routes:
                comp_routes = ["text_hybrid"]
            return RoutingDecision(
                mode="comparative",
                selected_routes=comp_routes,
                routing_reasons=["Pedagogical side-by-side comparative routing enabled."],
                intent_detected=["comparative_evaluation"],
                capabilities_checked=cap_report.model_dump(),
                warnings=[],
                fallback_decisions=[],
                effective_page_number=effective_page,
            )

        # 3. Automatic Intelligent Routing Mode
        q_lower = question.lower()
        selected: List[str] = []
        reasons: List[str] = []
        intents: List[str] = []
        warnings: List[str] = []
        fallbacks: List[str] = []

        if effective_page is not None:
            if page_number is not None:
                reasons.append(f"Applied explicit request page filter: {effective_page}.")
            else:
                reasons.append(f"Extracted explicit page {effective_page} constraint from query.")

        is_table_q = any(re.search(pat, q_lower) for pat in self.TABLE_PATTERNS)
        is_visual_q = any(re.search(pat, q_lower) for pat in self.VISUAL_PATTERNS)
        is_graph_q = any(re.search(pat, q_lower) for pat in self.GRAPH_PATTERNS)
        is_agentic_q = any(re.search(pat, q_lower) for pat in self.AGENTIC_PATTERNS)

        if is_table_q:
            intents.append("tabular_query")
        if is_visual_q:
            intents.append("visual_query")
        if is_graph_q:
            intents.append("relational_query")
        if is_agentic_q:
            intents.append("multi_step_query")

        # Routing logic:
        # A. Table Intent
        if is_table_q:
            if "multimodal_table" in cap_report.supported_routes:
                selected.append("multimodal_table")
                reasons.append("Question contains tabular terms or parameters and document contains extracted tables.")
            else:
                warnings.append("Tabular intent detected, but document has no extracted table structures.")

        # B. Visual Intent
        if is_visual_q:
            if "multimodal_visual" in cap_report.supported_routes:
                selected.append("multimodal_visual")
                reasons.append("Visual or diagrammatic intent detected and visual page assets are available.")
            else:
                warnings.append("Visual intent detected, but visual/CLIP embeddings are unavailable.")

        # C. Graph Intent
        if is_graph_q:
            if "graph" in cap_report.supported_routes:
                selected.append("graph")
                reasons.append("Entity relational intent detected and knowledge graph evidence is present.")
            else:
                warnings.append("Entity relational intent detected, but Knowledge Graph is not constructed.")

        # D. Agentic Intent (Only when explicitly matched or requested, never mandatory)
        if is_agentic_q and "agentic" in cap_report.supported_routes:
            selected.append("agentic")
            reasons.append("Complex multi-step analytical intent detected.")

        # E. Default Text / Fallback Handling
        if not selected:
            if "text_hybrid" in cap_report.supported_routes:
                selected.append("text_hybrid")
                reasons.append("Standard textual query routed to hybrid dense + lexical retrieval.")
            elif "text_dense" in cap_report.supported_routes:
                selected.append("text_dense")
                reasons.append("Text query routed to dense vector search.")
            elif "multimodal_table" in cap_report.supported_routes:
                selected.append("multimodal_table")
                reasons.append(f"Document '{cap_report.filename}' lacks text chunks; falling back to multimodal table retrieval.")
                fallbacks.append("Fallback to multimodal_table due to absence of text chunks.")
            elif "multimodal_visual" in cap_report.supported_routes:
                selected.append("multimodal_visual")
                reasons.append(f"Document '{cap_report.filename}' lacks text chunks; falling back to visual page retrieval.")
                fallbacks.append("Fallback to multimodal_visual due to absence of text chunks.")
            else:
                selected.append("text_hybrid")
                warnings.append(f"Selected document '{cap_report.filename}' has not been chunked or embedded yet.")

        # Ensure no duplicate routes while preserving order
        dedup_selected = list(dict.fromkeys(selected))

        return RoutingDecision(
            mode="auto",
            selected_routes=dedup_selected,
            routing_reasons=reasons,
            intent_detected=intents or ["general_text"],
            capabilities_checked=cap_report.model_dump(),
            warnings=warnings,
            fallback_decisions=fallbacks,
            effective_page_number=effective_page,
        )


intelligent_router = IntelligentQueryRouter()
