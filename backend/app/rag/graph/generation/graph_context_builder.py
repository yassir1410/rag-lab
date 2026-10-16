import re
from typing import List, Dict, Any, Tuple
from app.models.graph import GraphEntity, GraphRelation, GraphEvidence


class GraphContextBuilder:
    """
    Builds structured, traceable context envelopes for Graph-Grounded LLM Generation.
    Strictly segregates knowledge graph triples/paths from verbatim documentary evidence [S1], [S2]
    and validates final citations.
    """

    def build_context(
        self,
        entities: List[GraphEntity],
        relations: List[GraphRelation],
        evidence: List[GraphEvidence],
        traversal_paths: List[List[str]],
        max_tokens: int = 3500,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        sections = []

        # 1. Structured Knowledge Triples & Paths
        triple_lines = []
        for path in traversal_paths:
            if len(path) == 3:
                triple_lines.append(f"• ({path[0]}) --[{path[1]}]--> ({path[2]})")
            elif len(path) == 1:
                triple_lines.append(f"• {path[0]}")

        if triple_lines:
            sections.append("### STRUCTURED KNOWLEDGE GRAPH RELATIONSHIPS:")
            sections.extend(triple_lines[:15])
            sections.append("")

        # 2. Key Entities
        ent_lines = []
        for ent in entities[:10]:
            aliases_str = f" (aliases: {', '.join(ent.aliases)})" if ent.aliases else ""
            ent_lines.append(f"• {ent.canonical_name} [{ent.entity_type}]{aliases_str}")

        if ent_lines:
            sections.append("### RELEVANT ENTITIES:")
            sections.extend(ent_lines)
            sections.append("")

        # 3. Authoritative Documentary Evidence Snippets
        sections.append("### DOCUMENTARY SOURCE EVIDENCE PASSAGES:")
        source_manifest: List[Dict[str, Any]] = []

        for idx, ev in enumerate(evidence, start=1):
            source_id = f"[S{idx}]"
            source_manifest.append({
                "source_id": source_id,
                "evidence_id": str(ev.id),
                "document_id": str(ev.document_id),
                "chunk_id": str(ev.chunk_id) if ev.chunk_id else None,
                "page_reference": ev.page_reference,
                "snippet": ev.evidence_text[:160] + "..." if len(ev.evidence_text) > 160 else ev.evidence_text,
            })

            sections.append(
                f"--- Source {source_id} (Page/Seq: {ev.page_reference or 'N/A'}) ---\n"
                f"{ev.evidence_text.strip()}\n"
            )

        if not evidence:
            sections.append("No direct documentary passages located for this query.")

        return "\n".join(sections), source_manifest

    def validate_citations(
        self,
        answer_text: str,
        source_manifest: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        cited_markers = re.findall(r"\[S\d+\]", answer_text)
        unique_markers = sorted(list(set(cited_markers)))

        manifest_by_id = {item["source_id"]: item for item in source_manifest}
        valid_citations: List[Dict[str, Any]] = []
        hallucinated: List[str] = []

        for m in unique_markers:
            if m in manifest_by_id:
                valid_citations.append(manifest_by_id[m])
            else:
                hallucinated.append(m)

        return valid_citations, hallucinated


graph_context_builder = GraphContextBuilder()
