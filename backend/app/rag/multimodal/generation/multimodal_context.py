from typing import List, Dict, Any, Tuple


class MultimodalContextBuilder:
    """Builds structured multimodal context blocks containing textual excerpts, markdown tables, and image references."""

    @classmethod
    def build_context(
        cls,
        evidence_items: List[Dict[str, Any]],
        max_tokens: int = 2000,
    ) -> Tuple[str, List[str]]:
        """Formats evidence items into cited blocks with stable [TAG] references.
        
        Returns:
            context_string: str
            attached_image_paths: List[str]
        """
        lines = []
        image_paths = []

        for item in evidence_items:
            tag = item.get("citation_tag", "[EVID]")
            elem_type = item.get("element_type", "text")
            page = item.get("page_number", 1)
            content = item.get("content_text", "").strip()
            img_path = item.get("image_path")

            if img_path and img_path not in image_paths:
                image_paths.append(img_path)

            lines.append(f"--- EVIDENCE {tag} (Page {page}, Type: {elem_type}) ---")
            if elem_type == "table" and "markdown" in item.get("structured_data", {}):
                lines.append(item["structured_data"]["markdown"])
            elif content:
                lines.append(content)
            else:
                lines.append(f"[Visual asset attached: {elem_type} on page {page}]")
            lines.append("")

        context_str = "\n".join(lines)
        return context_str, image_paths
