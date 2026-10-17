import os
from typing import List, Dict, Any, Tuple
from app.core.config import settings


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
            attached_image_paths: List[str]  (validated safe server-side filepaths)
        """
        lines = []
        image_paths = []
        allowed_dir = os.path.realpath(str(getattr(settings, "ABSOLUTE_STORAGE_DIR", "/app/data/uploads")))

        for item in evidence_items:
            tag = item.get("citation_tag", "[EVID]")
            elem_type = item.get("element_type", "text")
            page = item.get("page_number", 1)
            content = item.get("content_text", "").strip()

            # Prefer verified local server path for reading image bytes
            local_img = item.get("_local_image_path")
            if not local_img and item.get("image_path") and not item.get("image_path").startswith("/api/"):
                local_img = item.get("image_path")

            if local_img and os.path.exists(local_img):
                real_p = os.path.realpath(local_img)
                # Ensure path is inside authorized storage directory to prevent arbitrary file reading
                if (real_p.startswith(allowed_dir) or real_p.startswith("/app/data")) and real_p not in image_paths:
                    if os.path.getsize(real_p) <= 15 * 1024 * 1024:
                        image_paths.append(real_p)

            lines.append(f"--- EVIDENCE {tag} (Page {page}, Type: {elem_type}) ---")

            # Structured table priority
            struct = item.get("structured_data", {})
            md_table = struct.get("markdown") or struct.get("markdown_table")
            if elem_type == "table" and md_table:
                lines.append(md_table)
            elif content:
                lines.append(content)
            else:
                lines.append(f"[Visual asset attached: {elem_type} on page {page}]")

            lines.append("")

        context_str = "\n".join(lines)
        return context_str, image_paths
