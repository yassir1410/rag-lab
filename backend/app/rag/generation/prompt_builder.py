import os
from dataclasses import dataclass
from typing import Dict, List, Optional
from app.rag.generation.schemas import PromptTemplateResponse


@dataclass
class BuiltPrompt:
    template_id: str
    system_prompt: str
    user_prompt: str
    full_prompt: str
    estimated_tokens: int


TEMPLATE_METADATA = {
    "basic": {
        "name": "Basic RAG",
        "description": "Standard naive question and context prompt without strict citation constraints.",
    },
    "grounded": {
        "name": "Grounded RAG (Strict)",
        "description": "Evidence-only answers with bracketed [S#] citations and refusal when evidence is missing.",
    },
    "concise": {
        "name": "Concise RAG",
        "description": "Direct, 1-3 sentence answers with essential bracketed citations.",
    },
    "detailed": {
        "name": "Detailed Analytical RAG",
        "description": "Structured analytical walkthroughs with sections, bullet points, and source attributions.",
    },
}


class PromptBuilder:
    """
    Loads, parses, and formats configurable RAG prompt templates.
    Stored as plain-text files in templates/ directory for inspection and clarity.
    """

    def __init__(self, templates_dir: Optional[str] = None):
        if templates_dir is None:
            templates_dir = os.path.join(os.path.dirname(__file__), "templates")
        self.templates_dir = templates_dir
        self._raw_templates: Dict[str, str] = {}
        self._load_templates()

    def _load_templates(self) -> None:
        if not os.path.exists(self.templates_dir):
            return
        for fname in os.listdir(self.templates_dir):
            if fname.endswith(".txt"):
                t_id = fname[:-4]
                fpath = os.path.join(self.templates_dir, fname)
                with open(fpath, "r", encoding="utf-8") as f:
                    self._raw_templates[t_id] = f.read()

    def list_templates(self) -> List[PromptTemplateResponse]:
        templates = []
        for t_id, meta in TEMPLATE_METADATA.items():
            raw = self._raw_templates.get(t_id, "")
            sys_part, user_part = self._split_template(raw)
            templates.append(
                PromptTemplateResponse(
                    id=t_id,
                    name=meta["name"],
                    description=meta["description"],
                    system_prompt_preview=sys_part[:120] + "..." if len(sys_part) > 120 else sys_part,
                    user_prompt_preview=user_part[:120] + "..." if len(user_part) > 120 else user_part,
                )
            )
        return templates

    def _split_template(self, raw_text: str) -> tuple[str, str]:
        parts = raw_text.split("===USER===")
        if len(parts) == 2:
            sys_part = parts[0].replace("===SYSTEM===", "").strip()
            user_part = parts[1].strip()
            return sys_part, user_part
        return "", raw_text.strip()

    def build_prompt(
        self,
        template_id: str,
        question: str,
        context: str,
    ) -> BuiltPrompt:
        raw = self._raw_templates.get(template_id)
        if not raw:
            # Fallback to grounded template
            raw = self._raw_templates.get("grounded", "")
            template_id = "grounded"

        sys_part, user_part = self._split_template(raw)
        formatted_user = user_part.format(question=question, context=context)

        full = f"System: {sys_part}\n\nUser: {formatted_user}" if sys_part else formatted_user
        est_tokens = len(full.split())  # Fast rough token count

        return BuiltPrompt(
            template_id=template_id,
            system_prompt=sys_part,
            user_prompt=formatted_user,
            full_prompt=full,
            estimated_tokens=est_tokens,
        )


prompt_builder = PromptBuilder()
