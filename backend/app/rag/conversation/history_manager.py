import re
from typing import List, Tuple, Optional
from app.models.conversation import ConversationMessage
from app.rag.chunking.tokenizer import default_tokenizer
from app.rag.conversation.schemas import ConversationMessageResponse


class HistoryManager:
    """
    Manages conversational history hygiene, validation, and serialization.
    Enforces role transitions (no arbitrary system injections), prevents conversational tampering,
    and formats memory blocks for prompt construction.
    """

    ALLOWED_ROLES = {"user", "assistant"}

    def __init__(self):
        self.tokenizer = default_tokenizer

    def validate_incoming_role(self, role: str) -> str:
        role_clean = (role or "").lower().strip()
        if role_clean not in self.ALLOWED_ROLES:
            raise ValueError(
                f"Invalid message role '{role}'. Only 'user' and 'assistant' are permitted."
            )
        return role_clean

    def sanitize_message_content(self, content: str) -> str:
        clean = (content or "").strip()
        if not clean:
            raise ValueError("Message content cannot be empty.")
        # Strip null bytes
        clean = clean.replace("\x00", "")
        return clean

    def validate_turn_order(self, current_role: str, last_message: Optional[ConversationMessage]) -> None:
        """
        Validates that conversation follows proper alternating turn order:
        User -> Assistant -> User -> Assistant.
        """
        if not last_message:
            if current_role != "user":
                raise ValueError("The first message in a conversation must be from the user.")
            return

        if last_message.role == current_role and current_role == "assistant":
            raise ValueError("Cannot append consecutive assistant messages.")

    def format_history_for_prompt(
        self,
        messages: List[ConversationMessageResponse],
        summary: Optional[str] = None,
    ) -> str:
        """
        Formats conversation memory into a clearly demarcated, segregated dialogue block.
        Includes rolling summary if present.
        """
        blocks = []

        if summary and summary.strip():
            blocks.append(f"--- CONVERSATION SUMMARY (PREVIOUS TURNS) ---\n{summary.strip()}\n")

        if messages:
            blocks.append("--- RECENT CONVERSATION HISTORY ---")
            for m in messages:
                speaker = "User" if m.role == "user" else "Assistant"
                blocks.append(f"{speaker}: {m.content}")

        return "\n".join(blocks) if blocks else ""

    def count_tokens(self, text: str) -> int:
        return self.tokenizer.count_tokens(text)


history_manager = HistoryManager()
