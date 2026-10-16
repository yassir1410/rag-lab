import uuid
from datetime import datetime, timezone
from typing import List, Optional, Tuple, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import func, desc, or_

from app.models.conversation import Conversation, ConversationMessage, ConversationSummary


class ConversationRepository:
    """
    Data Access Layer for conversational sessions, sequence-ordered messages,
    and rolling summaries. Enforces atomic sequence allocation and cascading cleanup.
    """

    def create_conversation(
        self,
        session: Session,
        title: Optional[str] = None,
        settings: Optional[Dict[str, Any]] = None,
    ) -> Conversation:
        conv_title = (title or "").strip() or "Nouvelle Conversation"
        conv = Conversation(
            id=uuid.uuid4(),
            title=conv_title,
            settings=settings or {},
            status="active",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        session.add(conv)
        session.commit()
        session.refresh(conv)
        return conv

    def get_conversation(
        self,
        session: Session,
        conversation_id: uuid.UUID,
    ) -> Optional[Conversation]:
        return session.query(Conversation).filter(Conversation.id == conversation_id).first()

    def list_conversations(
        self,
        session: Session,
        skip: int = 0,
        limit: int = 50,
        search: Optional[str] = None,
        status: Optional[str] = None,
    ) -> Tuple[List[Conversation], int]:
        q = session.query(Conversation)
        if status:
            q = q.filter(Conversation.status == status)
        if search:
            search_clean = f"%{search.strip()}%"
            q = q.filter(Conversation.title.ilike(search_clean))

        total_count = q.count()
        conversations = (
            q.order_by(desc(Conversation.updated_at))
            .offset(skip)
            .limit(limit)
            .all()
        )
        return conversations, total_count

    def update_conversation(
        self,
        session: Session,
        conversation_id: uuid.UUID,
        title: Optional[str] = None,
        settings: Optional[Dict[str, Any]] = None,
        status: Optional[str] = None,
    ) -> Optional[Conversation]:
        conv = self.get_conversation(session, conversation_id)
        if not conv:
            return None

        if title is not None:
            conv.title = title.strip() or "Sans titre"
        if settings is not None:
            # Shallow merge or replace settings
            current_settings = dict(conv.settings or {})
            current_settings.update(settings)
            conv.settings = current_settings
        if status is not None:
            conv.status = status

        conv.updated_at = datetime.now(timezone.utc)
        session.commit()
        session.refresh(conv)
        return conv

    def delete_conversation(
        self,
        session: Session,
        conversation_id: uuid.UUID,
    ) -> bool:
        conv = self.get_conversation(session, conversation_id)
        if not conv:
            return False

        session.delete(conv)
        session.commit()
        return True

    def add_message(
        self,
        session: Session,
        conversation_id: uuid.UUID,
        role: str,
        content: str,
        metadata_json: Optional[Dict[str, Any]] = None,
        trace_id: Optional[str] = None,
    ) -> ConversationMessage:
        conv = self.get_conversation(session, conversation_id)
        if not conv:
            raise ValueError(f"Conversation '{conversation_id}' not found.")

        # Compute monotonic sequence number
        max_seq = (
            session.query(func.max(ConversationMessage.sequence_number))
            .filter(ConversationMessage.conversation_id == conversation_id)
            .scalar()
        )
        next_seq = (max_seq or 0) + 1

        msg = ConversationMessage(
            id=uuid.uuid4(),
            conversation_id=conversation_id,
            role=role,
            content=content,
            sequence_number=next_seq,
            metadata_json=metadata_json or {},
            trace_id=trace_id,
            created_at=datetime.now(timezone.utc),
        )
        session.add(msg)
        conv.updated_at = datetime.now(timezone.utc)
        session.commit()
        session.refresh(msg)
        return msg

    def get_all_ordered_messages(
        self,
        session: Session,
        conversation_id: uuid.UUID,
    ) -> List[ConversationMessage]:
        return (
            session.query(ConversationMessage)
            .filter(ConversationMessage.conversation_id == conversation_id)
            .order_by(ConversationMessage.sequence_number.asc())
            .all()
        )

    def get_messages(
        self,
        session: Session,
        conversation_id: uuid.UUID,
        skip: int = 0,
        limit: int = 100,
    ) -> Tuple[List[ConversationMessage], int]:
        q = (
            session.query(ConversationMessage)
            .filter(ConversationMessage.conversation_id == conversation_id)
        )
        total = q.count()
        messages = (
            q.order_by(ConversationMessage.sequence_number.asc())
            .offset(skip)
            .limit(limit)
            .all()
        )
        return messages, total

    def get_latest_summary(
        self,
        session: Session,
        conversation_id: uuid.UUID,
    ) -> Optional[ConversationSummary]:
        return (
            session.query(ConversationSummary)
            .filter(ConversationSummary.conversation_id == conversation_id)
            .order_by(desc(ConversationSummary.covered_until_sequence))
            .first()
        )

    def save_summary(
        self,
        session: Session,
        conversation_id: uuid.UUID,
        summary_text: str,
        covered_until_sequence: int,
        summarizer_model: Optional[str] = None,
        metadata_json: Optional[Dict[str, Any]] = None,
    ) -> ConversationSummary:
        summary = ConversationSummary(
            id=uuid.uuid4(),
            conversation_id=conversation_id,
            summary_text=summary_text,
            covered_until_sequence=covered_until_sequence,
            summarizer_model=summarizer_model,
            metadata_json=metadata_json or {},
            created_at=datetime.now(timezone.utc),
        )
        session.add(summary)
        session.commit()
        session.refresh(summary)
        return summary


conversation_repo = ConversationRepository()
