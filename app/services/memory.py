"""Persisted conversational memory: chat history + structured preferences."""
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Candidate, ChatMessage
from app.schemas import Preferences


def load_preferences(candidate: Candidate) -> Preferences:
    return Preferences(**(candidate.preferences or {}))


def save_preferences(db: Session, candidate: Candidate, prefs: Preferences) -> None:
    candidate.preferences = prefs.model_dump()
    db.add(candidate)
    db.commit()


def add_message(db: Session, candidate_id: int, role: str, content: str) -> None:
    db.add(ChatMessage(candidate_id=candidate_id, role=role, content=content))
    db.commit()


def history(db: Session, candidate_id: int, limit: int = 20) -> list[dict]:
    rows = db.execute(
        select(ChatMessage)
        .where(ChatMessage.candidate_id == candidate_id)
        .order_by(ChatMessage.created_at.desc())
        .limit(limit)
    ).scalars().all()
    rows.reverse()
    return [{"role": m.role, "content": m.content} for m in rows]
