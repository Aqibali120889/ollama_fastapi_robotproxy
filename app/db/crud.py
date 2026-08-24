import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.db.models import ConversationSession, Utterance


async def create_session(db: AsyncSession, session_id: uuid.UUID, source: str = "voice") -> ConversationSession:
    session = ConversationSession(id=session_id, source=source)
    db.add(session)
    await db.commit()
    await db.refresh(session)
    return session


async def append_utterance(
    db: AsyncSession,
    session_id: uuid.UUID,
    speaker: str,
    text: str,
    extra: Optional[Dict[str, Any]] = None,
) -> Utterance:
    if speaker not in ("human", "robot"):
        raise ValueError(f"speaker must be 'human' or 'robot', got {speaker!r}")
    utterance = Utterance(session_id=session_id, speaker=speaker, text=text, extra=extra)
    db.add(utterance)
    await db.commit()
    await db.refresh(utterance)
    return utterance


async def close_session(db: AsyncSession, session_id: uuid.UUID) -> Optional[ConversationSession]:
    session = await db.get(ConversationSession, session_id)
    if session is None:
        return None
    session.closed_at = datetime.now(timezone.utc)
    await db.commit()
    await db.refresh(session)
    return session


async def get_session_with_utterances(db: AsyncSession, session_id: uuid.UUID) -> Optional[ConversationSession]:
    stmt = (
        select(ConversationSession)
        .where(ConversationSession.id == session_id)
        .options(selectinload(ConversationSession.utterances))
    )
    result = await db.execute(stmt)
    return result.scalar_one_or_none()
