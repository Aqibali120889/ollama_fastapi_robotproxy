import asyncio
import logging
import uuid
from typing import Any, Dict, Optional, Set

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.db import crud

logger = logging.getLogger(__name__)

SessionFactory = async_sessionmaker[AsyncSession]

# Fire-and-forget tasks must be kept referenced somewhere, or asyncio is free
# to garbage-collect them mid-execution (a well-known asyncio.create_task
# gotcha) -- this module-level set is that anchor, self-cleaning via the
# done callback.
_background_tasks: Set[asyncio.Task] = set()


def _fire_and_forget(coro) -> asyncio.Task:
    task = asyncio.create_task(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return task


async def _create_session(session_factory: SessionFactory, session_id: str, source: str) -> None:
    try:
        async with session_factory() as db:
            await crud.create_session(db, uuid.UUID(session_id), source=source)
    except Exception:
        logger.warning("Failed to persist session start for %s", session_id, exc_info=True)


async def _log_utterance(
    session_factory: SessionFactory, session_id: str, speaker: str, text: str, extra: Optional[Dict[str, Any]]
) -> None:
    try:
        async with session_factory() as db:
            await crud.append_utterance(db, uuid.UUID(session_id), speaker, text, extra=extra)
    except Exception:
        logger.warning("Failed to persist %s utterance for session %s", speaker, session_id, exc_info=True)


async def _close_session(session_factory: SessionFactory, session_id: str) -> None:
    try:
        async with session_factory() as db:
            await crud.close_session(db, uuid.UUID(session_id))
    except Exception:
        logger.warning("Failed to persist session close for %s", session_id, exc_info=True)


async def create_session_safe(
    session_factory: Optional[SessionFactory], session_id: str, source: str = "voice"
) -> Optional[asyncio.Task]:
    """Fire-and-forget: persistence must never add latency to a live voice
    turn (a Postgres outage can take multiple seconds to time out -- see
    app/db/database.py's connect timeout comment), only best-effort record it
    in the background. Returns the scheduled Task so tests can await it
    deterministically; production call sites ignore the return value."""
    if session_factory is None:
        return None
    return _fire_and_forget(_create_session(session_factory, session_id, source))


async def log_utterance_safe(
    session_factory: Optional[SessionFactory],
    session_id: str,
    speaker: str,
    text: str,
    extra: Optional[Dict[str, Any]] = None,
) -> Optional[asyncio.Task]:
    if session_factory is None or not text.strip():
        return None
    return _fire_and_forget(_log_utterance(session_factory, session_id, speaker, text, extra))


async def close_session_safe(session_factory: Optional[SessionFactory], session_id: str) -> Optional[asyncio.Task]:
    if session_factory is None:
        return None
    return _fire_and_forget(_close_session(session_factory, session_id))
