import uuid

from fastapi import APIRouter, HTTPException, Request

from app.db import crud
from app.db.models import ConversationSession
from app.schemas import (
    AppendUtteranceRequest,
    CreateSessionRequest,
    SessionReplayResponse,
    SessionResponse,
    UtteranceResponse,
)

router = APIRouter(prefix="/api/sessions", tags=["sessions"])


@router.post("", response_model=SessionResponse, status_code=201)
async def create_session_endpoint(payload: CreateSessionRequest, request: Request) -> SessionResponse:
    session_factory = request.app.state.db_session_factory
    async with session_factory() as db:
        session = await crud.create_session(db, uuid.uuid4(), source=payload.source)
        return SessionResponse.model_validate(session)


@router.post("/{session_id}/utterances", response_model=UtteranceResponse, status_code=201)
async def append_utterance_endpoint(
    session_id: uuid.UUID, payload: AppendUtteranceRequest, request: Request
) -> UtteranceResponse:
    session_factory = request.app.state.db_session_factory
    async with session_factory() as db:
        if await db.get(ConversationSession, session_id) is None:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found.")
        utterance = await crud.append_utterance(db, session_id, payload.speaker, payload.text, extra=payload.extra)
        return UtteranceResponse.model_validate(utterance)


@router.post("/{session_id}/close", response_model=SessionResponse)
async def close_session_endpoint(session_id: uuid.UUID, request: Request) -> SessionResponse:
    session_factory = request.app.state.db_session_factory
    async with session_factory() as db:
        session = await crud.close_session(db, session_id)
        if session is None:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found.")
        return SessionResponse.model_validate(session)


@router.get("/{session_id}", response_model=SessionReplayResponse)
async def replay_session_endpoint(session_id: uuid.UUID, request: Request) -> SessionReplayResponse:
    session_factory = request.app.state.db_session_factory
    async with session_factory() as db:
        session = await crud.get_session_with_utterances(db, session_id)
        if session is None:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found.")
        return SessionReplayResponse(
            session=SessionResponse.model_validate(session),
            utterances=[UtteranceResponse.model_validate(u) for u in session.utterances],
        )
