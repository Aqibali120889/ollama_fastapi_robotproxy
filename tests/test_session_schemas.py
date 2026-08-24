import uuid

import pytest
from pydantic import ValidationError

from app.schemas import AppendUtteranceRequest, CreateSessionRequest, SessionResponse, UtteranceResponse


def test_create_session_request_defaults_source_to_voice():
    req = CreateSessionRequest()
    assert req.source == "voice"


def test_append_utterance_request_rejects_invalid_speaker():
    with pytest.raises(ValidationError):
        AppendUtteranceRequest(speaker="robotic", text="hello")


def test_append_utterance_request_rejects_empty_text():
    with pytest.raises(ValidationError):
        AppendUtteranceRequest(speaker="human", text="")


def test_append_utterance_request_accepts_valid_payload():
    req = AppendUtteranceRequest(speaker="robot", text="Hello there.", extra={"used_rag": True})
    assert req.speaker == "robot"
    assert req.extra == {"used_rag": True}


def test_session_response_parses_from_orm_like_object():
    class _FakeSession:
        id = uuid.uuid4()
        created_at = "2026-01-01T00:00:00+00:00"
        closed_at = None
        source = "voice"

    parsed = SessionResponse.model_validate(_FakeSession())
    assert parsed.source == "voice"
    assert parsed.closed_at is None


def test_utterance_response_requires_valid_speaker_free_text_is_fine():
    # UtteranceResponse.speaker is a plain str (DB is the source of truth via
    # the CHECK constraint) -- only the *request* schema restricts to the
    # Literal, so a response round-trip for either value must parse cleanly.
    resp = UtteranceResponse(
        id=1,
        session_id=uuid.uuid4(),
        speaker="human",
        timestamp="2026-01-01T00:00:00+00:00",
        text="hi",
    )
    assert resp.speaker == "human"
