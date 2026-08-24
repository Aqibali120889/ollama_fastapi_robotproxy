from app.db import safe_logging


class _FakeAsyncSessionCM:
    async def __aenter__(self):
        return object()

    async def __aexit__(self, *exc):
        return False


class _FakeSessionFactory:
    def __call__(self):
        return _FakeAsyncSessionCM()


async def test_no_op_when_session_factory_is_none():
    assert await safe_logging.create_session_safe(None, "s1") is None
    assert await safe_logging.log_utterance_safe(None, "s1", "human", "hi") is None
    assert await safe_logging.close_session_safe(None, "s1") is None


async def test_log_utterance_no_op_on_blank_text(monkeypatch):
    calls = []

    async def fake_append_utterance(db, session_id, speaker, text, extra=None):
        calls.append((session_id, speaker, text))

    monkeypatch.setattr(safe_logging.crud, "append_utterance", fake_append_utterance)

    task = await safe_logging.log_utterance_safe(_FakeSessionFactory(), "s1", "human", "   ")
    assert task is None
    assert calls == []


async def test_create_session_calls_crud_and_succeeds(monkeypatch):
    calls = []

    async def fake_create_session(db, session_id, source="voice"):
        calls.append((session_id, source))

    monkeypatch.setattr(safe_logging.crud, "create_session", fake_create_session)

    task = await safe_logging.create_session_safe(_FakeSessionFactory(), "11111111-1111-1111-1111-111111111111", source="voice")
    await task  # deterministically wait for the background task instead of racing it

    assert len(calls) == 1
    assert str(calls[0][0]) == "11111111-1111-1111-1111-111111111111"
    assert calls[0][1] == "voice"


async def test_log_utterance_swallows_db_errors_without_raising(monkeypatch, caplog):
    async def raising_append_utterance(db, session_id, speaker, text, extra=None):
        raise RuntimeError("connection refused")

    monkeypatch.setattr(safe_logging.crud, "append_utterance", raising_append_utterance)

    task = await safe_logging.log_utterance_safe(
        _FakeSessionFactory(), "11111111-1111-1111-1111-111111111111", "robot", "hello there"
    )
    await task  # must complete without raising, despite the underlying failure

    assert "Failed to persist robot utterance" in caplog.text


async def test_close_session_swallows_db_errors_without_raising(monkeypatch):
    async def raising_close_session(db, session_id):
        raise RuntimeError("timeout")

    monkeypatch.setattr(safe_logging.crud, "close_session", raising_close_session)

    task = await safe_logging.close_session_safe(_FakeSessionFactory(), "11111111-1111-1111-1111-111111111111")
    await task  # no exception should propagate
