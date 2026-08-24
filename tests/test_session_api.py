import pytest

pytest.importorskip("langgraph")  # app.main -> app.conversation_graph -> langgraph

import httpx  # noqa: E402
from httpx import ASGITransport  # noqa: E402

import app.main as main_module  # noqa: E402
from app.config import get_settings  # noqa: E402

FAKE_SESSION_ID = "00000000-0000-0000-0000-000000000000"


@pytest.fixture
async def _database_available():
    """Live check: skips gracefully if PostgreSQL/TimescaleDB isn't reachable
    (e.g. this sandbox), mirroring tests/test_rag_accuracy.py's
    _ollama_available pattern. Meaningful once run against the real `db`
    docker-compose service on the workstation."""
    import asyncpg

    settings = get_settings()
    dsn = settings.database_url.replace("postgresql+asyncpg://", "postgresql://")
    try:
        conn = await asyncpg.connect(dsn, timeout=settings.database_connect_timeout_seconds)
        await conn.close()
    except Exception:
        pytest.skip("PostgreSQL/TimescaleDB is not reachable from this environment; skipping session-API tests.")


def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=ASGITransport(app=main_module.app), base_url="http://test")


@pytest.mark.integration
async def test_session_lifecycle_create_append_close_replay(_database_available):
    async with _client() as client:
        create_resp = await client.post("/api/sessions", json={"source": "voice"})
        assert create_resp.status_code == 201
        session = create_resp.json()
        session_id = session["id"]
        assert session["closed_at"] is None
        assert session["source"] == "voice"

        human_resp = await client.post(
            f"/api/sessions/{session_id}/utterances",
            json={"speaker": "human", "text": "What does the robot do?"},
        )
        assert human_resp.status_code == 201
        assert human_resp.json()["speaker"] == "human"

        robot_resp = await client.post(
            f"/api/sessions/{session_id}/utterances",
            json={
                "speaker": "robot",
                "text": "It answers questions about the project.",
                "extra": {"used_rag": True, "sources": ["project_overview.md"]},
            },
        )
        assert robot_resp.status_code == 201

        close_resp = await client.post(f"/api/sessions/{session_id}/close")
        assert close_resp.status_code == 200
        assert close_resp.json()["closed_at"] is not None

        replay_resp = await client.get(f"/api/sessions/{session_id}")
        assert replay_resp.status_code == 200
        replay = replay_resp.json()

        assert replay["session"]["id"] == session_id
        assert replay["session"]["closed_at"] is not None
        assert len(replay["utterances"]) == 2

        first, second = replay["utterances"]
        assert first["speaker"] == "human"
        assert first["text"] == "What does the robot do?"
        assert second["speaker"] == "robot"
        assert second["extra"] == {"used_rag": True, "sources": ["project_overview.md"]}
        assert first["timestamp"] <= second["timestamp"]


@pytest.mark.integration
async def test_append_utterance_to_unknown_session_returns_404(_database_available):
    async with _client() as client:
        resp = await client.post(
            f"/api/sessions/{FAKE_SESSION_ID}/utterances", json={"speaker": "human", "text": "hi"}
        )
        assert resp.status_code == 404


@pytest.mark.integration
async def test_close_unknown_session_returns_404(_database_available):
    async with _client() as client:
        resp = await client.post(f"/api/sessions/{FAKE_SESSION_ID}/close")
        assert resp.status_code == 404


@pytest.mark.integration
async def test_replay_unknown_session_returns_404(_database_available):
    async with _client() as client:
        resp = await client.get(f"/api/sessions/{FAKE_SESSION_ID}")
        assert resp.status_code == 404


@pytest.mark.integration
async def test_append_utterance_rejects_invalid_speaker(_database_available):
    async with _client() as client:
        create_resp = await client.post("/api/sessions", json={})
        session_id = create_resp.json()["id"]

        resp = await client.post(
            f"/api/sessions/{session_id}/utterances", json={"speaker": "alien", "text": "hi"}
        )
        assert resp.status_code == 422


@pytest.mark.integration
async def test_append_utterance_rejects_empty_text(_database_available):
    async with _client() as client:
        create_resp = await client.post("/api/sessions", json={})
        session_id = create_resp.json()["id"]

        resp = await client.post(f"/api/sessions/{session_id}/utterances", json={"speaker": "human", "text": ""})
        assert resp.status_code == 422
