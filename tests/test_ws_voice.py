import json

import pytest

pytest.importorskip("langgraph")  # /ws/voice pulls in app.main -> app.conversation_graph -> langgraph

from fastapi.testclient import TestClient  # noqa: E402

import app.main as main_module  # noqa: E402


@pytest.fixture(autouse=True)
def _patch_voice_pipeline(monkeypatch):
    async def fake_conversation_ainvoke(state):
        return {
            "response_text": "This is a test answer.",
            "used_rag": True,
            "sources": ["kb.md"],
        }

    monkeypatch.setattr(main_module.conversation_graph, "ainvoke", fake_conversation_ainvoke)
    monkeypatch.setattr(main_module, "denoise", lambda audio, sr: audio)
    monkeypatch.setattr(main_module, "transcribe", lambda *a, **k: "what does the robot do")
    monkeypatch.setattr(main_module, "synthesize", lambda *a, **k: (b"\x00\x01" * 10, 24000))


def test_ws_voice_full_round_trip():
    client = TestClient(main_module.app)
    with client.websocket_connect("/ws/voice") as ws:
        session_msg = json.loads(ws.receive_text())
        assert session_msg["type"] == "session"
        session_id = session_msg["session_id"]

        ws.send_bytes(b"\x00\x01" * 100)

        transcript_msg = json.loads(ws.receive_text())
        assert transcript_msg == {"type": "transcript", "text": "what does the robot do"}

        meta_msg = json.loads(ws.receive_text())
        assert meta_msg["type"] == "meta"
        assert meta_msg["used_rag"] is True
        assert meta_msg["sources"] == ["kb.md"]
        assert meta_msg["answer"] == "This is a test answer."

        audio_meta_msg = json.loads(ws.receive_text())
        assert audio_meta_msg["type"] == "audio_meta"
        assert audio_meta_msg["sample_rate"] == 24000

        audio_bytes = ws.receive_bytes()
        assert audio_bytes == b"\x00\x01" * 10

        done_msg = json.loads(ws.receive_text())
        assert done_msg["type"] == "done"
        assert done_msg["session_id"] == session_id


def test_ws_voice_no_speech_detected(monkeypatch):
    monkeypatch.setattr(main_module, "transcribe", lambda *a, **k: "   ")
    client = TestClient(main_module.app)
    with client.websocket_connect("/ws/voice") as ws:
        ws.receive_text()  # session
        ws.send_bytes(b"\x00\x01" * 100)
        error_msg = json.loads(ws.receive_text())
        assert error_msg == {"type": "error", "message": "No speech detected."}


def test_ws_voice_reuses_one_session_id_across_utterances():
    client = TestClient(main_module.app)
    with client.websocket_connect("/ws/voice") as ws:
        session_msg = json.loads(ws.receive_text())
        session_id = session_msg["session_id"]

        for _ in range(2):
            ws.send_bytes(b"\x00\x01" * 100)
            ws.receive_text()  # transcript
            ws.receive_text()  # meta
            ws.receive_text()  # audio_meta
            ws.receive_bytes()  # audio
            done_msg = json.loads(ws.receive_text())
            assert done_msg["session_id"] == session_id
