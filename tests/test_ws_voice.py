import json

import pytest

pytest.importorskip("langgraph")  # /ws/voice pulls in app.main -> app.conversation_graph -> langgraph

from fastapi.testclient import TestClient  # noqa: E402

import app.main as main_module  # noqa: E402
import app.voice_pipeline as voice_pipeline_module  # noqa: E402


async def _single_sentence_stream(settings, messages):
    yield "This is a test answer."


@pytest.fixture(autouse=True)
def _patch_voice_pipeline(monkeypatch):
    # run_voice_turn (imported once into app.main) reads these names from
    # app.voice_pipeline's own module globals at call time, so patching them
    # here -- not on main_module -- is what actually takes effect.
    monkeypatch.setattr(voice_pipeline_module, "denoise", lambda audio, sr: audio)
    monkeypatch.setattr(voice_pipeline_module, "transcribe", lambda *a, **k: "what does the robot do")

    async def fake_prepare_context(client, settings, question, top_k):
        return {"prompt": "PROMPT", "used_rag": True, "sources": ["kb.md"]}

    monkeypatch.setattr(voice_pipeline_module, "_prepare_context", fake_prepare_context)
    monkeypatch.setattr(voice_pipeline_module, "stream_chat_request", _single_sentence_stream)
    monkeypatch.setattr(voice_pipeline_module, "synthesize", lambda text, *a, **k: (b"\x00\x01" * 10, 24000))


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

        sentence_meta_msg = json.loads(ws.receive_text())
        assert sentence_meta_msg["type"] == "sentence_audio_meta"
        assert sentence_meta_msg["index"] == 0
        assert sentence_meta_msg["text"] == "This is a test answer."
        assert sentence_meta_msg["sample_rate"] == 24000

        audio_bytes = ws.receive_bytes()
        assert audio_bytes == b"\x00\x01" * 10

        done_msg = json.loads(ws.receive_text())
        assert done_msg["type"] == "done"
        assert done_msg["session_id"] == session_id
        assert done_msg["answer"] == "This is a test answer."


def test_ws_voice_streams_one_chunk_per_sentence_in_order(monkeypatch):
    async def multi_sentence_stream(settings, messages):
        for token in ["First sentence.", " Second sentence."]:
            yield token

    monkeypatch.setattr(voice_pipeline_module, "stream_chat_request", multi_sentence_stream)

    client = TestClient(main_module.app)
    with client.websocket_connect("/ws/voice") as ws:
        ws.receive_text()  # session
        ws.send_bytes(b"\x00\x01" * 100)
        ws.receive_text()  # transcript
        ws.receive_text()  # meta

        first_meta = json.loads(ws.receive_text())
        assert first_meta["index"] == 0
        assert first_meta["text"] == "First sentence."
        ws.receive_bytes()

        second_meta = json.loads(ws.receive_text())
        assert second_meta["index"] == 1
        assert second_meta["text"] == "Second sentence."
        ws.receive_bytes()

        done_msg = json.loads(ws.receive_text())
        assert done_msg["type"] == "done"
        assert done_msg["answer"] == "First sentence. Second sentence."


def test_ws_voice_no_speech_detected(monkeypatch):
    monkeypatch.setattr(voice_pipeline_module, "transcribe", lambda *a, **k: "   ")
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
            ws.receive_text()  # sentence_audio_meta
            ws.receive_bytes()  # audio
            done_msg = json.loads(ws.receive_text())
            assert done_msg["session_id"] == session_id
