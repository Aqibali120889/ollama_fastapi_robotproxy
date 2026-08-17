import asyncio

import pytest

from app import voice_pipeline
from app.config import get_settings


def test_split_sentence_chunks_flushes_on_period():
    buffer, chunk = voice_pipeline.split_sentence_chunks("Hello world.", "")
    assert chunk == "Hello world."
    assert buffer == ""


def test_split_sentence_chunks_accumulates_until_terminator():
    buffer, chunk = voice_pipeline.split_sentence_chunks("Hello ", "")
    assert chunk is None
    assert buffer == "Hello "

    buffer, chunk = voice_pipeline.split_sentence_chunks("world", buffer)
    assert chunk is None
    assert buffer == "Hello world"

    buffer, chunk = voice_pipeline.split_sentence_chunks("!", buffer)
    assert chunk == "Hello world!"
    assert buffer == ""


def test_split_sentence_chunks_flushes_on_newline():
    buffer, chunk = voice_pipeline.split_sentence_chunks("line one\n", "")
    assert chunk == "line one"
    assert buffer == ""


def test_split_sentence_chunks_flushes_on_length_safety_valve():
    long_token = "a" * 250
    buffer, chunk = voice_pipeline.split_sentence_chunks(long_token, "")
    assert chunk == long_token
    assert buffer == ""


def test_split_sentence_chunks_never_emits_a_blank_chunk():
    # newline is a flush trigger but also whitespace -- "   " + "\n" strips to
    # "", which must not be emitted as a chunk.
    buffer, chunk = voice_pipeline.split_sentence_chunks("\n", "   ")
    assert chunk is None
    assert buffer == ""


class _FakeMemoryStore:
    def __init__(self):
        self.appended = []

    def get(self, session_id):
        return []

    def append(self, session_id, messages, max_turns):
        self.appended.append((session_id, messages, max_turns))


async def test_run_voice_turn_streams_one_event_pair_per_sentence(monkeypatch):
    fake_memory = _FakeMemoryStore()
    monkeypatch.setattr(voice_pipeline, "memory_store", fake_memory)
    monkeypatch.setattr(voice_pipeline, "denoise", lambda audio, sr: audio)
    monkeypatch.setattr(voice_pipeline, "transcribe", lambda *a, **k: "what does the robot do")

    async def fake_prepare_context(client, settings, question, top_k):
        return {"prompt": "PROMPT", "used_rag": True, "sources": ["kb.md"]}

    monkeypatch.setattr(voice_pipeline, "_prepare_context", fake_prepare_context)

    async def fake_stream_chat_request(settings, messages):
        for token in ["Hello world.", " Second sentence."]:
            yield token

    monkeypatch.setattr(voice_pipeline, "stream_chat_request", fake_stream_chat_request)
    monkeypatch.setattr(voice_pipeline, "synthesize", lambda text, *a, **k: (f"AUDIO:{text}".encode(), 24000))

    events = []
    audio_chunks = []

    async def on_event(payload):
        events.append(payload)

    async def on_audio(data):
        audio_chunks.append(data)

    settings = get_settings()
    await voice_pipeline.run_voice_turn(settings, "session-1", b"\x00\x01", on_event, on_audio)

    assert [e["type"] for e in events] == [
        "transcript",
        "meta",
        "sentence_audio_meta",
        "sentence_audio_meta",
        "done",
    ]
    assert events[0]["text"] == "what does the robot do"
    assert events[1]["used_rag"] is True
    assert events[1]["sources"] == ["kb.md"]
    assert events[2]["index"] == 0
    assert events[2]["text"] == "Hello world."
    assert events[3]["index"] == 1
    assert events[3]["text"] == "Second sentence."
    assert audio_chunks == [b"AUDIO:Hello world.", b"AUDIO:Second sentence."]
    assert events[-1]["answer"] == "Hello world. Second sentence."
    assert events[-1]["session_id"] == "session-1"
    assert fake_memory.appended, "the full turn should be recorded in session memory"


async def test_run_voice_turn_no_speech_detected_skips_generation(monkeypatch):
    monkeypatch.setattr(voice_pipeline, "denoise", lambda audio, sr: audio)
    monkeypatch.setattr(voice_pipeline, "transcribe", lambda *a, **k: "   ")

    events = []

    async def on_event(payload):
        events.append(payload)

    async def on_audio(data):
        raise AssertionError("should never synthesize when no speech was detected")

    settings = get_settings()
    await voice_pipeline.run_voice_turn(settings, "session-1", b"\x00\x01", on_event, on_audio)

    assert events == [{"type": "error", "message": "No speech detected."}]


async def test_run_voice_turn_propagates_generation_error_without_hanging(monkeypatch):
    """Regression test for the producer/consumer queue: if stream_chat_request
    raises mid-stream, the consumer must still be told to stop (via the
    finally-guaranteed sentinel) rather than waiting on an empty queue
    forever. asyncio.wait_for's timeout turns a hang into a loud failure
    instead of pytest blocking indefinitely."""
    monkeypatch.setattr(voice_pipeline, "denoise", lambda audio, sr: audio)
    monkeypatch.setattr(voice_pipeline, "transcribe", lambda *a, **k: "question")

    async def fake_prepare_context(client, settings, question, top_k):
        return {"prompt": "PROMPT", "used_rag": False, "sources": []}

    monkeypatch.setattr(voice_pipeline, "_prepare_context", fake_prepare_context)

    async def fake_stream_chat_request(settings, messages):
        yield "partial sentence"
        raise RuntimeError("boom")

    monkeypatch.setattr(voice_pipeline, "stream_chat_request", fake_stream_chat_request)

    async def on_event(payload):
        pass

    async def on_audio(data):
        pass

    settings = get_settings()
    with pytest.raises(RuntimeError, match="boom"):
        await asyncio.wait_for(
            voice_pipeline.run_voice_turn(settings, "session-1", b"\x00", on_event, on_audio),
            timeout=5,
        )
