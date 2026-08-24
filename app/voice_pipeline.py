import asyncio
import logging
import time
from typing import Awaitable, Callable, Dict, List, Optional, Tuple

import httpx

from app.config import Settings
from app.conversation_graph import memory_store
from app.db.safe_logging import SessionFactory, log_utterance_safe
from app.ollama_proxy import stream_chat_request
from app.rag_pipeline import _prepare_context
from app.schemas import ChatMessage
from app.voice_denoise import denoise
from app.voice_stt import transcribe
from app.voice_tts import synthesize

logger = logging.getLogger(__name__)

OnEvent = Callable[[dict], Awaitable[None]]
OnAudio = Callable[[bytes], Awaitable[None]]

_SENTENCE_END_CHARS = (".", "!", "?", "\n")


def split_sentence_chunks(token: str, buffer: str, max_buffer_len: int = 220) -> Tuple[str, Optional[str]]:
    """Feeds one streamed token into the running sentence buffer. Returns the
    updated buffer and a completed sentence (or None if the buffer isn't done
    yet). Flushes on sentence-ending punctuation/newline, or once the buffer
    gets long enough that waiting for real punctuation risks never flushing
    (e.g. a long run-on answer)."""
    buffer += token
    if any(p in token for p in _SENTENCE_END_CHARS) or len(buffer) > max_buffer_len:
        chunk = buffer.strip()
        return "", (chunk or None)
    return buffer, None


async def run_voice_turn(
    settings: Settings,
    session_id: str,
    audio_bytes: bytes,
    on_event: OnEvent,
    on_audio: OnAudio,
    db_session_factory: Optional[SessionFactory] = None,
) -> None:
    """Denoise -> transcribe -> RAG context -> streamed generation, synthesizing
    and emitting each sentence as soon as it's ready instead of waiting for the
    full answer. on_event(dict) sends a JSON control message; on_audio(bytes)
    sends one chunk's synthesized audio. Both are transport-agnostic so this
    can be unit-tested with plain recording fakes instead of a real socket.
    db_session_factory is optional and defaults to None so existing callers/
    tests that don't care about persistence are unaffected; when provided,
    both utterances are logged via the non-raising app.db.safe_logging
    wrappers -- a Postgres outage must never break a live voice turn."""
    total_start = time.perf_counter()
    timings: Dict[str, float] = {}

    t0 = time.perf_counter()
    if settings.enable_denoise:
        audio_bytes = await asyncio.to_thread(denoise, audio_bytes, settings.voice_sample_rate)
    timings["denoise_s"] = round(time.perf_counter() - t0, 3)

    t0 = time.perf_counter()
    question = await asyncio.to_thread(
        transcribe,
        audio_bytes,
        settings.stt_model_size,
        settings.stt_device,
        settings.stt_compute_type,
        settings.stt_language,
    )
    timings["stt_s"] = round(time.perf_counter() - t0, 3)

    if not question.strip():
        await on_event({"type": "error", "message": "No speech detected."})
        return

    await on_event({"type": "transcript", "text": question})
    await log_utterance_safe(db_session_factory, session_id, "human", question)

    t0 = time.perf_counter()
    async with httpx.AsyncClient(timeout=settings.request_timeout_seconds) as client:
        ctx = await _prepare_context(client, settings, question, settings.rag_top_k)
    timings["retrieval_s"] = round(time.perf_counter() - t0, 3)

    await on_event({"type": "meta", "used_rag": ctx["used_rag"], "sources": ctx["sources"]})

    history = memory_store.get(session_id)
    messages: List[Dict[str, str]] = [
        {
            "role": "system",
            "content": (
                "Use the user and assistant messages exactly as conversation context. "
                "Ground answers in provided context if present."
            ),
        },
        *[{"role": m.role, "content": m.content} for m in history],
        {"role": "user", "content": ctx["prompt"]},
    ]

    sentence_queue: "asyncio.Queue[Optional[str]]" = asyncio.Queue()
    full_answer_parts: List[str] = []
    first_audio_recorded = False

    async def _produce() -> None:
        buffer = ""
        try:
            async for token in stream_chat_request(settings, messages):
                full_answer_parts.append(token)
                buffer, chunk = split_sentence_chunks(token, buffer)
                if chunk:
                    await sentence_queue.put(chunk)
            leftover = buffer.strip()
            if leftover:
                await sentence_queue.put(leftover)
        finally:
            # Always signal the consumer to stop, even if streaming raised --
            # otherwise it would wait on an empty queue forever.
            await sentence_queue.put(None)

    async def _consume() -> None:
        nonlocal first_audio_recorded
        index = 0
        while True:
            chunk = await sentence_queue.get()
            if chunk is None:
                break
            audio_out, sample_rate = await asyncio.to_thread(
                synthesize,
                chunk,
                settings.tts_model_name,
                settings.tts_device,
                settings.tts_speaker,
                settings.tts_language,
            )
            if not first_audio_recorded:
                timings["time_to_first_audio_s"] = round(time.perf_counter() - total_start, 3)
                first_audio_recorded = True
            await on_event({"type": "sentence_audio_meta", "index": index, "text": chunk, "sample_rate": sample_rate})
            if audio_out:
                await on_audio(audio_out)
            index += 1

    t0 = time.perf_counter()
    await asyncio.gather(_produce(), _consume())
    timings["generation_and_tts_s"] = round(time.perf_counter() - t0, 3)

    answer_text = "".join(full_answer_parts).strip()
    memory_store.append(
        session_id,
        [ChatMessage(role="user", content=question), ChatMessage(role="assistant", content=answer_text)],
        max_turns=settings.memory_max_turns,
    )
    await log_utterance_safe(
        db_session_factory,
        session_id,
        "robot",
        answer_text,
        extra={"used_rag": ctx["used_rag"], "sources": ctx["sources"]},
    )

    timings["total_s"] = round(time.perf_counter() - total_start, 3)
    logger.info("voice turn timings: %s", timings)
    await on_event({"type": "done", "session_id": session_id, "answer": answer_text, "timings": timings})
