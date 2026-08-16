import asyncio
import json
import logging
import time
import uuid
from typing import Dict

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.conversation_graph import build_conversation_graph
from app.ollama_proxy import OllamaProxyError, check_ollama_reachability, detect_cuda
from app.rag_pipeline import rag_answer, rag_answer_stream
from app.schemas import ChatRequest, ChatResponse, HealthResponse, RagRequest, RagResponse
from app.voice_denoise import denoise
from app.voice_stt import transcribe
from app.voice_tts import synthesize

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

settings = get_settings()
app = FastAPI(title=settings.service_name)
conversation_graph = build_conversation_graph(settings)


@app.get("/health", response_model=HealthResponse)
async def health() -> Dict[str, object]:
    local = await check_ollama_reachability(settings.ollama_local_base_url, settings.health_timeout_seconds)
    remote_checks = [
        await check_ollama_reachability(url, settings.health_timeout_seconds) for url in settings.remote_urls()
    ]

    return {
        "status": "ok",
        "service": settings.service_name,
        "models": {
            "remote_primary": settings.remote_primary_model,
            "local_fallback": settings.local_fallback_model,
            "code_model": settings.code_model,
            "embedding_model": settings.embedding_model,
            "vision_model": settings.vision_model,
        },
        "cuda": detect_cuda(),
        "ollama": {
            "local": local.model_dump(),
            "remote": [item.model_dump() for item in remote_checks],
        },
    }


@app.post("/chat", response_model=ChatResponse)
@app.post("/agent", response_model=ChatResponse)
async def chat(payload: ChatRequest) -> ChatResponse:
    session_id = payload.session_id or str(uuid.uuid4())
    try:
        question = payload.prompt or (payload.messages[-1].content if payload.messages else "")
        state = await conversation_graph.ainvoke(
            {
                "session_id": session_id,
                "user_question": question,
                "incoming_messages": payload.messages or [],
                "mode": payload.mode,
                "model": payload.model,
                "top_k": settings.rag_top_k,
                "metadata_filters": payload.metadata_filters,
            }
        )
        return ChatResponse(
            model=state.get("response_model", settings.remote_primary_model),
            output=state.get("response_text", ""),
            source=state.get("response_source", "unknown"),
            session_id=state.get("session_id", session_id),
            used_rag=state.get("used_rag", False),
            timings=state.get("timings", {}),
            sources=state.get("sources", []),
        )
    except OllamaProxyError as exc:
        logger.exception("Chat routing failed")
        raise HTTPException(
            status_code=503,
            detail={
                "message": "Unable to get response from configured Ollama routes.",
                "error": str(exc),
                "attempts": exc.attempts,
            },
        ) from exc
    except Exception as exc:
        logger.exception("Conversation orchestration failed")
        raise HTTPException(
            status_code=503,
            detail={
                "message": "Conversation orchestration failed.",
                "error": str(exc),
                "hint": "Check Ollama endpoints and knowledge-base indexing status.",
            },
        ) from exc


@app.post("/rag/ask", response_model=RagResponse)
async def rag_ask(payload: RagRequest) -> RagResponse:
    try:
        result = await rag_answer(
            settings,
            payload.question,
            top_k=payload.top_k,
            session_id=payload.session_id,
            metadata_filters=payload.metadata_filters,
        )
        return RagResponse(**result)
    except Exception as exc:
        logger.exception("RAG query failed")
        raise HTTPException(
            status_code=503,
            detail={
                "message": "RAG pipeline failed.",
                "error": str(exc),
                "hint": "Ensure Ollama embedding/chat endpoints are healthy and run scripts/ingest_kb.py.",
            },
        ) from exc


@app.websocket("/ws/rag")
async def ws_rag(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            question = await websocket.receive_text()
            if not question.strip():
                continue
            try:
                async for event in rag_answer_stream(settings, question):
                    await websocket.send_text(json.dumps(event))
            except Exception as exc:
                logger.exception("Streaming RAG query failed")
                await websocket.send_text(json.dumps({"type": "error", "message": str(exc)}))
    except WebSocketDisconnect:
        pass


@app.websocket("/ws/voice")
async def ws_voice(websocket: WebSocket):
    """Audio-in -> denoise -> Whisper -> RAG/LLM -> XTTS-v2 -> audio-out.

    One session_id is minted for the lifetime of the connection and reused for
    every utterance, so conversation_graph's SessionMemoryStore gives multi-turn
    continuity for free -- unlike /ws/rag, which mints a fresh session per message.
    """
    await websocket.accept()
    session_id = str(uuid.uuid4())
    await websocket.send_text(json.dumps({"type": "session", "session_id": session_id}))

    try:
        while True:
            audio_bytes = await websocket.receive_bytes()
            if not audio_bytes:
                continue

            total_start = time.perf_counter()
            timings: Dict[str, float] = {}

            try:
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
                    await websocket.send_text(json.dumps({"type": "error", "message": "No speech detected."}))
                    continue

                await websocket.send_text(json.dumps({"type": "transcript", "text": question}))

                t0 = time.perf_counter()
                state = await conversation_graph.ainvoke(
                    {
                        "session_id": session_id,
                        "user_question": question,
                        "incoming_messages": [],
                        "mode": "chat",
                        "model": None,
                        "top_k": settings.rag_top_k,
                        "metadata_filters": None,
                    }
                )
                timings["rag_generation_s"] = round(time.perf_counter() - t0, 3)

                answer_text = state.get("response_text", "")
                await websocket.send_text(
                    json.dumps(
                        {
                            "type": "meta",
                            "used_rag": state.get("used_rag", False),
                            "sources": state.get("sources", []),
                            "answer": answer_text,
                        }
                    )
                )

                t0 = time.perf_counter()
                audio_out, sample_rate = await asyncio.to_thread(
                    synthesize,
                    answer_text,
                    settings.tts_model_name,
                    settings.tts_device,
                    settings.tts_speaker,
                    settings.tts_language,
                )
                timings["tts_s"] = round(time.perf_counter() - t0, 3)
                timings["total_s"] = round(time.perf_counter() - total_start, 3)

                await websocket.send_text(
                    json.dumps({"type": "audio_meta", "sample_rate": sample_rate, "timings": timings})
                )
                if audio_out:
                    await websocket.send_bytes(audio_out)
                await websocket.send_text(json.dumps({"type": "done", "session_id": session_id, "timings": timings}))
            except Exception as exc:
                logger.exception("Voice pipeline stage failed")
                await websocket.send_text(json.dumps({"type": "error", "message": str(exc)}))
    except WebSocketDisconnect:
        pass


@app.exception_handler(HTTPException)
async def http_exception_handler(_, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})
