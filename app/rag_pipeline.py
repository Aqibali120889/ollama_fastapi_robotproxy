import json
import logging
import time
import uuid
from difflib import SequenceMatcher
from typing import AsyncGenerator, Dict, List, Optional, Union

import httpx

from app.config import Settings
from app.hybrid_retriever import hybrid_retrieve
from app.ollama_proxy import route_chat_request
from app.rag_logging import log_rag_event
from app.rag_store import embed_text, get_collection
from app.schemas import ChatMessage, ChatRequest

logger = logging.getLogger(__name__)


def _keep_alive_value(raw: str) -> Union[int, str]:
    try:
        return int(raw)
    except ValueError:
        return raw


RAG_PROMPT_TEMPLATE = """You are an AI assistant representing a company at a business expo booth. \
Answer the visitor's question using ONLY the context below. If the answer is not contained in the \
context, say you don't have that information and offer to connect them with a team member. Keep the \
answer concise and professional (2-4 sentences).

Context:
{context}

Visitor question: {question}

Answer:"""

REWRITE_PROMPT_TEMPLATE = """Rewrite the visitor's question as a short, clear search query using the \
kind of formal terminology likely to appear in company documentation. Return ONLY the rewritten \
query, with no extra commentary or punctuation.

Visitor question: {question}

Rewritten query:"""

DIRECT_CHAT_TEMPLATE = """You are a friendly AI assistant for a robot representing a company at a \
business expo booth. The visitor's message does not relate to specific company information, so reply \
naturally and conversationally without inventing any company facts. Keep it brief (1-2 sentences).

Visitor: {question}

Response:"""


def _dedupe_and_cap_context(
    settings: Settings,
    documents: List[str],
    metadatas: List[Dict[str, object]],
    distances: List[float],
    fusion_scores: List[float],
) -> Dict[str, List[object]]:
    selected_docs: List[str] = []
    selected_meta: List[Dict[str, object]] = []
    selected_distances: List[float] = []
    selected_fusion: List[float] = []
    char_budget = 0

    for doc, metadata, distance, fusion_score in zip(documents, metadatas, distances, fusion_scores):
        if any(
            SequenceMatcher(None, doc, existing).ratio() >= settings.rag_dedupe_similarity_threshold
            for existing in selected_docs
        ):
            continue

        next_budget = char_budget + len(doc)
        if next_budget > settings.rag_context_max_chars or len(selected_docs) >= settings.rag_context_max_chunks:
            break

        selected_docs.append(doc)
        selected_meta.append(metadata)
        selected_distances.append(distance)
        selected_fusion.append(fusion_score)
        char_budget = next_budget

    return {
        "documents": selected_docs,
        "metadatas": selected_meta,
        "distances": selected_distances,
        "fusion_scores": selected_fusion,
    }


async def _rewrite_query(client: httpx.AsyncClient, settings: Settings, question: str) -> str:
    response = await client.post(
        f"{settings.ollama_local_base_url.rstrip('/')}/api/generate",
        json={
            "model": settings.local_fallback_model,
            "prompt": REWRITE_PROMPT_TEMPLATE.format(question=question),
            "stream": False,
            "keep_alive": _keep_alive_value(settings.ollama_keep_alive),
        },
    )
    response.raise_for_status()
    rewritten = response.json().get("response", "").strip()
    return rewritten or question


async def _prepare_context(
    client: httpx.AsyncClient,
    settings: Settings,
    question: str,
    top_k: int,
    metadata_filters: Optional[Dict[str, str]] = None,
) -> Dict[str, object]:
    timings: Dict[str, float] = {}
    collection = get_collection(settings.chroma_persist_dir, settings.chroma_collection_name)

    t0 = time.perf_counter()
    question_embedding = await embed_text(
        client, settings.ollama_local_base_url, settings.embedding_model, question
    )
    timings["embed_s"] = round(time.perf_counter() - t0, 3)

    t0 = time.perf_counter()
    initial = hybrid_retrieve(
        collection=collection,
        query_embedding=question_embedding,
        query_text=question,
        top_k=top_k,
        dense_weight=settings.rag_dense_weight,
        keyword_weight=settings.rag_keyword_weight,
        candidate_pool_size=top_k * settings.rag_candidate_pool_multiplier,
        min_fusion_score=settings.rag_min_fusion_score,
        metadata_filters=metadata_filters,
    )
    timings["retrieval_s"] = round(time.perf_counter() - t0, 3)

    routing_distances = initial["distances"]
    best_distance = min(routing_distances) if routing_distances else float("inf")
    used_rag = best_distance <= settings.rag_distance_threshold and bool(initial["documents"])

    search_query = question
    final = initial

    if used_rag and settings.enable_query_rewrite:
        t0 = time.perf_counter()
        search_query = await _rewrite_query(client, settings, question)
        timings["rewrite_s"] = round(time.perf_counter() - t0, 3)

        t0 = time.perf_counter()
        rewritten_embedding = await embed_text(
            client, settings.ollama_local_base_url, settings.embedding_model, search_query
        )
        timings["search_embed_s"] = round(time.perf_counter() - t0, 3)

        t0 = time.perf_counter()
        final = hybrid_retrieve(
            collection=collection,
            query_embedding=rewritten_embedding,
            query_text=search_query,
            top_k=top_k,
            dense_weight=settings.rag_dense_weight,
            keyword_weight=settings.rag_keyword_weight,
            candidate_pool_size=top_k * settings.rag_candidate_pool_multiplier,
            min_fusion_score=settings.rag_min_fusion_score,
            metadata_filters=metadata_filters,
        )
        timings["search_retrieval_s"] = round(time.perf_counter() - t0, 3)

    filtered = _dedupe_and_cap_context(
        settings=settings,
        documents=final["documents"],
        metadatas=final["metadatas"],
        distances=final["distances"],
        fusion_scores=final["fusion_scores"],
    )

    sources = [meta.get("source", "unknown") for meta in filtered["metadatas"]]

    if used_rag and filtered["documents"]:
        context = "\n\n---\n\n".join(filtered["documents"])
        prompt = RAG_PROMPT_TEMPLATE.format(context=context, question=question)
    else:
        used_rag = False
        prompt = DIRECT_CHAT_TEMPLATE.format(question=question)

    return {
        "prompt": prompt,
        "used_rag": used_rag,
        "search_query": search_query,
        "sources": sources,
        "retrieved_chunks": filtered["documents"],
        "distances": filtered["distances"],
        "fusion_scores": filtered["fusion_scores"],
        "metadatas": filtered["metadatas"],
        "best_distance": best_distance,
        "timings": timings,
    }


async def rag_answer(
    settings: Settings,
    question: str,
    top_k: Optional[int] = None,
    session_id: Optional[str] = None,
    metadata_filters: Optional[Dict[str, str]] = None,
) -> Dict[str, object]:
    top_k = top_k or settings.rag_top_k
    session_id = session_id or str(uuid.uuid4())
    total_start = time.perf_counter()

    try:
        async with httpx.AsyncClient(timeout=settings.request_timeout_seconds) as client:
            ctx = await _prepare_context(client, settings, question, top_k, metadata_filters=metadata_filters)
            timings = ctx["timings"]

            t0 = time.perf_counter()
            system_prompt = (
                "Use the user and assistant messages exactly as conversation context. "
                "Ground answers in provided context if present."
            )
            messages = [
                ChatMessage(role="system", content=system_prompt),
                ChatMessage(role="user", content=ctx["prompt"]),
            ]
            chat_payload = ChatRequest(messages=messages, mode="chat", session_id=session_id)
            answer, model, source_meta = await route_chat_request(settings, chat_payload)
            timings["generation_s"] = round(time.perf_counter() - t0, 3)

        timings["total_s"] = round(time.perf_counter() - total_start, 3)

        result = {
            "answer": answer,
            "session_id": session_id,
            "model": model,
            "source": source_meta[0]["source"] if source_meta else "unknown",
            "sources": ctx["sources"],
            "retrieved_metadata": [{k: str(v) for k, v in meta.items()} for meta in ctx["metadatas"]],
            "retrieved_chunks": ctx["retrieved_chunks"],
            "distances": ctx["distances"],
            "used_rag": ctx["used_rag"],
            "search_query": ctx["search_query"],
            "timings": timings,
        }
        _log(settings, question, ctx, answer, timings, session_id)
        return result
    except Exception as exc:
        logger.exception("RAG answer failed")
        raise RuntimeError(
            "RAG pipeline failed. Verify Ollama reachability and run scripts/ingest_kb.py to populate Chroma."
        ) from exc


async def rag_answer_stream(
    settings: Settings,
    question: str,
    top_k: Optional[int] = None,
    metadata_filters: Optional[Dict[str, str]] = None,
) -> AsyncGenerator[Dict[str, object], None]:
    top_k = top_k or settings.rag_top_k
    session_id = str(uuid.uuid4())
    total_start = time.perf_counter()

    async with httpx.AsyncClient(timeout=settings.request_timeout_seconds) as client:
        ctx = await _prepare_context(client, settings, question, top_k, metadata_filters=metadata_filters)
        timings = ctx["timings"]

        yield {
            "type": "meta",
            "session_id": session_id,
            "used_rag": ctx["used_rag"],
            "sources": ctx["sources"],
            "search_query": ctx["search_query"],
        }

        t0 = time.perf_counter()
        answer_parts: List[str] = []

        if ctx["used_rag"]:
            prompt = ctx["prompt"]
        else:
            prompt = DIRECT_CHAT_TEMPLATE.format(question=question)

        try:
            async with client.stream(
                "POST",
                f"{settings.ollama_local_base_url.rstrip('/')}/api/generate",
                json={
                    "model": settings.remote_primary_model,
                    "prompt": prompt,
                    "stream": True,
                    "keep_alive": _keep_alive_value(settings.ollama_keep_alive),
                },
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    chunk = json.loads(line)
                    token = chunk.get("response", "")
                    if token:
                        answer_parts.append(token)
                        yield {"type": "token", "text": token}
                    if chunk.get("done"):
                        break
        except Exception:
            fallback = await rag_answer(
                settings=settings,
                question=question,
                top_k=top_k,
                session_id=session_id,
                metadata_filters=metadata_filters,
            )
            answer = fallback["answer"]
            timings.update(fallback.get("timings", {}))
            yield {"type": "token", "text": answer}
            yield {
                "type": "done",
                "session_id": session_id,
                "answer": answer,
                "used_rag": fallback["used_rag"],
                "sources": fallback["sources"],
                "timings": timings,
            }
            return

        timings["generation_s"] = round(time.perf_counter() - t0, 3)

    timings["total_s"] = round(time.perf_counter() - total_start, 3)
    answer = "".join(answer_parts).strip()

    _log(settings, question, ctx, answer, timings, session_id)

    yield {
        "type": "done",
        "session_id": session_id,
        "answer": answer,
        "used_rag": ctx["used_rag"],
        "sources": ctx["sources"],
        "timings": timings,
    }


def _log(
    settings: Settings,
    question: str,
    ctx: Dict[str, object],
    answer: str,
    timings: Dict[str, float],
    session_id: str,
) -> None:
    best_distance = ctx["best_distance"]
    log_rag_event(
        settings.rag_log_path,
        {
            "session_id": session_id,
            "question": question,
            "search_query": ctx["search_query"],
            "used_rag": ctx["used_rag"],
            "best_distance": None if best_distance == float("inf") else round(best_distance, 4),
            "sources": ctx["sources"],
            "metadata": ctx["metadatas"],
            "answer": answer,
            "timings": timings,
        },
    )
