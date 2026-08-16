import json
import logging
import time
import uuid
from difflib import SequenceMatcher
from pathlib import Path
from typing import AsyncGenerator, Dict, List, Optional, Union

import httpx

from app.config import Settings
from app.hybrid_retriever import hybrid_retrieve
from app.ollama_proxy import route_chat_request
from app.rag_cache import get_context_cache
from app.rag_logging import log_rag_event
from app.rag_store import embed_text, get_collection
from app.schemas import ChatMessage, ChatRequest
from app.vision import describe_image

logger = logging.getLogger(__name__)


def _keep_alive_value(raw: str) -> Union[int, str]:
    try:
        return int(raw)
    except ValueError:
        return raw


RAG_PROMPT_TEMPLATE = """You are an AI assistant representing a company at a business expo booth. \
Answer the visitor's question using ONLY the context below. If the answer is not contained in the \
context, say you don't have that information and offer to connect them with a team member. Answer \
directly with no preamble, in at most 4 concise, professional sentences.

Context:
{context}

Visitor question: {question}

Answer:"""

REWRITE_PROMPT_TEMPLATE = """Rewrite the visitor's question as a short search query (3-8 words) using \
the kind of formal terminology likely to appear in company documentation. Return ONLY the rewritten \
query -- no punctuation, no quotes, no explanation.

Visitor question: {question}

Rewritten query:"""

DIRECT_CHAT_TEMPLATE = """You are a friendly AI assistant for a robot representing a company at a \
business expo booth. The visitor's message does not relate to specific company information, so reply \
naturally and conversationally without inventing any company facts. Keep it brief (1-2 sentences).

Visitor: {question}

Response:"""

# Conversational phrasing that suggests the question would benefit from being
# rewritten into a terser, documentation-style search query before embedding.
_REWRITE_FILLER_PATTERNS = (
    "i was wondering",
    "can you tell me",
    "could you tell me",
    "do you know",
    "what about",
    "i want to know",
    "i'd like to know",
    "so, ",
    "so what",
    "hey,",
    "hey ",
    "um,",
    "um ",
)


def _needs_rewrite(question: str, settings: Settings) -> bool:
    """Fast, LLM-free heuristic deciding whether a question should be rewritten
    into a short search query before embedding. Short, already query-like
    questions skip the rewrite LLM call entirely."""
    if not settings.enable_query_rewrite:
        return False
    stripped = question.strip().lower()
    if not stripped:
        return False
    if len(stripped.split()) > settings.rag_rewrite_word_threshold:
        return True
    return any(pattern in stripped for pattern in _REWRITE_FILLER_PATTERNS)


def _normalize_question(question: str) -> str:
    return " ".join(question.strip().lower().split())


def _context_cache_key(
    settings: Settings,
    question: str,
    top_k: int,
    metadata_filters: Optional[Dict[str, str]],
):
    filters_key = tuple(sorted(metadata_filters.items())) if metadata_filters else None
    return (
        _normalize_question(question),
        top_k,
        filters_key,
        settings.rag_dense_weight,
        settings.rag_keyword_weight,
    )


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
        if len(selected_docs) >= settings.rag_context_max_chunks:
            break

        if any(
            SequenceMatcher(None, doc, existing).ratio() >= settings.rag_dedupe_similarity_threshold
            for existing in selected_docs
        ):
            continue

        next_budget = char_budget + len(doc)
        if next_budget > settings.rag_context_max_chars:
            # Skip (don't stop) so a later, smaller chunk still gets a chance to fit.
            continue

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
            "options": {"num_predict": settings.rewrite_max_tokens},
        },
    )
    response.raise_for_status()
    rewritten = response.json().get("response", "").strip()
    return rewritten or question


async def _resolve_image_chunks(
    client: httpx.AsyncClient,
    settings: Settings,
    question: str,
    filtered: Dict[str, List[object]],
) -> None:
    """For any retrieved chunk whose metadata marks it as an image, replaces its
    generic indexed caption with a fresh, question-focused LLaVA description of
    the original image. Falls back silently to the indexed caption on any
    failure (missing file, oversized, vision call error)."""
    kb_dir = Path(settings.knowledge_base_dir)

    for i, metadata in enumerate(filtered["metadatas"]):
        if metadata.get("type") != "image":
            continue

        image_source = metadata.get("source")
        if not image_source:
            continue

        image_path = kb_dir / str(image_source)
        try:
            if not image_path.is_file():
                continue
            if image_path.stat().st_size > settings.vision_max_image_bytes:
                logger.warning("Skipping oversized image for vision captioning: %s", image_path)
                continue

            image_bytes = image_path.read_bytes()
            prompt = (
                "Answer based only on what is visible in this image. "
                f"Visitor question: {question}\n"
                "Provide a factual, concise description relevant to the question."
            )
            description = await describe_image(
                client,
                settings.ollama_local_base_url,
                settings.vision_model,
                image_bytes,
                keep_alive=_keep_alive_value(settings.ollama_keep_alive),
                prompt=prompt,
            )
            if description:
                filtered["documents"][i] = description
        except Exception:
            logger.warning("Vision captioning failed for %s; using indexed description", image_path, exc_info=True)
            continue


async def _prepare_context(
    client: httpx.AsyncClient,
    settings: Settings,
    question: str,
    top_k: int,
    metadata_filters: Optional[Dict[str, str]] = None,
) -> Dict[str, object]:
    cache = get_context_cache(settings.rag_cache_max_size, settings.rag_cache_ttl_seconds) if settings.enable_rag_cache else None
    cache_key = _context_cache_key(settings, question, top_k, metadata_filters)

    if cache is not None:
        cached = cache.get(cache_key)
        if cached is not None:
            result = dict(cached)
            result["timings"] = {"cache_hit": True}
            return result

    timings: Dict[str, float] = {}
    collection = get_collection(settings.chroma_persist_dir, settings.chroma_collection_name)

    search_query = question
    if _needs_rewrite(question, settings):
        t0 = time.perf_counter()
        search_query = await _rewrite_query(client, settings, question)
        timings["rewrite_s"] = round(time.perf_counter() - t0, 3)

    t0 = time.perf_counter()
    question_embedding = await embed_text(
        client,
        settings.ollama_local_base_url,
        settings.embedding_model,
        search_query,
        keep_alive=_keep_alive_value(settings.ollama_keep_alive),
        use_cache=settings.enable_rag_cache,
        cache_max_size=settings.embedding_cache_max_size,
        cache_ttl_seconds=settings.rag_cache_ttl_seconds,
    )
    timings["embed_s"] = round(time.perf_counter() - t0, 3)

    t0 = time.perf_counter()
    retrieved = hybrid_retrieve(
        collection=collection,
        query_embedding=question_embedding,
        query_text=search_query,
        top_k=top_k,
        dense_weight=settings.rag_dense_weight,
        keyword_weight=settings.rag_keyword_weight,
        candidate_pool_size=top_k * settings.rag_candidate_pool_multiplier,
        min_fusion_score=settings.rag_min_fusion_score,
        metadata_filters=metadata_filters,
    )
    timings["retrieval_s"] = round(time.perf_counter() - t0, 3)

    routing_distances = retrieved["distances"]
    best_distance = min(routing_distances) if routing_distances else float("inf")
    used_rag = best_distance <= settings.rag_distance_threshold and bool(retrieved["documents"])

    filtered = _dedupe_and_cap_context(
        settings=settings,
        documents=retrieved["documents"],
        metadatas=retrieved["metadatas"],
        distances=retrieved["distances"],
        fusion_scores=retrieved["fusion_scores"],
    )

    if used_rag and filtered["documents"] and any(m.get("type") == "image" for m in filtered["metadatas"]):
        t0 = time.perf_counter()
        await _resolve_image_chunks(client, settings, question, filtered)
        timings["vision_s"] = round(time.perf_counter() - t0, 3)

    sources = [meta.get("source", "unknown") for meta in filtered["metadatas"]]

    if used_rag and filtered["documents"]:
        context = "\n\n---\n\n".join(filtered["documents"])
        prompt = RAG_PROMPT_TEMPLATE.format(context=context, question=question)
    else:
        used_rag = False
        prompt = DIRECT_CHAT_TEMPLATE.format(question=question)

    result = {
        "prompt": prompt,
        "used_rag": used_rag,
        "search_query": search_query,
        "sources": sources,
        "retrieved_chunks": filtered["documents"],
        "distances": filtered["distances"],
        "fusion_scores": filtered["fusion_scores"],
        "metadatas": filtered["metadatas"],
        "best_distance": best_distance,
    }

    if cache is not None:
        cache.set(cache_key, dict(result))

    result["timings"] = timings
    return result


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
            chat_payload = ChatRequest(
                messages=messages,
                mode="chat",
                session_id=session_id,
                options={"num_predict": settings.rag_answer_max_tokens},
            )
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
                    "options": {"num_predict": settings.rag_answer_max_tokens},
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
