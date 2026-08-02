import logging
import time
import uuid
from typing import Dict, List, Optional, TypedDict

from langgraph.graph import END, StateGraph

from app.config import Settings
from app.ollama_proxy import route_chat_request
from app.rag_pipeline import _prepare_context
from app.schemas import ChatMessage, ChatRequest

logger = logging.getLogger(__name__)


class ConversationState(TypedDict, total=False):
    session_id: str
    user_question: str
    mode: str
    model: Optional[str]
    top_k: int
    metadata_filters: Optional[Dict[str, str]]
    incoming_messages: List[ChatMessage]
    memory_messages: List[ChatMessage]
    response_text: str
    response_model: str
    response_source: str
    used_rag: bool
    sources: List[str]
    search_query: str
    timings: Dict[str, float]
    rag_prompt: str


class SessionMemoryStore:
    def __init__(self):
        self._sessions: Dict[str, List[ChatMessage]] = {}

    def get(self, session_id: str) -> List[ChatMessage]:
        return list(self._sessions.get(session_id, []))

    def append(self, session_id: str, messages: List[ChatMessage], max_turns: int) -> None:
        current = self._sessions.get(session_id, [])
        updated = (current + messages)[-max_turns * 2 :]
        self._sessions[session_id] = updated


memory_store = SessionMemoryStore()


def _new_session_id(session_id: Optional[str]) -> str:
    return session_id or str(uuid.uuid4())


def build_conversation_graph(settings: Settings):
    graph = StateGraph(ConversationState)

    def load_memory(state: ConversationState) -> ConversationState:
        session_id = _new_session_id(state.get("session_id"))
        return {
            "session_id": session_id,
            "memory_messages": memory_store.get(session_id),
            "timings": {},
        }

    async def route_node(state: ConversationState) -> ConversationState:
        question = state["user_question"]
        timings = dict(state.get("timings") or {})
        t0 = time.perf_counter()

        from httpx import AsyncClient

        async with AsyncClient(timeout=settings.request_timeout_seconds) as client:
            ctx = await _prepare_context(
                client,
                settings,
                question,
                state.get("top_k", settings.rag_top_k),
                metadata_filters=state.get("metadata_filters"),
            )

        timings["graph_route_s"] = round(time.perf_counter() - t0, 3)
        return {
            "used_rag": ctx["used_rag"],
            "sources": ctx["sources"],
            "search_query": ctx["search_query"],
            "rag_prompt": ctx["prompt"],
            "timings": {**timings, **ctx["timings"]},
        }

    async def rag_generate(state: ConversationState) -> ConversationState:
        incoming = state.get("incoming_messages", [])
        prompt_message = ChatMessage(role="user", content=state["rag_prompt"])
        messages = [
            *state.get("memory_messages", []),
            *incoming[:-1],
            prompt_message,
        ]
        payload = ChatRequest(messages=messages, mode="chat", model=state.get("model"), session_id=state["session_id"])
        output, model, source_meta = await route_chat_request(settings, payload)
        return {
            "response_text": output,
            "response_model": model,
            "response_source": source_meta[0]["source"] if source_meta else "unknown",
        }

    async def chat_generate(state: ConversationState) -> ConversationState:
        incoming = state.get("incoming_messages", [])
        if incoming:
            direct_messages = [*state.get("memory_messages", []), *incoming]
        else:
            direct_messages = [*state.get("memory_messages", []), ChatMessage(role="user", content=state["user_question"])]
        payload = ChatRequest(
            messages=direct_messages,
            mode=state.get("mode", "chat"),
            model=state.get("model"),
            session_id=state["session_id"],
        )
        output, model, source_meta = await route_chat_request(settings, payload)
        return {
            "response_text": output,
            "response_model": model,
            "response_source": source_meta[0]["source"] if source_meta else "unknown",
            "search_query": state.get("user_question", ""),
        }

    def save_memory(state: ConversationState) -> ConversationState:
        incoming = state.get("incoming_messages", [])
        latest_user = state["user_question"]
        for msg in reversed(incoming):
            if msg.role == "user":
                latest_user = msg.content
                break
        new_messages = [
            ChatMessage(role="user", content=latest_user),
            ChatMessage(role="assistant", content=state.get("response_text", "")),
        ]
        memory_store.append(state["session_id"], new_messages, max_turns=settings.memory_max_turns)
        return state

    def pick_path(state: ConversationState) -> str:
        return "rag_generate" if state.get("used_rag") else "chat_generate"

    graph.add_node("load_memory", load_memory)
    graph.add_node("route", route_node)
    graph.add_node("rag_generate", rag_generate)
    graph.add_node("chat_generate", chat_generate)
    graph.add_node("save_memory", save_memory)

    graph.set_entry_point("load_memory")
    graph.add_edge("load_memory", "route")
    graph.add_conditional_edges("route", pick_path, {"rag_generate": "rag_generate", "chat_generate": "chat_generate"})
    graph.add_edge("rag_generate", "save_memory")
    graph.add_edge("chat_generate", "save_memory")
    graph.add_edge("save_memory", END)
    return graph.compile()
