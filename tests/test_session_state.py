from app.conversation_graph import SessionMemoryStore
from app.schemas import ChatMessage


def test_session_memory_store_get_unknown_session_returns_empty():
    store = SessionMemoryStore()
    assert store.get("unknown-session") == []


def test_session_memory_store_append_and_get_round_trip():
    store = SessionMemoryStore()
    msgs = [ChatMessage(role="user", content="hi"), ChatMessage(role="assistant", content="hello")]
    store.append("s1", msgs, max_turns=8)
    assert store.get("s1") == msgs


def test_session_memory_store_get_returns_a_copy():
    store = SessionMemoryStore()
    store.append("s1", [ChatMessage(role="user", content="hi")], max_turns=8)
    history = store.get("s1")
    history.append(ChatMessage(role="user", content="mutated"))
    assert len(store.get("s1")) == 1


def test_session_memory_store_truncates_to_max_turns():
    store = SessionMemoryStore()
    for i in range(5):
        store.append(
            "s1",
            [ChatMessage(role="user", content=f"q{i}"), ChatMessage(role="assistant", content=f"a{i}")],
            max_turns=2,
        )
    history = store.get("s1")
    assert len(history) == 4
    assert [m.content for m in history] == ["q3", "a3", "q4", "a4"]


def test_session_memory_store_isolated_between_sessions():
    store = SessionMemoryStore()
    store.append("s1", [ChatMessage(role="user", content="hi from s1")], max_turns=8)
    store.append("s2", [ChatMessage(role="user", content="hi from s2")], max_turns=8)
    assert store.get("s1")[0].content == "hi from s1"
    assert store.get("s2")[0].content == "hi from s2"
