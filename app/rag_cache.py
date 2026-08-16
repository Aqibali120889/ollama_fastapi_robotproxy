import time
from collections import OrderedDict
from threading import Lock
from typing import Any, Optional


class TTLCache:
    """Simple in-process LRU cache with per-entry TTL expiry.

    No external cache store (e.g. Redis) is warranted for a single-process
    FastAPI service — this trades perfect correctness across process restarts
    for zero added infrastructure.
    """

    def __init__(self, max_size: int, ttl_seconds: int):
        self._max_size = max_size
        self._ttl_seconds = ttl_seconds
        self._store: "OrderedDict[Any, tuple[float, Any]]" = OrderedDict()
        self._lock = Lock()

    def get(self, key: Any) -> Optional[Any]:
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            expires_at, value = entry
            if self._ttl_seconds and time.time() >= expires_at:
                del self._store[key]
                return None
            self._store.move_to_end(key)
            return value

    def set(self, key: Any, value: Any) -> None:
        with self._lock:
            expires_at = time.time() + self._ttl_seconds if self._ttl_seconds else float("inf")
            self._store[key] = (expires_at, value)
            self._store.move_to_end(key)
            while len(self._store) > self._max_size:
                self._store.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()


_embedding_cache: Optional[TTLCache] = None
_context_cache: Optional[TTLCache] = None


def get_embedding_cache(max_size: int, ttl_seconds: int) -> TTLCache:
    global _embedding_cache
    if _embedding_cache is None:
        _embedding_cache = TTLCache(max_size=max_size, ttl_seconds=ttl_seconds)
    return _embedding_cache


def get_context_cache(max_size: int, ttl_seconds: int) -> TTLCache:
    global _context_cache
    if _context_cache is None:
        _context_cache = TTLCache(max_size=max_size, ttl_seconds=ttl_seconds)
    return _context_cache


def clear_all() -> None:
    if _embedding_cache is not None:
        _embedding_cache.clear()
    if _context_cache is not None:
        _context_cache.clear()
