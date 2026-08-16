from typing import List, Optional, Union

import chromadb
import httpx

from app.rag_cache import get_embedding_cache


async def embed_text(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    text: str,
    keep_alive: Optional[Union[int, str]] = None,
    use_cache: bool = True,
    cache_max_size: int = 512,
    cache_ttl_seconds: int = 600,
) -> List[float]:
    cache = get_embedding_cache(cache_max_size, cache_ttl_seconds) if use_cache else None
    cache_key = (model, text)

    if cache is not None:
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

    payload = {"model": model, "prompt": text}
    if keep_alive is not None:
        payload["keep_alive"] = keep_alive

    response = await client.post(
        f"{base_url.rstrip('/')}/api/embeddings",
        json=payload,
    )
    response.raise_for_status()
    embedding = response.json()["embedding"]

    if cache is not None:
        cache.set(cache_key, embedding)

    return embedding


def get_collection(persist_dir: str, collection_name: str):
    client = chromadb.PersistentClient(path=persist_dir)
    return client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})


def reset_collection(persist_dir: str, collection_name: str):
    client = chromadb.PersistentClient(path=persist_dir)
    existing = [c.name for c in client.list_collections()]
    if collection_name in existing:
        client.delete_collection(collection_name)
    return client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})
