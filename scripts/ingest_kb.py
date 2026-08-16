import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx

from app.config import get_settings
from app.rag_ingest import build_chunks
from app.rag_store import embed_text, reset_collection
from app.vision import describe_image


def _keep_alive_value(raw: str):
    try:
        return int(raw)
    except ValueError:
        return raw


async def main():
    settings = get_settings()
    keep_alive = _keep_alive_value(settings.ollama_keep_alive)
    image_extensions = set(settings.image_extensions())

    print(f"Loading and chunking documents from: {settings.knowledge_base_dir}")
    chunks = build_chunks(
        settings.knowledge_base_dir,
        settings.rag_chunk_size,
        settings.rag_chunk_overlap,
        image_extensions=image_extensions,
    )
    text_chunks = [c for c in chunks if c.chunk_type == "text"]
    image_chunks = [c for c in chunks if c.chunk_type == "image"]
    print(f"Produced {len(text_chunks)} text chunks and {len(image_chunks)} image chunks from the knowledge base.")

    if not chunks:
        print("No chunks produced. Check that knowledge_base/ contains .txt, .md, .pdf, or image files.")
        return

    collection = reset_collection(settings.chroma_persist_dir, settings.chroma_collection_name)

    async with httpx.AsyncClient(timeout=settings.request_timeout_seconds) as client:
        total = len(chunks)
        count = 0

        for chunk in image_chunks:
            count += 1
            if len(chunk.image_bytes or b"") > settings.vision_max_image_bytes:
                print(f"  [{count}/{total}] SKIPPED (too large): {chunk.id}")
                continue
            try:
                description = await describe_image(
                    client,
                    settings.ollama_local_base_url,
                    settings.vision_model,
                    chunk.image_bytes,
                    keep_alive=keep_alive,
                )
            except Exception as exc:
                print(f"  [{count}/{total}] SKIPPED (LLaVA captioning failed: {exc}): {chunk.id}")
                continue
            if not description:
                print(f"  [{count}/{total}] SKIPPED (empty caption): {chunk.id}")
                continue
            chunk.text = description

            embedding = await embed_text(
                client, settings.ollama_local_base_url, settings.embedding_model, chunk.text, keep_alive=keep_alive
            )
            collection.add(
                ids=[chunk.id],
                embeddings=[embedding],
                documents=[chunk.text],
                metadatas=[
                    {
                        "type": "image",
                        "source": chunk.image_path,
                        "description": chunk.text,
                        "source_path": chunk.source_path,
                        "extension": chunk.extension,
                        "file_type": chunk.file_type,
                        "chunk_index": chunk.chunk_index,
                    }
                ],
            )
            print(f"  [{count}/{total}] captioned, embedded and stored (image): {chunk.id}")

        for chunk in text_chunks:
            count += 1
            embedding = await embed_text(
                client, settings.ollama_local_base_url, settings.embedding_model, chunk.text, keep_alive=keep_alive
            )
            collection.add(
                ids=[chunk.id],
                embeddings=[embedding],
                documents=[chunk.text],
                metadatas=[
                    {
                        "type": "text",
                        "source": chunk.source,
                        "source_path": chunk.source_path,
                        "extension": chunk.extension,
                        "file_type": chunk.file_type,
                        "chunk_index": chunk.chunk_index,
                    }
                ],
            )
            print(f"  [{count}/{total}] embedded and stored: {chunk.id}")

    print(f"\nDone. Collection '{settings.chroma_collection_name}' now has {collection.count()} chunks.")
    print(f"Persisted at: {settings.chroma_persist_dir}")


if __name__ == "__main__":
    asyncio.run(main())
