import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx

from app.config import get_settings
from app.rag_ingest import build_chunks
from app.rag_store import embed_text, reset_collection


async def main():
    settings = get_settings()

    print(f"Loading and chunking documents from: {settings.knowledge_base_dir}")
    chunks = build_chunks(settings.knowledge_base_dir, settings.rag_chunk_size, settings.rag_chunk_overlap)
    print(f"Produced {len(chunks)} chunks from the knowledge base.")

    if not chunks:
        print("No chunks produced. Check that knowledge_base/ contains .txt, .md, or .pdf files.")
        return

    collection = reset_collection(settings.chroma_persist_dir, settings.chroma_collection_name)

    async with httpx.AsyncClient(timeout=settings.request_timeout_seconds) as client:
        for i, chunk in enumerate(chunks, start=1):
            embedding = await embed_text(
                client, settings.ollama_local_base_url, settings.embedding_model, chunk.text
            )
            collection.add(
                ids=[chunk.id],
                embeddings=[embedding],
                documents=[chunk.text],
                metadatas=[
                    {
                        "source": chunk.source,
                        "source_path": chunk.source_path,
                        "extension": chunk.extension,
                        "file_type": chunk.file_type,
                        "chunk_index": chunk.chunk_index,
                    }
                ],
            )
            print(f"  [{i}/{len(chunks)}] embedded and stored: {chunk.id}")

    print(f"\nDone. Collection '{settings.chroma_collection_name}' now has {collection.count()} chunks.")
    print(f"Persisted at: {settings.chroma_persist_dir}")


if __name__ == "__main__":
    asyncio.run(main())
