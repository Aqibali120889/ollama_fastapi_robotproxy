import argparse
import asyncio
import csv
import json
import math
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx

from app.config import get_settings
from app.rag_store import embed_text


def cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    a_norm = math.sqrt(sum(x * x for x in a))
    b_norm = math.sqrt(sum(x * x for x in b))
    if a_norm == 0 or b_norm == 0:
        return 0.0
    return dot / (a_norm * b_norm)


async def benchmark_model(client: httpx.AsyncClient, model: str, dataset: Dict[str, object], top_k: int) -> Dict[str, object]:
    documents = dataset["documents"]
    queries = dataset["queries"]

    doc_embeddings = {}
    doc_latencies = []
    for doc in documents:
        t0 = time.perf_counter()
        doc_embeddings[doc["source"]] = await embed_text(client, get_settings().ollama_local_base_url, model, doc["text"])
        doc_latencies.append((time.perf_counter() - t0) * 1000)

    query_latencies = []
    hits = 0
    for row in queries:
        t0 = time.perf_counter()
        q_embedding = await embed_text(client, get_settings().ollama_local_base_url, model, row["question"])
        query_latencies.append((time.perf_counter() - t0) * 1000)

        ranked = sorted(
            (
                (source, cosine_similarity(q_embedding, emb))
                for source, emb in doc_embeddings.items()
            ),
            key=lambda x: x[1],
            reverse=True,
        )
        if row["expected_source"] in [source for source, _ in ranked[:top_k]]:
            hits += 1

    hit_at_k = hits / max(len(queries), 1)
    all_latencies = doc_latencies + query_latencies

    return {
        "model": model,
        "documents": len(documents),
        "queries": len(queries),
        "hit_at_k": round(hit_at_k, 4),
        "latency_ms_mean": round(statistics.mean(all_latencies), 2),
        "latency_ms_p95": round(sorted(all_latencies)[max(int(len(all_latencies) * 0.95) - 1, 0)], 2),
    }


async def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark embedding models (latency + hit@k)")
    parser.add_argument("--models", nargs="+", required=True, help="Embedding model names")
    parser.add_argument("--dataset", default="evaluation/embedding_benchmark_dataset.json")
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--out-dir", default="artifacts/benchmarks")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    dataset_path = root / args.dataset
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    with dataset_path.open("r", encoding="utf-8") as f:
        dataset = json.load(f)

    settings = get_settings()
    timeout = httpx.Timeout(settings.request_timeout_seconds)

    results = []
    async with httpx.AsyncClient(timeout=timeout) as client:
        for model in args.models:
            print(f"Benchmarking {model} ...")
            results.append(await benchmark_model(client, model, dataset, args.top_k))

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = out_dir / f"embedding_benchmark_{ts}.json"
    csv_path = out_dir / f"embedding_benchmark_{ts}.csv"

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(args.dataset),
        "top_k": args.top_k,
        "results": results,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    print(f"Wrote benchmark results:\n- {json_path}\n- {csv_path}")


if __name__ == "__main__":
    asyncio.run(main())
