import argparse
import asyncio
import csv
import json
import os
import resource
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx


def percentile(values, p):
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = max(int(len(ordered) * p) - 1, 0)
    return ordered[idx]


async def worker(client: httpx.AsyncClient, endpoint: str, payload: dict, latencies: list):
    start = time.perf_counter()
    response = await client.post(endpoint, json=payload)
    response.raise_for_status()
    latencies.append((time.perf_counter() - start) * 1000)


async def main_async(args):
    endpoint = f"{args.base_url.rstrip('/')}/{args.endpoint.lstrip('/')}"
    payload = {"question": args.question} if "rag" in args.endpoint else {"prompt": args.question}
    latencies = []

    async with httpx.AsyncClient(timeout=args.timeout) as client:
        start = time.perf_counter()
        for _ in range(args.requests // args.concurrency):
            await asyncio.gather(*[worker(client, endpoint, payload, latencies) for _ in range(args.concurrency)])
        remainder = args.requests % args.concurrency
        if remainder:
            await asyncio.gather(*[worker(client, endpoint, payload, latencies) for _ in range(remainder)])
        elapsed = time.perf_counter() - start

    throughput = len(latencies) / elapsed if elapsed > 0 else 0.0
    usage = resource.getrusage(resource.RUSAGE_SELF)

    return {
        "endpoint": endpoint,
        "requests": len(latencies),
        "concurrency": args.concurrency,
        "latency_ms_p50": round(percentile(latencies, 0.5), 2),
        "latency_ms_p95": round(percentile(latencies, 0.95), 2),
        "latency_ms_mean": round(statistics.mean(latencies), 2) if latencies else 0.0,
        "throughput_rps": round(throughput, 2),
        "cpu_time_s_user": round(usage.ru_utime, 3),
        "cpu_time_s_system": round(usage.ru_stime, 3),
        "max_rss_kb": int(usage.ru_maxrss),
        "gpu_note": "GPU stats are environment-specific; collect via nvidia-smi/tegrastats in deployment.",
    }


def main():
    parser = argparse.ArgumentParser(description="Basic latency/throughput profiler for FastAPI endpoints")
    parser.add_argument("--base-url", default=os.getenv("PROFILE_BASE_URL", "http://localhost:8000"))
    parser.add_argument("--endpoint", default="/rag/ask", choices=["/rag/ask", "/chat"])
    parser.add_argument("--question", default="What products do you offer?")
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--out-dir", default="artifacts/profiling")
    args = parser.parse_args()

    result = asyncio.run(main_async(args))

    root = Path(__file__).resolve().parent.parent
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = out_dir / f"profile_{ts}.json"
    csv_path = out_dir / f"profile_{ts}.csv"

    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(result.keys()))
        writer.writeheader()
        writer.writerow(result)

    print(json.dumps(result, indent=2))
    print(f"Saved:\n- {json_path}\n- {csv_path}")


if __name__ == "__main__":
    main()
