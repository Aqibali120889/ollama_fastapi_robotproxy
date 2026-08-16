import argparse
import asyncio
import csv
import json
import time
import wave
from datetime import datetime, timezone
from pathlib import Path

import websockets


def _read_wav_bytes(path: Path) -> bytes:
    with wave.open(str(path), "rb") as wf:
        return wf.readframes(wf.getnframes())


async def _run_once(ws_url: str, audio_bytes: bytes) -> dict:
    start = time.perf_counter()
    async with websockets.connect(ws_url, open_timeout=10, max_size=None) as ws:
        await ws.recv()  # session
        await ws.send(audio_bytes)

        timings: dict = {}
        async for raw in ws:
            if isinstance(raw, bytes):
                continue
            event = json.loads(raw)
            if event.get("type") == "error":
                raise RuntimeError(event.get("message", "voice pipeline error"))
            if event.get("type") == "done":
                timings = event.get("timings", {})
                break

        timings["wall_clock_s"] = round(time.perf_counter() - start, 3)
        return timings


async def main_async(args):
    audio_bytes = _read_wav_bytes(Path(args.audio))
    results = []
    for i in range(args.requests):
        timings = await _run_once(args.ws_url, audio_bytes)
        print(f"[{i + 1}/{args.requests}] {timings}")
        results.append(timings)
    return results


def main():
    parser = argparse.ArgumentParser(description="Profile /ws/voice end-to-end latency (target: <4s)")
    parser.add_argument("--ws-url", default="ws://localhost:8000/ws/voice")
    parser.add_argument("--audio", required=True, help="Path to a mono 16-bit PCM WAV utterance sample")
    parser.add_argument("--requests", type=int, default=5)
    parser.add_argument("--out-dir", default="artifacts/profiling")
    args = parser.parse_args()

    results = asyncio.run(main_async(args))

    root = Path(__file__).resolve().parent.parent
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    stage_keys = sorted({k for r in results for k in r.keys()})
    avg = {k: round(sum(r.get(k, 0.0) for r in results) / len(results), 3) for k in stage_keys}

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = out_dir / f"voice_latency_{ts}.json"
    csv_path = out_dir / f"voice_latency_{ts}.csv"

    payload = {"average": avg, "target_total_s": 4.0, "runs": results}
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=stage_keys)
        writer.writeheader()
        writer.writerows(results)

    print("\nAverage stage timings (s):")
    for k, v in avg.items():
        print(f"- {k}: {v}")

    total = avg.get("total_s", avg.get("wall_clock_s"))
    if total is not None:
        verdict = "PASS" if total <= 4.0 else "FAIL"
        print(f"\n{verdict}: average total_s={total} (target <= 4.0s)")

    print(f"\nSaved:\n- {json_path}\n- {csv_path}")


if __name__ == "__main__":
    main()
