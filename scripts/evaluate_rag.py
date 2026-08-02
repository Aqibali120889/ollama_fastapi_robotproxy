import argparse
import asyncio
import csv
import json
import re
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Set

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings
from app.rag_pipeline import rag_answer


def _tokens(text: str) -> Set[str]:
    return set(re.findall(r"[a-zA-Z0-9]+", text.lower()))


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    a_set = set(a)
    b_set = set(b)
    if not a_set and not b_set:
        return 1.0
    if not a_set or not b_set:
        return 0.0
    return len(a_set & b_set) / len(a_set | b_set)


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run lightweight RAG evaluation metrics")
    parser.add_argument("--dataset", default="evaluation/rag_eval_dataset.json")
    parser.add_argument("--out-dir", default="artifacts/evaluation")
    parser.add_argument("--top-k", type=int, default=None)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    dataset_path = root / args.dataset
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    settings = get_settings()

    rows: List[dict] = []
    for item in dataset:
        result = await rag_answer(settings, item["question"], top_k=args.top_k)

        expected_sources = set(item.get("expected_sources", []))
        retrieved_sources = set(result.get("sources", []))
        source_hits = len(expected_sources & retrieved_sources)

        precision = source_hits / len(retrieved_sources) if retrieved_sources else 0.0
        recall = source_hits / len(expected_sources) if expected_sources else 0.0

        relevance = jaccard(_tokens(result["answer"]), set(k.lower() for k in item.get("reference_keywords", [])))
        context_text = " ".join(result.get("retrieved_chunks", []))
        answer_tokens = _tokens(result["answer"])
        context_tokens = _tokens(context_text)
        faithfulness = len(answer_tokens & context_tokens) / len(answer_tokens) if answer_tokens else 0.0

        rows.append(
            {
                "question": item["question"],
                "session_id": result.get("session_id", ""),
                "used_rag": result.get("used_rag", False),
                "answer_relevance": round(relevance, 4),
                "context_precision": round(precision, 4),
                "context_recall": round(recall, 4),
                "faithfulness_proxy": round(faithfulness, 4),
                "latency_s": round(float(result.get("timings", {}).get("total_s", 0.0)), 4),
                "sources": ";".join(result.get("sources", [])),
            }
        )

    metrics = {
        "answer_relevance_avg": round(statistics.mean(r["answer_relevance"] for r in rows), 4),
        "context_precision_avg": round(statistics.mean(r["context_precision"] for r in rows), 4),
        "context_recall_avg": round(statistics.mean(r["context_recall"] for r in rows), 4),
        "faithfulness_proxy_avg": round(statistics.mean(r["faithfulness_proxy"] for r in rows), 4),
        "latency_s_avg": round(statistics.mean(r["latency_s"] for r in rows), 4),
    }

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = out_dir / f"rag_eval_{ts}.json"
    csv_path = out_dir / f"rag_eval_{ts}.csv"

    json_path.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "dataset": args.dataset,
                "assumptions": {
                    "answer_relevance": "Jaccard overlap between answer tokens and reference keywords",
                    "context_precision_recall": "Source overlap against expected_sources",
                    "faithfulness_proxy": "Fraction of answer tokens present in retrieved context",
                },
                "aggregate_metrics": metrics,
                "rows": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("Aggregate metrics:")
    for k, v in metrics.items():
        print(f"- {k}: {v}")
    print(f"\nSaved:\n- {json_path}\n- {csv_path}")


if __name__ == "__main__":
    asyncio.run(main())
