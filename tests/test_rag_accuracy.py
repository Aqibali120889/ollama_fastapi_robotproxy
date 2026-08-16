import json
from pathlib import Path

import httpx
import pytest

from app.config import get_settings

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_keyword_coverage_full_and_partial_and_empty():
    from scripts.evaluate_rag import keyword_coverage

    assert keyword_coverage("The robot uses Tailscale for networking.", ["Tailscale"]) == 1.0
    assert keyword_coverage("The robot uses Tailscale.", ["Tailscale", "WireGuard"]) == 0.5
    assert keyword_coverage("No relevant info here.", ["Tailscale", "WireGuard"]) == 0.0
    assert keyword_coverage("Anything.", []) == 1.0


def test_question_passes_requires_both_rag_usage_and_coverage():
    from scripts.evaluate_rag import question_passes

    assert question_passes(used_rag=True, coverage=0.6, keyword_pass_threshold=0.5) is True
    assert question_passes(used_rag=True, coverage=0.4, keyword_pass_threshold=0.5) is False
    assert question_passes(used_rag=False, coverage=1.0, keyword_pass_threshold=0.5) is False


def test_eval_dataset_has_at_least_20_questions_with_required_fields():
    dataset = json.loads((REPO_ROOT / "evaluation" / "rag_eval_dataset.json").read_text(encoding="utf-8"))
    assert len(dataset) >= 20
    for item in dataset:
        assert item["question"].strip()
        assert item["expected_sources"]
        assert item["reference_keywords"]


@pytest.fixture
def _ollama_available():
    settings = get_settings()
    try:
        response = httpx.get(f"{settings.ollama_local_base_url.rstrip('/')}/api/tags", timeout=2.0)
        response.raise_for_status()
    except Exception:
        pytest.skip("Ollama is not reachable from this environment; skipping live RAG accuracy check.")


@pytest.mark.integration
async def test_rag_accuracy_meets_80_percent_on_eval_dataset(_ollama_available):
    """Live check: >=80% of the 20-question eval set must pass (used_rag + keyword
    coverage). Requires a reachable Ollama with the knowledge base already ingested
    via scripts/ingest_kb.py -- not runnable in a sandbox with no Ollama/Chroma."""
    from app.rag_pipeline import rag_answer
    from scripts.evaluate_rag import keyword_coverage, question_passes

    settings = get_settings()
    dataset = json.loads((REPO_ROOT / "evaluation" / "rag_eval_dataset.json").read_text(encoding="utf-8"))

    passed = 0
    for item in dataset:
        result = await rag_answer(settings, item["question"])
        coverage = keyword_coverage(result["answer"], item["reference_keywords"])
        if question_passes(result.get("used_rag", False), coverage, keyword_pass_threshold=0.5):
            passed += 1

    accuracy = passed / len(dataset)
    assert accuracy >= 0.8, f"RAG accuracy {accuracy:.2%} is below the required 80%"
