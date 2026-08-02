# Week 4 Upgrade Report

## Summary
Week 4 upgraded the proxy from basic routing to LangGraph-managed conversational orchestration with session memory, hybrid retrieval, richer metadata filtering, and practical benchmarking/evaluation/profiling utilities.

## Challenges and Solutions

- **Challenge:** Maintaining API compatibility while introducing session memory and graph orchestration.
  - **Solution:** Kept existing endpoints and made `session_id` optional in request schemas.

- **Challenge:** Dense-only retrieval missed exact lexical matches.
  - **Solution:** Added BM25-like keyword scoring and weighted fusion with configurable weights.

- **Challenge:** Prompt bloat from duplicate or oversized context.
  - **Solution:** Added dedupe and context budget controls (`RAG_CONTEXT_MAX_CHARS`, `RAG_CONTEXT_MAX_CHUNKS`).

- **Challenge:** Practical evaluation without heavy tooling.
  - **Solution:** Added lightweight reproducible evaluators with explicit heuristic assumptions.

## Where to Validate

- Main documentation: `README.md` -> **Week 4 Changes** section
- LangGraph orchestration: `app/conversation_graph.py`
- Hybrid retrieval: `app/hybrid_retriever.py`
- RAG metrics eval: `scripts/evaluate_rag.py`
- Embedding benchmark: `scripts/benchmark_embeddings.py`
- Profiling: `scripts/profile_performance.py`
