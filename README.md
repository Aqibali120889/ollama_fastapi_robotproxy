# ollama_fastapi_robotproxy

FastAPI-based Ollama proxy with remote-first fallback, LangGraph conversational orchestration, hybrid retrieval RAG, and practical Week 4 evaluation/profiling tooling.

Repository: `shanooo773/ollama_fastapi_robotproxy`

---

## Week 4 Changes (LangGraph + Hybrid Retrieval + Memory + Docker + Eval/Profiling)

This section explicitly summarizes what was added/changed this week and why.

1. **LangGraph orchestration + multi-turn memory**
   - Added `app/conversation_graph.py` with a LangGraph workflow (`load_memory -> route -> rag/chat -> save_memory`).
   - Added session-aware memory store so `/chat` and `/agent` can keep recent turns by `session_id`.
   - Why: move from ad-hoc orchestration to explicit workflow state and better multi-turn continuity.

2. **Hybrid retrieval (dense + keyword fusion)**
   - Added `app/hybrid_retriever.py` (dense Chroma results + BM25-like keyword scoring + weighted fusion).
   - Added config controls in `app/config.py`: `RAG_DENSE_WEIGHT`, `RAG_KEYWORD_WEIGHT`, `RAG_MIN_FUSION_SCORE`, `RAG_CANDIDATE_POOL_MULTIPLIER`.
   - Why: improve recall on exact keyword-heavy queries while keeping semantic retrieval strength.

3. **Metadata-aware search + context filtering**
   - Ingestion now stores richer metadata (`source`, `source_path`, `extension/file_type`, `chunk_index`).
   - Added optional metadata filters in schemas (`metadata_filters`) and retrieval path.
   - Added context filtering stage: near-duplicate chunk removal + context char/chunk budgets.
   - Why: improve retrieval precision, traceability, and prompt efficiency.

4. **Retrieval optimization + observability**
   - Extended timing instrumentation with stage-level timings (`embed`, `retrieval`, `rewrite`, `generation`, `total`, graph route timing).
   - Improved actionable errors for orchestration/RAG failures.
   - Preserved remote-primary -> remote-secondary -> local fallback via existing chat router.
   - Why: easier debugging and operational visibility.

5. **Embedding benchmark utility**
   - Added `scripts/benchmark_embeddings.py` and sample dataset `evaluation/embedding_benchmark_dataset.json`.
   - Outputs JSON/CSV artifacts under `artifacts/benchmarks/`.
   - Why: compare embedding model latency and hit@k proxy before deployment decisions.

6. **RAG evaluation metrics**
   - Upgraded `scripts/evaluate_rag.py` to produce lightweight reproducible metrics:
     - answer relevance proxy (keyword overlap),
     - context precision/recall proxy (expected vs retrieved sources),
     - faithfulness proxy (answer/context token overlap),
     - latency.
   - Added sample dataset: `evaluation/rag_eval_dataset.json`.
   - Outputs JSON/CSV under `artifacts/evaluation/`.
   - Why: establish a practical baseline without heavyweight frameworks.

7. **Dockerization**
   - Added `Dockerfile`, `docker-compose.yml`, and `.dockerignore`.
   - Includes env-driven configuration and volume mounts for Chroma/log/artifact persistence.
   - Why: consistent local/server deployment path.

8. **Performance profiling**
   - Added `scripts/profile_performance.py` for p50/p95 latency, throughput, and CPU/RAM snapshot notes.
   - Outputs JSON/CSV under `artifacts/profiling/`.
   - Why: quick operational performance checks before and after tuning.

---

## Architecture Overview

### Runtime flow

- **API**: FastAPI (`app/main.py`) exposes:
  - `POST /chat`
  - `POST /agent`
  - `POST /rag/ask`
  - `WS /ws/rag`
  - `GET /health`
- **Conversation Orchestration**: LangGraph (`app/conversation_graph.py`) decides RAG vs direct chat and persists session memory.
- **Retriever**:
  - Dense vector retrieval from ChromaDB
  - Keyword scoring (BM25-like)
  - Weighted fusion + metadata filters
  - Context dedupe + budget cap
- **Generation routing**:
  - remote primary model
  - remote secondary model
  - local fallback model

---

## Setup (Local)

### 1) Install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2) Configure env

```bash
cp .env.example .env
# edit values (remote/local Ollama URLs, model names, retrieval weights)
```

### 3) Ingest KB into Chroma

```bash
python scripts/ingest_kb.py
```

### 4) Run API

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

---

## Docker Run

### Build + run with compose

```bash
cp .env.example .env
docker compose up --build
```

Default service:
- API: `http://localhost:8000`
- Volumes:
  - `./chroma_store` -> Chroma persistence
  - `./logs` -> RAG logs
  - `./artifacts` -> benchmark/eval/profile outputs

> Set `OLLAMA_LOCAL_BASE_URL` / `OLLAMA_REMOTE_URLS` in `.env` to host or remote Ollama endpoints.

---

## API Usage Examples

### Health

```bash
curl -s http://localhost:8000/health | jq
```

### Chat / Agent with session memory

```bash
curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"demo-1","prompt":"What does NovaTech Robotics do?"}' | jq
```

### RAG ask with metadata filter

```bash
curl -s -X POST http://localhost:8000/rag/ask \
  -H "Content-Type: application/json" \
  -d '{
    "session_id":"demo-rag-1",
    "question":"What is the assembly arm pricing?",
    "top_k":4,
    "metadata_filters":{"extension":"txt"}
  }' | jq
```

---

## Embedding Benchmark Usage

```bash
python scripts/benchmark_embeddings.py \
  --models nomic-embed-text all-minilm \
  --dataset evaluation/embedding_benchmark_dataset.json \
  --top-k 1
```

Artifacts:
- `artifacts/benchmarks/embedding_benchmark_*.json`
- `artifacts/benchmarks/embedding_benchmark_*.csv`

---

## RAG Evaluation Usage

```bash
python scripts/evaluate_rag.py --dataset evaluation/rag_eval_dataset.json
```

Artifacts:
- `artifacts/evaluation/rag_eval_*.json`
- `artifacts/evaluation/rag_eval_*.csv`

---

## Performance Profiling Usage

```bash
python scripts/profile_performance.py \
  --base-url http://localhost:8000 \
  --endpoint /rag/ask \
  --requests 20 \
  --concurrency 4
```

Artifacts:
- `artifacts/profiling/profile_*.json`
- `artifacts/profiling/profile_*.csv`

---

## Troubleshooting

- **`RAG pipeline failed`**: verify Ollama is reachable and run `python scripts/ingest_kb.py`.
- **No useful retrieval context**: tune `RAG_DENSE_WEIGHT`, `RAG_KEYWORD_WEIGHT`, `RAG_MIN_FUSION_SCORE`, or metadata filters.
- **Latency spikes**: profile with `scripts/profile_performance.py`, then adjust model size, `REQUEST_TIMEOUT_SECONDS`, and context caps.
- **Docker cannot reach Ollama**: use reachable host in `.env` (often `http://host.docker.internal:11434` on Docker Desktop).

---

## Key Files Added/Updated

- `app/conversation_graph.py`
- `app/hybrid_retriever.py`
- `app/rag_pipeline.py`
- `app/rag_ingest.py`
- `app/schemas.py`
- `app/config.py`
- `scripts/benchmark_embeddings.py`
- `scripts/evaluate_rag.py`
- `scripts/profile_performance.py`
- `evaluation/embedding_benchmark_dataset.json`
- `evaluation/rag_eval_dataset.json`
- `Dockerfile`
- `docker-compose.yml`
- `.dockerignore`
