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
   - Preserved remote-primary -> local fallback via existing chat router.
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

## RAG Speed + Multimodal Upgrade (Query Rewrite, Caching, LLaVA Ingestion)

This section summarizes the follow-up optimization pass and why each change was made.

1. **Single-pass, conditional query rewrite**
   - Reordered the pipeline to `rewrite (conditional) -> embed -> retrieve` instead of `embed -> retrieve -> rewrite -> embed -> retrieve`, removing a wasted first retrieval.
   - Added `_needs_rewrite()` (`app/rag_pipeline.py`): a fast, LLM-free heuristic (question length + conversational filler phrases) that decides whether rewriting is worth it, gated by `RAG_REWRITE_WORD_THRESHOLD`.
   - Why: cuts one embed+retrieve round trip on every request, and rewrite is no longer gated on the very retrieval it's meant to improve.

2. **Lightweight, capped rewrite**
   - Rewrite still uses `LOCAL_FALLBACK_MODEL` (small/local), now with `"options": {"num_predict": REWRITE_MAX_TOKENS}` and a tightened prompt asking for a 3-8 word query only.
   - Why: keeps the rewrite call itself fast.

3. **Concise-answer generation cap**
   - Added `options` passthrough on `ChatRequest` -> `route_chat_request` -> Ollama `/api/chat`, and `RAG_ANSWER_MAX_TOKENS` applied to RAG/direct-chat generations (not `code` mode).
   - Why: bounds generation time on top of the existing 2-4 sentence prompt instruction.

4. **Context budget: skip-not-stop**
   - `_dedupe_and_cap_context` now skips an over-budget chunk and keeps trying smaller candidates instead of stopping at the first overflow.
   - Why: better utilizes the existing `RAG_CONTEXT_MAX_CHARS`/`RAG_CONTEXT_MAX_CHUNKS` budget without raising it.

5. **In-process retrieval + embedding caching**
   - Added `app/rag_cache.py` (TTL + LRU, no external cache store). `embed_text()` caches by `(model, text)`; `_prepare_context()` caches the full retrieval/context result by `(question, top_k, metadata_filters, fusion weights)`.
   - Repeated questions skip rewrite + embed + retrieval entirely; the final answer is always freshly generated. Config: `ENABLE_RAG_CACHE`, `RAG_CACHE_TTL_SECONDS`, `RAG_CACHE_MAX_SIZE`, `EMBEDDING_CACHE_MAX_SIZE`.
   - Why: avoids repeated full-collection BM25 scans and redundant Nomic calls for common questions, without risking stale results across different filters.

6. **Consistent `keep_alive` on every Ollama call**
   - `/api/chat` (`_call_ollama_chat`) and `/api/embeddings` (`embed_text`) now pass `keep_alive` alongside the two `/api/generate` call sites that already did.
   - Why: the `/api/chat` path (used for every real answer) previously did not honor `OLLAMA_KEEP_ALIVE`, undermining the warm-model strategy.

7. **Multimodal knowledge-base ingestion**
   - `app/rag_ingest.py` now also loads standalone images (`KB_IMAGE_EXTENSIONS`) and extracts embedded images from PDFs (via `pypdf`'s `page.images`) into `knowledge_base/_extracted_images/`.
   - New `app/vision.py` calls `VISION_MODEL` (LLaVA) to caption each image; `scripts/ingest_kb.py` embeds that caption text with Nomic (LLaVA does not replace Nomic) and stores it in Chroma with `{"type": "image", "source": <path>, "description": <caption>}` metadata alongside the existing text-chunk metadata shape (`"type": "text"`).
   - Why: brings diagrams/screenshots/PDF figures into the same retrievable index as text, without changing the retrieval algorithm.

8. **Query-time LLaVA, only when relevant**
   - `_prepare_context` checks retrieved chunk metadata for `type == "image"`; only then does it load the original image from disk and call LLaVA again with a question-focused prompt, replacing the generic indexed caption with a fresh, targeted one (`timings.vision_s`).
   - Plain text questions never invoke LLaVA — the fast path is unchanged.
   - Why: keeps the common (text-only) path fast while still supporting image-grounded answers.

`hybrid_retrieve`'s dense+BM25 fusion algorithm and the 70/30 dense/keyword weighting are unchanged by this pass.

---

## Team Beta: STT → RAG → LLM → TTS Voice Pipeline

Adds a full spoken-conversation loop on top of the existing RAG/LLM backend: mic audio in, Whisper transcription, the existing RAG/LLM pipeline for the answer, Coqui XTTS-v2 speech out.

**Deployment**: everything (denoising, Whisper, RAG, LLM, XTTS-v2) runs server-side on this GPU workstation. The Jetson Nano (`jetson_client/`) is a thin client — mic capture (PyAudio) + speaker playback (sounddevice) + a network hop over Tailscale — no ML models run on the edge device.

1. **New `WS /ws/voice` endpoint** (`app/main.py`, orchestration in `app/voice_pipeline.py`)
   - Mints one `session_id` per connection and reuses it for every utterance on that connection, reusing the same `conversation_graph.memory_store` session memory `/chat` uses — so multi-turn continuity isn't a separate mechanism. This differs from `/ws/rag`, which mints a fresh session per message and has no memory.
   - Per utterance: `denoise()` → `transcribe()` → send `{"type":"transcript"}` → `_prepare_context()` (same RAG retrieval `/chat`/`/rag/ask` use) → send `{"type":"meta"}` → **streamed** generation, synthesizing and sending each sentence as soon as it's ready (`{"type":"sentence_audio_meta"}` + binary audio frame, repeated) → `{"type":"done", "answer": "<full text>", "timings": {...}}`.
   - Why: reuses the existing hybrid-retrieval RAG pipeline and session memory rather than duplicating them for voice.

1b. **Sentence-by-sentence streaming, not one wait-for-everything response** (`app/voice_pipeline.py`, `stream_chat_request` in `app/ollama_proxy.py`)
   - Real measurements showed TTS synthesis alone regularly took 5-17s depending on answer length — waiting for the *entire* answer to generate, then synthesizing it *entirely*, then sending one blob meant visitors could wait 20+ seconds before hearing anything.
   - Now: `stream_chat_request()` streams LLM tokens via `/api/chat` with `stream: true`; `split_sentence_chunks()` buffers tokens until a sentence boundary (or a length safety valve); a producer/consumer `asyncio.Queue` lets sentence N+1's generation overlap with sentence N's synthesis instead of serializing them. First audio now arrives after roughly one sentence's worth of work, not the whole answer — tracked explicitly as `timings.time_to_first_audio_s`.
   - Explicitly not in this pass: barge-in/interruption. The mic still stays paused during playback; only the response's *shape* (streamed sentences vs. one blob) changed. Both `jetson_client/voice_client.py` and `laptop_ui/index.html` were updated to play a queue of incoming chunks back-to-back instead of assuming exactly one.
   - Why: total compute time is essentially unchanged, but perceived latency — the number that actually matters for whether this feels responsive — drops substantially.

2. **Server-side STT/denoise/TTS wrappers** (`app/voice_stt.py`, `app/voice_denoise.py`, `app/voice_tts.py`)
   - `faster-whisper`, DeepFilterNet, and Coqui XTTS-v2 (built-in speaker) respectively. Each lazily loads its model once and keeps it resident (mirrors the `OLLAMA_KEEP_ALIVE` warm-model strategy) — per-request loads would blow the latency budget.
   - Heavy imports (`torch`, `faster_whisper`, `df`, `TTS`) are deferred inside the loader/inference functions so the modules — and the rest of the app — import fine even without `requirements-voice.txt` installed.
   - Why: keeps the default install lightweight while making the voice modules independently testable (see Testing below).

3. **Jetson Nano thin client** (`jetson_client/voice_client.py`)
   - Simple energy-threshold silence detection to segment one utterance at a time (no ML VAD on the edge device), sends it over `websockets` to `/ws/voice`, plays the returned audio via `sounddevice`. Deliberately excludes torch/Whisper/TTS from `jetson_client/requirements.txt`.

3b. **Browser voice test client** (`laptop_ui/index.html`, "🎤 Start Voice Chat" button)
   - Lets you exercise the full `/ws/voice` loop from any laptop's mic/speakers over Tailscale, without the Jetson hardware. Same silence-based utterance segmentation as the Jetson client, implemented with the Web Audio API (`ScriptProcessorNode` + manual linear-interpolation resampling to 16kHz PCM16, since the browser's native audio rate is usually 44.1/48kHz).
   - Reuses the server URL field already on the page — derives the voice endpoint by swapping `/ws/rag` for `/ws/voice`.
   - Requires a secure context (`https://` or `localhost`/`file://`) for `getUserMedia`, and requests `echoCancellation` to avoid the speaker output being picked back up by the mic.

4. **Testing** (`tests/`, first pytest infrastructure in this repo)
   - Fast unit tests (no GPU/models/hardware needed, run in CI): mock the model-loader seams in each voice wrapper to verify wiring, exercise the full `/ws/voice` message sequence (including the multi-sentence streamed protocol) via FastAPI's `TestClient`, unit-test `split_sentence_chunks()` and `run_voice_turn()` in isolation (`tests/test_voice_pipeline.py`, including a regression test that a mid-stream generation error can't hang the producer/consumer queue), and test `SessionMemoryStore` (pre-existing code, previously untested).
   - Slow/integration tests marked `@pytest.mark.integration` (skipped by default, need real Ollama/models/audio): the 20-question RAG accuracy gate, real Whisper/XTTS-v2 calls, WER-before/after-denoising comparison.

5. **Evaluation / profiling**
   - `evaluation/rag_eval_dataset.json` expanded from 3 stale questions (referencing files that don't even exist in `knowledge_base/`) to 20 questions grounded in the real KB content, with an explicit pass/fail accuracy gate in `scripts/evaluate_rag.py` (`--min-accuracy`, default 80%).
   - `scripts/evaluate_stt_wer.py` (new): WER with vs. without DeepFilterNet, via `jiwer`. Needs real recorded audio in `evaluation/stt_eval_dataset.json` (ideally at ~65dB simulated background noise) to produce meaningful numbers — ships as scaffolding.
   - `scripts/profile_voice_latency.py`: drives `/ws/voice` with a sample WAV and reports per-stage timings (`denoise_s`, `stt_s`, `retrieval_s`, `time_to_first_audio_s`, `generation_and_tts_s`, `total_s`) against the 4s target — reports separate PASS/FAIL verdicts for time-to-first-audio (the number streaming targets) and total time (informational; long answers may legitimately exceed 4s end-to-end), following the existing `scripts/profile_performance.py` JSON/CSV artifact convention.

**Not in this pass**: voice deps are *not* added to the default `Dockerfile`/`requirements.txt` (multi-GB torch/model downloads shouldn't be forced onto the existing text-only image) — install `requirements-voice.txt` separately when building a voice-enabled deployment.

---

## Persistent Session/Utterance Storage (PostgreSQL + TimescaleDB)

A durable, queryable conversation record — separate from and complementary to two things that already existed: the in-memory `SessionMemoryStore` (short-term LLM context, wiped on restart, no schema) and the JSONL `app/rag_logging.py` (RAG retrieval diagnostics, not a speaker-labeled transcript). This is the durable, replayable history.

1. **Schema** (`app/db/models.py`) — `sessions` (id, created_at, closed_at, source) and `utterances` (id, timestamp, session_id, speaker `human`/`robot`, text, optional `extra` JSONB). `utterances` is a **TimescaleDB hypertable** partitioned on `timestamp`, which is why its primary key is the composite `(id, timestamp)` rather than just `id` — hypertables require the partitioning column to be part of any unique/primary key constraint. Model class is named `ConversationSession`, not `Session`, to avoid clashing with SQLAlchemy's own `Session`/`AsyncSession`.
2. **API** (`app/session_api.py`, mounted under `/api/sessions`): `POST /api/sessions` (create), `POST /api/sessions/{id}/utterances` (append), `POST /api/sessions/{id}/close`, `GET /api/sessions/{id}` (replay — full session + all utterances as JSON, ordered by timestamp).
3. **Wired into `/ws/voice` live** (`app/main.py`, `app/voice_pipeline.py`) — this is what actually logs STT transcripts and LLM responses, not just an API nobody calls. The websocket's existing `session_id` (already used for `SessionMemoryStore`) is reused as the DB session id, so no new identifier was introduced. A session row is created on connect, closed in a `finally` block covering every disconnect path, and each turn logs one `human` utterance (the transcript) and one `robot` utterance (the full assembled answer, with `used_rag`/`sources` in `extra`). `/chat` is **not** wired up in this pass — the task explicitly asked for STT/LLM logging, which is voice-specific; an easy, obvious follow-up if wanted later.
4. **Never blocks or breaks a live voice turn** (`app/db/safe_logging.py`) — every DB write is fire-and-forget (`asyncio.create_task`, tracked in a module-level set so it can't be garbage-collected mid-flight) and wrapped so *any* exception is logged and swallowed, never raised. This was a deliberate fix, not just defensive boilerplate: awaiting these calls inline was measured to add up to the full `DATABASE_CONNECT_TIMEOUT_SECONDS` (default 2s) to every turn whenever Postgres is unreachable — directly eating into the same latency budget the streaming-TTS work targeted. `database_connect_timeout_seconds` itself exists so a genuine outage fails fast rather than hanging on the OS-level TCP timeout.
5. **Migrations** (Alembic, `alembic/`) — one initial migration creates both tables, enables the `timescaledb` extension, and calls `create_hypertable('utterances', 'timestamp')`. Alembic runs synchronously (`psycopg2`, already a dependency) even though the app is fully async at runtime — migrations don't need to be async, and it's simpler this way. `alembic/env.py` reads `DATABASE_URL` from `app.config` directly rather than needing a separate migration-only config.
6. **Docker Compose**: new `db` service (`timescale/timescaledb:latest-pg15`, named volume, healthcheck); `app` gets `depends_on: db: condition: service_healthy` and a `DATABASE_URL` pointing at the compose-network `db` hostname. Migrations are a **documented manual step** (`docker compose exec app alembic upgrade head`), not an auto-run entrypoint — simpler to reason about and debug than auto-migrating on every container start.
7. **Testing**: fast, DB-free tests for Pydantic schema validation (`tests/test_session_schemas.py`) and `safe_logging`'s error-swallowing/fire-and-forget behavior with a fake session factory (`tests/test_db_safe_logging.py`); `tests/test_session_api.py` (marked `@pytest.mark.integration`) drives the full create → append → close → replay lifecycle plus 404/422 error paths via `httpx.AsyncClient` against the real FastAPI app, guarded by a live-reachability fixture that skips cleanly without a real Postgres (same pattern as the existing Ollama-dependent integration tests).

**Verified in this sandbox**: all new models/schemas/routes import and compile cleanly; the initial migration's generated SQL was validated with `alembic upgrade head --sql` (offline mode, no live DB needed) and produces exactly the expected DDL including the hypertable conversion; `docker compose config` validates the full compose file. **Not verifiable here**: actual Postgres/TimescaleDB connectivity, the migration actually running, or the integration tests actually passing — Docker Desktop's engine isn't active in this sandbox (CLI present, daemon unreachable), so this needs the workstation, same as the rest of the voice-hardware-dependent work this session.

---

## Architecture Overview

### Runtime flow

- **API**: FastAPI (`app/main.py`) exposes:
  - `POST /chat`
  - `POST /agent`
  - `POST /rag/ask`
  - `WS /ws/rag`
  - `WS /ws/voice` (audio in/out; denoise → Whisper → RAG/LLM → XTTS-v2)
  - `GET /health`
  - `POST /api/sessions`, `POST /api/sessions/{id}/utterances`, `POST /api/sessions/{id}/close`, `GET /api/sessions/{id}` (persistent conversation storage, see below)
- **Conversation Orchestration**: LangGraph (`app/conversation_graph.py`) decides RAG vs direct chat and persists session memory.
- **Retriever**:
  - Dense vector retrieval from ChromaDB
  - Keyword scoring (BM25-like)
  - Weighted fusion + metadata filters
  - Context dedupe + budget cap
  - Retrieval/embedding cache (`app/rag_cache.py`) checked before hitting Chroma/Nomic
- **Multimodal ingestion**: text/PDF chunked as before; standalone images and PDF-embedded images are captioned by LLaVA (`app/vision.py`) and the caption is embedded by Nomic, so image content is retrievable through the same Chroma index (`metadata.type` = `"text"` or `"image"`).
- **Query-time vision**: if a retrieved chunk is an image, `_prepare_context` re-captions the original file with LLaVA using the visitor's actual question; text-only questions never call LLaVA.
- **Generation routing**:
  - primary model (single-GPU workstation deployment: one main chat model, no secondary)
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

`knowledge_base/` can contain `.txt`/`.md`/`.pdf` files as before, plus standalone images (`.png`/`.jpg`/`.jpeg`/`.webp`/`.bmp`, see `KB_IMAGE_EXTENSIONS`) and PDFs with embedded diagrams/screenshots — both are captioned via `VISION_MODEL` (LLaVA) and indexed alongside text. Requires the vision model to be pulled, e.g. `ollama pull llava:7b`.

### 4) Run API

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 5) (Optional) Voice pipeline + tests

```bash
pip install -r requirements-voice.txt   # faster-whisper, DeepFilterNet, Coqui XTTS-v2, jiwer -- heavy, GPU-optional
pip install -r requirements-dev.txt     # pytest, pytest-asyncio, websockets (for profiling)
ollama pull llava:7b                    # if not already pulled for multimodal ingestion

pytest tests/ -m "not integration"      # fast, no models/hardware required
pytest tests/ -m integration            # needs live Ollama + ingested KB + STT/TTS models
```

For the Jetson Nano side, see `jetson_client/` — a separate, lightweight `requirements.txt` (no torch) since it only captures/plays audio and talks to `/ws/voice` over the network.

### 6) (Optional) Persistent session storage (PostgreSQL + TimescaleDB)

The rest of the app works fine without this — a missing/unreachable database only disables the `/api/sessions/*` endpoints and skips logging voice transcripts (see `app/db/safe_logging.py`); it never breaks RAG/chat/voice itself.

```bash
docker compose up -d db                 # or run your own PostgreSQL 15 + TimescaleDB and set DATABASE_URL in .env
alembic upgrade head                    # creates sessions/utterances and converts utterances to a hypertable
```

#### Voice pipeline install notes (read before reporting a `df`/`torchaudio`/`TTS` import error)

`requirements-voice.txt` pins `torch`/`torchaudio`/`numpy`/`transformers` to specific ranges and uses `coqui-tts` instead of the old `TTS` package — these aren't arbitrary, they're the result of working through a real, verified conflict chain on the actual deployment GPU (see the comments at the top of that file for why each pin exists). Loosening any of them will likely reintroduce one of these:

- `deepfilternet==0.5.6` (the latest PyPI release) still imports `torchaudio.backend.common.AudioMetaData`, a path removed from `torchaudio`. **This one line needs a one-time manual patch after every fresh `pip install`** (the fix exists on DeepFilterNet's GitHub `main` but was never released to PyPI):
  ```bash
  sed -i 's/from torchaudio\.backend\.common import AudioMetaData/from torchaudio import AudioMetaData/' \
    "$(python -c "import df, os; print(os.path.dirname(df.__file__))")/io.py"
  ```
- If you see `ImportError: cannot import name 'Coqpit' from 'coqpit' (unknown location)`, an old `coqpit` install is shadowing `coqpit-config`'s namespace: `pip uninstall coqpit -y && pip install --force-reinstall --no-cache-dir coqpit-config`.
- If Whisper errors with `Requested float16 compute type, but the target device ... do not support efficient float16 computation`, your GPU (older/Pascal-class cards especially) doesn't have efficient native fp16 — `STT_COMPUTE_TYPE` already defaults to `int8_float16` for this reason; fall back further to `int8` in `.env` if needed.

After installing, verify all of this in one go before starting the server:
```bash
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python3 -c "import df; print('deepfilternet OK')"
python3 -c "import transformers; print('torch usable:', transformers.utils.is_torch_available())"
python3 -c "from TTS.api import TTS; print('coqui-tts OK')"
```

---

## Docker Run

### Build + run with compose

```bash
cp .env.example .env
docker compose up --build -d
docker compose exec app alembic upgrade head   # one-time (and after any future schema change)
```

Services:
- API: `http://localhost:8000`
- `db`: PostgreSQL 15 + TimescaleDB, port `5432`, named volume `db_data` for persistence
- Volumes:
  - `./chroma_store` -> Chroma persistence
  - `./logs` -> RAG logs
  - `./artifacts` -> benchmark/eval/profile outputs

> Set `OLLAMA_LOCAL_BASE_URL` / `OLLAMA_REMOTE_URLS` in `.env` to host or remote Ollama endpoints.
> Migrations are a deliberate manual step (`alembic upgrade head`), not run automatically on container start — simpler to reason about and debug than auto-migrating on every restart.

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

### Voice pipeline (WS /ws/voice)

Not curl-able (binary WS frames in/out). Drive it with a sample WAV via the profiling script, or connect the Jetson client:

```bash
python scripts/profile_voice_latency.py --ws-url ws://localhost:8000/ws/voice --audio sample_utterance.wav
```

### Session storage (create → append → close → replay)

```bash
SESSION_ID=$(curl -s -X POST http://localhost:8000/api/sessions -H "Content-Type: application/json" \
  -d '{"source":"voice"}' | jq -r .id)

curl -s -X POST http://localhost:8000/api/sessions/$SESSION_ID/utterances -H "Content-Type: application/json" \
  -d '{"speaker":"human","text":"What does the robot do?"}' | jq

curl -s -X POST http://localhost:8000/api/sessions/$SESSION_ID/utterances -H "Content-Type: application/json" \
  -d '{"speaker":"robot","text":"It presents company info and answers FAQs.","extra":{"used_rag":true}}' | jq

curl -s -X POST http://localhost:8000/api/sessions/$SESSION_ID/close | jq

curl -s http://localhost:8000/api/sessions/$SESSION_ID | jq   # full replay
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
python scripts/evaluate_rag.py --dataset evaluation/rag_eval_dataset.json --min-accuracy 0.8
```

20-question dataset grounded in the real `knowledge_base/` content. Exits non-zero if accuracy (RAG used + reference keywords covered) falls below `--min-accuracy`.

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

## Voice Pipeline Evaluation (STT WER + Latency)

```bash
# WER with vs. without DeepFilterNet -- needs real recorded audio in
# evaluation/stt_eval_dataset.json first (see the file for the expected shape)
python scripts/evaluate_stt_wer.py --dataset evaluation/stt_eval_dataset.json

# End-to-end /ws/voice latency against the 4s target
python scripts/profile_voice_latency.py --audio sample_utterance.wav --requests 5
```

Artifacts:
- `artifacts/evaluation/stt_wer_*.{json,csv}`
- `artifacts/profiling/voice_latency_*.{json,csv}`

---

## Troubleshooting

- **`RAG pipeline failed`**: verify Ollama is reachable and run `python scripts/ingest_kb.py`.
- **No useful retrieval context**: tune `RAG_DENSE_WEIGHT`, `RAG_KEYWORD_WEIGHT`, `RAG_MIN_FUSION_SCORE`, or metadata filters.
- **Latency spikes**: profile with `scripts/profile_performance.py` (text) or `scripts/profile_voice_latency.py` (voice), then adjust model size, `REQUEST_TIMEOUT_SECONDS`, and context caps.
- **Docker cannot reach Ollama**: use reachable host in `.env` (often `http://host.docker.internal:11434` on Docker Desktop).
- **`/ws/voice` errors immediately**: confirm `requirements-voice.txt` is installed and the STT/TTS/denoise models have been downloaded (first call triggers the download if not cached) — check server logs for the specific stage that failed.
- **`/api/sessions/*` returns 500 / a warning like "Failed to persist ... utterance"**: PostgreSQL isn't reachable or migrations haven't run — `docker compose up -d db` then `docker compose exec app alembic upgrade head` (or `alembic upgrade head` locally). This never breaks RAG/chat/voice itself, only session persistence.
- **`relation "sessions" does not exist`**: migrations haven't been run yet — `alembic upgrade head`.

---

## Key Files Added/Updated

- `app/conversation_graph.py`
- `app/hybrid_retriever.py`
- `app/rag_pipeline.py`
- `app/rag_ingest.py`
- `app/rag_cache.py`
- `app/vision.py`
- `app/voice_stt.py`
- `app/voice_denoise.py`
- `app/voice_tts.py`
- `app/voice_pipeline.py`
- `jetson_client/voice_client.py`
- `app/db/models.py`, `app/db/database.py`, `app/db/crud.py`, `app/db/safe_logging.py`
- `app/session_api.py`
- `alembic.ini`, `alembic/env.py`, `alembic/versions/`
- `tests/` (pytest suite)
- `app/schemas.py`
- `app/config.py`
- `scripts/benchmark_embeddings.py`
- `scripts/evaluate_rag.py`
- `scripts/evaluate_stt_wer.py`
- `scripts/profile_performance.py`
- `scripts/profile_voice_latency.py`
- `evaluation/embedding_benchmark_dataset.json`
- `evaluation/rag_eval_dataset.json`
- `evaluation/stt_eval_dataset.json`
- `requirements-voice.txt`
- `requirements-dev.txt`
- `pytest.ini`
- `Dockerfile`
- `docker-compose.yml`
- `.dockerignore`
