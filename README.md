---
title: RoleMatch
emoji: 🎯
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 8000
pinned: false
---

# RoleMatch

A self-contained AI career agent that parses a resume, remembers a candidate's
preferences across a conversation, retrieves jobs by semantic similarity, ranks them
against those preferences, and explains each match.

It implements six core capabilities:

| # | Capability | Where |
|---|------------|-------|
| 1 | Resume parsing & profile extraction | [`app/services/resume_parser.py`](app/services/resume_parser.py), `POST /candidates` |
| 2 | Conversational agent with memory | [`app/services/agent.py`](app/services/agent.py), [`memory.py`](app/services/memory.py), `POST /candidates/{id}/chat` |
| 3 | Semantic search over jobs (embeddings) | [`app/services/embeddings.py`](app/services/embeddings.py), [`search.py`](app/services/search.py) (pgvector) |
| 4 | Retrieval + preference ranking | [`app/services/ranking.py`](app/services/ranking.py), [`tools.py`](app/services/tools.py) |
| 5 | LLM-generated explanations | [`app/services/explanations.py`](app/services/explanations.py) |
| 6 | Tool calling (salary/location/experience filters) | [`app/services/tools.py`](app/services/tools.py) + LLM function calling |
| — | Evaluation: Precision@K, Recall@K, NDCG@K | [`app/eval/`](app/eval/) |

## Stack

FastAPI · PostgreSQL + pgvector · Sentence Transformers (local embeddings) ·
OpenAI (agent + explanations, pluggable) · Docker.

## Design notes

- **LLM is pluggable.** `LLM_PROVIDER=mock` (default) runs the full pipeline offline
  with a deterministic stand-in, so the app boots with no API key. Set
  `LLM_PROVIDER=openai` and `OPENAI_API_KEY=...` for real extraction, explanations,
  and function calling. See [`app/llm/`](app/llm/). Set `OPENAI_BASE_URL` to use any
  OpenAI-compatible provider instead of OpenAI — e.g. Groq
  (`https://api.groq.com/openai/v1`, model `openai/gpt-oss-120b`) or OpenRouter.
  If the provider errors (no credits, rate limit), the app degrades to the offline
  agent instead of failing.
- **Embeddings run locally** via `all-MiniLM-L6-v2` (384-dim) — no key needed. Jobs are
  embedded on insert and searched with pgvector cosine distance.
- **Memory** is persisted: structured `preferences` on the candidate plus a
  `chat_messages` history. The agent merges new preferences each turn.
- **Ranking** blends semantic relevance (0.6) with a preference-fit score (0.4) and
  drops any job that violates a stated dealbreaker.

## Quick start (Docker)

```bash
cp .env.example .env            # optional: set LLM_PROVIDER=openai + OPENAI_API_KEY
docker compose up --build       # API on http://localhost:8000
# in another shell, seed jobs into the running stack:
docker compose exec api python -m scripts.seed_jobs
```

## Quick start (local Python)

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# Start Postgres+pgvector (compose just for the db):
docker compose up -d db
cp .env.example .env
python -m scripts.seed_jobs
uvicorn app.main:app --reload
```

Open http://localhost:8000/docs for the interactive API.

## End-to-end example

```bash
# 1. Create a candidate from the sample resume
curl -s -F "file=@data/sample_resume.txt" http://localhost:8000/candidates
# -> {"id": 1, "title": "Senior Backend Engineer", "skills": [...], ...}

# 2. Chat — the agent extracts preferences, calls the filter tool, returns ranked matches
curl -s -X POST http://localhost:8000/candidates/1/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "I want a remote backend role paying at least 160k"}'

# 3. Fetch recommendations directly
curl -s "http://localhost:8000/candidates/1/recommendations?k=5"
```

Each recommendation includes `score`, `semantic_score`, `preference_score`, and a
natural-language `explanation`.

## Evaluation

```bash
python -m scripts.seed_jobs      # ensure jobs are loaded
python -m app.eval.run_eval      # prints Precision@K, Recall@K, NDCG@K per case + mean
```

Metric implementations and unit tests:
[`app/eval/metrics.py`](app/eval/metrics.py), [`tests/test_metrics.py`](tests/test_metrics.py).

```bash
pytest -q        # metric unit tests (no DB required)
```

## API surface

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/candidates` | Create candidate from resume text or PDF/txt upload |
| GET  | `/candidates/{id}` | Fetch candidate profile |
| POST | `/jobs` | Add a job (auto-embedded) |
| GET  | `/jobs` | List jobs |
| POST | `/candidates/{id}/chat` | Converse; updates memory, runs tool-calling + ranking |
| GET  | `/candidates/{id}/recommendations` | Ranked, explained recommendations |
| GET  | `/health` | Liveness + active providers |

## What's intentionally out of scope

Auth, multi-tenant companies, the hiring side, resume tailoring, and background job
ingestion — this is a focused demo of the candidate-side recommendation core.

## License

MIT — see [LICENSE](LICENSE).
