# KIRA Deployment Guide

## Required services

KIRA depends on four external services. All must be configured before
the pipeline will run.

```
OpenAI          — LLM and embeddings
Pinecone        — Vector store for knowledge base
Supabase        — PostgreSQL database (operational data + metrics)
Qdrant          — Vector store for episodic memory (runs as Docker container)
AWS S3          — Report storage (optional — pipeline degrades gracefully without it)
```

## Environment variables

Copy `.env.example` to `.env` and fill in all values.

```bash
# OpenAI
OPENAI_API_KEY=sk-...
OPENAI_CHAT_MODEL=gpt-4o
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

# Pinecone
PINECONE_API_KEY=...
PINECONE_INDEX_NAME=kira-knowledge
PINECONE_ENVIRONMENT=...

# Database — Supabase transaction pooler
# Use port 6543 (transaction pooler), NOT port 5432 (direct connection)
# KIRA uses NullPool — stateless connections — which matches transaction pooler
DATABASE_URL=postgresql://postgres.YOUR_PROJECT_REF:YOUR_PASSWORD@aws-1-us-east-1.pooler.supabase.com:6543/postgres

# Vector store backend
VECTOR_STORE_BACKEND=pinecone

# AWS S3 (optional)
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
S3_BUCKET_NAME=...

# Memory
MEM0_ENABLED=true
```

## Local development setup

### 1. Start required Docker containers

```bash
# PostgreSQL (local dev only — use Supabase in production)
docker run -d --name kira-postgres \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=kira \
  -p 5434:5432 \
  postgres:15

# Qdrant (required in all environments)
docker run -d --name kira-qdrant \
  -p 6333:6333 \
  -v $(pwd)/data/memory:/qdrant/storage \
  qdrant/qdrant
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Create database tables

```bash
python -c "
from dotenv import load_dotenv
load_dotenv()
from infra.database import create_tables
create_tables()
"
```

### 4. Load initial data

```bash
# Load clients and metrics
python -c "
from dotenv import load_dotenv
load_dotenv()
from infra.database import get_session, load_clients, load_metrics_from_csv
session = get_session()
load_clients(session)
load_metrics_from_csv(session)
session.commit()
session.close()
"

# Load tabular operational data
python ingestion/data_loader.py --all
```

### 5. Generate and ingest knowledge base

```bash
# Generate source documents in native formats
python scripts/generate_native_formats.py

# Run ingestion pipeline
python ingestion/pipeline.py --input data/raw/documents/ --client global
# Repeat with --client natura, vitalblend, peakgear, lumina for transcripts

# Embed and ingest to Pinecone with D2 representation
python scripts/embed_and_ingest_pipeline.py
```

### 6. Run the pipeline

```bash
python pipeline_runner.py --client natura --query "Weekly performance report"
```

### 7. Start the API

```bash
uvicorn output.api:app --host 0.0.0.0 --port 8000 --reload
```

API available at `http://localhost:8000`
Docs at `http://localhost:8000/docs`

## Production deployment (Render)

### Services to deploy

```
Web Service     — output/api.py (FastAPI)
                  Start command: uvicorn output.api:app --host 0.0.0.0 --port $PORT
```

### Environment variables on Render

Set all variables from the `.env` section above in Render's environment
variable dashboard. Do not commit `.env` to git.

Key differences from local:
- `DATABASE_URL` — use Supabase transaction pooler URL (port 6543)
- `VECTOR_STORE_BACKEND` — set to `pinecone`
- `MEM0_ENABLED` — set to `true`

### Qdrant on Render

Qdrant requires a persistent disk. Options:

**Option A — Render private service (recommended)**
Deploy Qdrant as a private service on Render:
- Docker image: `qdrant/qdrant`
- Port: 6333
- Persistent disk mounted at `/qdrant/storage`
- Set `QDRANT_HOST` environment variable to the private service URL

Update `intelligence/memory.py` MEMORY_CONFIG to use the host from env:

```python
import os
MEMORY_CONFIG = {
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "collection_name": "kira_memory",
            "host": os.getenv("QDRANT_HOST", "localhost"),
            "port": int(os.getenv("QDRANT_PORT", "6333")),
        },
    },
}
```

**Option B — Qdrant Cloud**
Use Qdrant's managed cloud service. Free tier available.
Set `QDRANT_HOST` and `QDRANT_API_KEY` environment variables.

## Database notes

### Why Supabase transaction pooler, not direct connection

KIRA uses SQLAlchemy with `NullPool` — each database operation opens
a connection and closes it immediately. This is stateless connection
behaviour.

Direct connection (port 5432) is designed for persistent connections
that stay open between requests. Using it with NullPool causes rapid
connect/disconnect cycles that exhaust PostgreSQL's connection limit.

Transaction pooler (port 6543, PgBouncer) handles rapid open/close
cycles correctly — it maintains a pool of persistent connections to
PostgreSQL and assigns them to application requests as needed.

**Always use port 6543 for KIRA.**

### Migrations

KIRA uses SQLAlchemy's `create_all()` for schema management — not Alembic
migrations. When adding new tables or columns:

1. Add the model class to `infra/database.py`
2. Run `create_tables()` — it is idempotent and safe to re-run
3. Existing tables and data are not affected

## Known operational requirements

```
Docker          — Required for Qdrant (memory persistence)
                  kira-qdrant container must be running for memory to work
                  Pipeline degrades gracefully if Qdrant is unavailable

Pinecone index  — Must exist before ingestion
                  Index name: kira-knowledge
                  Dimensions: 1536 (text-embedding-3-small)
                  Metric: cosine

ChromaDB        — Local development fallback for vector store
                  Set VECTOR_STORE_BACKEND=chromadb for offline dev
                  Currently out of sync with production chunks — use Pinecone
```

## Running all four clients

```bash
for client in natura vitalblend peakgear lumina; do
    python pipeline_runner.py \
      --client $client \
      --query "Weekly performance report with anomaly analysis"
done
```
