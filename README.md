# KIRA — Knowledge Intelligence & Reporting Agent

A production multi-agent AI platform that automates performance analysis,
anomaly detection, and compliance-checked report generation for e-commerce agencies.

Built as a portfolio project to demonstrate end-to-end AI engineering —
from embeddings and vector retrieval to agent orchestration, memory, and deployment.

**Live demo:** https://ogeanni-kira-outputapp-irgjbk.streamlit.app/
**API:** https://kira-api-pvpl.onrender.com/docs 

---

## The Problem

E-commerce agencies managing multiple clients drown in unstructured knowledge —
SOPs, meeting transcripts, compliance documents, analysis frameworks — scattered
across files and people's heads. Analysts spend 2-3 hours per client per week
retrieving context, writing reports, and checking compliance before any strategic
work can begin. This doesn't scale.

**Before KIRA:** An analyst pulls data manually, searches Google Drive for the
right SOP, writes a narrative report from scratch, checks it against compliance
guidelines, and formats it for the client. ~2.5 hours per client per week.

**After KIRA:** A scheduled pipeline retrieves relevant context semantically,
detects anomalies against a rolling baseline, generates a structured report, and
validates it against compliance rules automatically. ~60 seconds per client.

---

## Architecture

```
Unstructured docs (SOPs, transcripts, frameworks)
        │
        ▼
  Chunking pipeline (sentence_window strategy — chosen by Ragas eval)
        │
        ▼
  OpenAI text-embedding-3-small → Pinecone (namespace per client)
        │
        ▼
  MCP Server (retrieve_context · query_metrics · read_memory)
        │
        ▼
  Research Agent → Analysis Agent → Report Agent → Compliance Agent
        │                │                              │
   RAG retrieval    Rolling baseline              GPT-4o check against
   via Pinecone     anomaly detection             compliance docs
        │
        ▼
  Streamlit UI / FastAPI / AWS S3
```

**Key design decisions:**

- **Sentence-window chunking** — embeds one sentence for precise matching,
  returns 5-sentence window for rich LLM context. Chosen over fixed-size and
  recursive after benchmarking all three with Ragas (faithfulness: 0.917).

- **Pinecone namespaces** — one namespace per client. A query for Natura never
  retrieves VitalBlend data. Backed by an abstract `VectorStore` interface so
  ChromaDB (dev) and Pinecone (prod) are a one-line config swap.

- **MCP tool protocol** — agents never import retriever or pandas directly.
  They call tools through the MCP server. Clean separation of concerns,
  swappable implementations.

- **Per-metric anomaly thresholds** — ACOS uses a 20% threshold, revenue/sessions
  use 40%. Flat thresholds fail on tight metrics. Rolling 14-day baseline avoids
  the anomaly contaminating its own reference window.

- **LoRA stub** — compliance agent checks `settings.has_lora` before calling GPT-4o-mini.
  When the Phi-3-mini adapter is trained and loaded, it fires first as a
  zero-cost classifier. Uncertain cases escalate to GPT-4o-mini. The pipeline
  degrades gracefully without it.

---

## Tech Stack

| Layer             | Technology                                        |
|-------------------|---------------------------------------------------|
| Embeddings        | OpenAI text-embedding-3-small                     |
| Vector store      | Pinecone (prod) · ChromaDB (dev)                  |
| LLM               | GPT-4o-mini                                       |
| Agent framework   | Custom pipeline with Pydantic state               |
| Tool protocol     | MCP (Model Context Protocol)                      |
| Memory            | Mem0 + ChromaDB (cross-session episodic memory)   |
| Retrieval eval    | Ragas (faithfulness · context precision)          |
| Observability     | LangSmith (tracing) · cost tracking per run       |
| API               | FastAPI + uvicorn                                 |
| UI                | Streamlit                                         |
| Storage           | AWS S3                                            |
| Deployment        | Render (live) · EC2 + Nginx + systemd (documented)|

---

## Production Metrics

| Metric                            | Value                         |
|-----------------------------------|-------------------------------|
| Retrieval faithfulness (Ragas)    | 0.917 (threshold: 0.85)       |
| Context precision (Ragas)         | 0.933 (threshold: 0.80)       |
| Live retrieval faithfulness       | 0.867 ✓                       |
| Cost per pipeline run             | ~$0.003                       |
| End-to-end latency                | ~60s                          |
| Anomaly detection                 | 4/4 injected anomalies caught |
| Compliance pass rate              | 100% on synthetic dataset     |

---

## Anomaly Detection

Four clients, four anomaly types, all detected:

| Client        | Anomaly               | Deviation                     | Detected  |
|---------------|-----------------------|-------------------------------|-----------|
| Lumina        | Listing suppression   | Sessions -89%, Revenue -84%   | ✓         |
| VitalBlend    | ACOS spike            | ACOS +30% above baseline      | ✓         |
| PeakGear      | Seasonal spike        | Sessions +42% above baseline  | ✓         |
| Natura        | Buy Box loss          | Buy Box -57%                  | ✓ (day 45)|

Detection uses a rolling 14-day baseline with per-metric thresholds.
The rolling window excludes the anomaly day from its own baseline —
preventing the anomaly from contaminating the reference it's measured against.

---

## Project Structure

```
kira/
├── config/
│   └── settings.py          # Central config — all env vars in one place
├── ingestion/
│   ├── chunker.py            # 3 chunking strategies with Ragas benchmarking
│   ├── embedder.py           # Embedding cache + cost tracking
│   └── eval_chunking.py      # Strategy selection via Ragas scores
├── knowledge/
│   ├── vector_store.py       # Abstract interface — backend swap = 1 env var
│   ├── chroma_store.py       # ChromaDB (dev)
│   ├── pinecone_store.py     # Pinecone (prod) with namespace isolation
│   └── retriever.py          # Hybrid retrieval — merges global + client namespaces
├── intelligence/
│   ├── mcp_server.py         # MCP tools: retrieve_context, query_metrics, read_memory
│   ├── llm_client.py         # OpenAI wrapper with retry + cost tracking
│   ├── memory.py             # Mem0 cross-session memory
│   └── lora/                 # Phi-3-mini LoRA compliance classifier (stub)
├── agents/
│   ├── __init__.py           # PipelineState — Pydantic validated handoff packet
│   ├── research_agent.py     # RAG retrieval via MCP
│   ├── analysis_agent.py     # Metrics + rolling baseline anomaly detection
│   ├── report_agent.py       # Structured narrative report generation
│   └── compliance_agent.py   # Policy validation — LoRA gate + GPT-4o fallback
├── output/
│   ├── app.py                # Streamlit UI
│   └── api.py                # FastAPI — POST /run-pipeline, GET /report, POST /feedback
├── infra/
│   └── s3.py                 # S3 upload/download/list with explicit regional endpoint
├── observability/
│   └── tracer.py             # LangSmith tracing — graceful fallback when disabled
├── scripts/
│   ├── generate_data.py      # Synthetic data: 10 docs, 90-day metrics, compliance examples
│   ├── ingest_docs.py        # Chunk + embed + load into vector store
│   ├── run_pipeline.py       # Full agent pipeline with memory + cost summary
│   ├── pull_data_from_s3.py  # Container startup: pull data, ingest if Pinecone empty
│   └── eval_retrieval.py     # Live Ragas eval — quality gate against thresholds
├── deploy/
│   ├── EC2_DEPLOYMENT.md     # Step-by-step EC2 + Nginx + systemd deployment guide
│   ├── nginx.conf            # Reverse proxy config with timeout tuning
│   ├── kira-api.service      # systemd service — restart on crash, boot on start
│   └── deploy.sh             # One-shot setup script for fresh Ubuntu instance
├── Dockerfile                # Container — pulls data from S3 at startup
└── requirements.txt          # Direct dependencies only (not pip freeze)
```

---

## Running Locally

**Prerequisites:** Python 3.10+, OpenAI API key, Pinecone account

```bash
# Clone and set up environment
git clone https://github.com/Ogeanni/kira.git
cd kira
conda create -n kira python=3.10 -y
conda activate kira
pip install -r requirements.txt

# Configure
cp .env.template .env
# Add your API keys to .env

# Generate synthetic data
python scripts/generate_data.py

# Chunk, embed, ingest into ChromaDB (local dev)
python ingestion/chunker.py
python ingestion/embedder.py
python scripts/ingest_docs.py

# Run the pipeline
python scripts/run_pipeline.py --client lumina

# Run the Streamlit UI
streamlit run output/app.py
```

---

## Deployment

**Current:** FastAPI on Render, UI on Streamlit Cloud, data on AWS S3.

```bash
# API health check
curl https://kira-api-pvpl.onrender.com/health

# Trigger a pipeline run
curl -X POST https://kira-api-pvpl.onrender.com/run-pipeline \
  -H "Content-Type: application/json" \
  -d '{"client_id": "lumina", "query": "weekly performance report"}'
```

**EC2 alternative:** Full AWS EC2 + Nginx + systemd deployment documented
in `deploy/EC2_DEPLOYMENT.md`. Includes nginx reverse proxy config,
systemd service file, firewall rules, and SSL setup with Certbot.
Instance not running to avoid ongoing costs — full config committed to repo.

---

## Retrieval Quality

Chunking strategy was chosen by benchmarking all three options with Ragas
on a held-out eval set of 5 questions with known ground truth:

| Strategy          | Faithfulness  | Relevancy | Precision |
|-------------------|---------------|-----------|-----------|
| sentence_window   | **0.917**     | **0.964** | **0.933** |
| fixed_size        | 0.900         | 0.944     | 0.900     |
| recursive         | 0.875         | 0.943     | 0.900     |

`sentence_window` wins because it embeds at sentence granularity for
precise matching but returns surrounding context for richer generation.
Fixed-size scored lower because 11/50 chunks cut mid-sentence, reducing
answer groundedness.

Live production eval (against Pinecone): faithfulness 0.867, precision 0.923.
Both above the threshold defined in `settings.py`. The eval script exits
with code 1 if either drops below threshold — usable as a CI quality gate.

---

## What I Learned Building This

**Library versioning is a real engineering problem.** Ragas 0.4.x changed
its entire API across three releases with no clean migration path. The right
call was to pin to 0.1.21. I don't fight a library mid-refactor, I pinned and move on.

**Flat thresholds fail on tight metrics.** A 40% deviation threshold catches
listing suppression events (sessions -89%) but misses ACOS spikes (+29%).
Per-metric thresholds with a rolling baseline is how real monitoring works.

**Abstract interfaces pay off immediately.** Writing `VectorStore` as an ABC
with ChromaDB and Pinecone implementations took 30 extra minutes upfront.
It paid back when switching from local to production was a one-line env var
change, not a rewrite.

**Synthetic data with known ground truth is better than real data for evals.**
Because the anomaly dates are controlled, precision and recall are measurable.
"The system detected the anomaly" is provable, not assumed.

---

## Author

Building production systems