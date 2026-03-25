"""
output/api.py

FastAPI backend for KIRA.
Deployed on Railway — handles pipeline execution and report storage.

Streamlit Cloud calls this API over HTTP.
Reports are saved to S3 for persistence across Railway restarts.

Run locally:
    uvicorn output.api:app --reload --port 8000

Endpoints:
    GET  /health              — confirm API is running
    POST /run-pipeline        — trigger pipeline, save to S3
    GET  /report/{run_id}     — retrieve report from S3
    GET  /reports             — list all reports
    POST /feedback            — submit thumbs up/down
    GET  /feedback            — get all feedback
"""

import json
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import get_settings
from scripts.run_pipeline import run_pipeline
from infra.s3 import upload_report, download_report, list_reports

settings = get_settings()

app = FastAPI(
    title="KIRA API",
    description="Knowledge Intelligence & Reporting Agent",
    version="0.1.0",
)

# Allow Streamlit Cloud to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response models ─────────────────────────────────────────

class PipelineRequest(BaseModel):
    client_id: str
    query: str = "Generate a weekly performance report with anomaly analysis"


class FeedbackRequest(BaseModel):
    run_id: str
    rating: str       # "up" or "down"
    comment: str = ""


class PipelineResponse(BaseModel):
    run_id: str
    client_id: str
    completed_agents: list[str]
    anomalies_detected: int
    compliance_passed: bool | None
    compliance_confidence: float | None
    final_report: str
    errors: list[str]
    s3_key: str = ""


# ── In-memory feedback store ──────────────────────────────────────────
# In production this would be a database.
# Railway's filesystem is ephemeral — feedback stored in memory
# persists for the lifetime of the process only.
_feedback: dict = {}


# ── Endpoints ─────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": settings.version,
        "environment": settings.environment,
        "vector_backend": settings.vector_store_backend,
        "model": settings.openai_chat_model,
        "memory_enabled": settings.mem0_enabled,
        "s3_enabled": settings.has_s3,
    }


@app.post("/run-pipeline", response_model=PipelineResponse)
def trigger_pipeline(request: PipelineRequest):
    """
    Triggers the full agent pipeline for a client.
    Saves the report to S3 for persistence.
    """
    valid_clients = ["natura", "vitalblend", "peakgear", "lumina"]
    if request.client_id not in valid_clients:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown client_id. Valid: {valid_clients}",
        )

    state = run_pipeline(
        client_id=request.client_id,
        query=request.query,
    )

    report_data = {
        "run_id": state.run_id,
        "client_id": state.client_id,
        "query": state.query,
        "started_at": state.started_at,
        "final_report": state.final_report,
        "report_draft": state.report_draft,
        "anomalies": [
            {
                "metric": a.metric,
                "description": a.description,
                "deviation_pct": a.deviation_pct,
            }
            for a in state.anomalies
        ],
        "compliance_passed": state.compliance.passed if state.compliance else None,
        "compliance_confidence": state.compliance.confidence if state.compliance else None,
        "errors": state.errors,
        "completed_agents": state.completed_agents,
    }

    # Save to S3
    s3_key = upload_report(report_data)

    return PipelineResponse(
        run_id=state.run_id,
        client_id=state.client_id,
        completed_agents=state.completed_agents,
        anomalies_detected=len(state.anomalies),
        compliance_passed=state.compliance.passed if state.compliance else None,
        compliance_confidence=state.compliance.confidence if state.compliance else None,
        final_report=state.final_report,
        errors=state.errors,
        s3_key=s3_key,
    )


@app.get("/report/{client_id}/{run_id}")
def get_report(client_id: str, run_id: str):
    """Retrieves a stored report from S3."""
    report = download_report(client_id=client_id, run_id=run_id)
    if not report:
        raise HTTPException(
            status_code=404,
            detail=f"Report {run_id} not found for {client_id}."
        )
    return report


@app.get("/reports")
def get_reports(client_id: str | None = None):
    """Lists all reports from S3, optionally filtered by client."""
    return list_reports(client_id=client_id)


@app.post("/feedback")
def submit_feedback(request: FeedbackRequest):
    """Logs thumbs up/down feedback on a report."""
    if request.rating not in ("up", "down"):
        raise HTTPException(
            status_code=400,
            detail="rating must be 'up' or 'down'",
        )
    _feedback[request.run_id] = {
        "rating": request.rating,
        "comment": request.comment,
        "timestamp": datetime.now().isoformat(),
    }
    return {"status": "saved", "run_id": request.run_id, "rating": request.rating}


@app.get("/feedback")
def get_all_feedback():
    """Returns all submitted feedback."""
    return _feedback