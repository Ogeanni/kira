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
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import get_settings
from pipeline_runner import run_pipeline
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
    run_id:             str
    client_id:          Optional[str] = None
    event_type:         str = "explicit"
    rating:             Optional[str] = None   # up | down
    comment:            Optional[str] = None
    time_on_report:     Optional[int] = None   # seconds
    faithfulness_score: Optional[float] = None


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
    """
    Records feedback on a pipeline run.

    Accepts both explicit (thumbs up/down) and implicit signals
    (copy event, view duration, re-run detection).

    event_type:
      explicit      — user rating
      report_copied — user copied the report to clipboard
      report_viewed — report was viewed (time_on_report populated)
      report_rerun  — client report was re-run within 10 minutes
    """
    valid_event_types = {'explicit', 'report_copied', 'report_viewed', 'report_rerun'}
    if request.event_type not in valid_event_types:
        raise HTTPException(
            status_code=400,
            detail=f"event_type must be one of {valid_event_types}"
        )

    if request.event_type == 'explicit' and request.rating not in ('up', 'down', None):
        raise HTTPException(
            status_code=400,
            detail="rating must be 'up' or 'down'"
        )

    try:
        from infra.database import get_session, Feedback
        session = get_session()
        record = Feedback(
            run_id=request.run_id,
            client_id=request.client_id,
            event_type=request.event_type,
            rating=request.rating,
            comment=request.comment,
            time_on_report=request.time_on_report,
            faithfulness_score=request.faithfulness_score,
        )
        session.add(record)
        session.commit()
        session.close()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save feedback: {e}")

    return {
        "status": "recorded",
        "run_id": request.run_id,
        "event_type": request.event_type,
    }


@app.get("/feedback")
def get_feedback(client_id: Optional[str] = None, event_type: Optional[str] = None):
    """Returns feedback records, optionally filtered by client or event type."""
    try:
        from infra.database import get_session, Feedback
        session = get_session()
        query = session.query(Feedback).order_by(Feedback.created_at.desc())
        if client_id:
            query = query.filter(Feedback.client_id == client_id)
        if event_type:
            query = query.filter(Feedback.event_type == event_type)
        records = query.limit(100).all()
        session.close()
        return [
            {
                "id":                r.id,
                "run_id":            r.run_id,
                "client_id":         r.client_id,
                "event_type":        r.event_type,
                "rating":            r.rating,
                "comment":           r.comment,
                "time_on_report":    r.time_on_report,
                "faithfulness_score":r.faithfulness_score,
                "created_at":        r.created_at.isoformat() if r.created_at else None,
            }
            for r in records
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve feedback: {e}")