"""
output/app.py

Streamlit UI for KIRA.
Deployed on Streamlit Cloud — calls the Railway FastAPI backend over HTTP.

Run locally against local API:
    uvicorn output.api:app --port 8000  (terminal 1)
    streamlit run output/app.py          (terminal 2)

Run locally against Railway:
    Set KIRA_API_URL in .env to your Railway URL.
    streamlit run output/app.py
"""

import json
import os
import re
import time
from datetime import datetime

import httpx
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="KIRA",
    page_icon="📊",
    layout="wide",
)

# API URL — local dev or Railway production
API_URL = os.getenv("KIRA_API_URL", "http://localhost:8000")


# ── Session state ─────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "running" not in st.session_state:
    st.session_state.running = False
if "feedback" not in st.session_state:
    st.session_state.feedback = {}


# ── API helpers ───────────────────────────────────────────────────────

def call_pipeline(client_id: str, query: str) -> dict:
    """Calls POST /run-pipeline on the FastAPI backend."""
    response = httpx.post(
        f"{API_URL}/run-pipeline",
        json={"client_id": client_id, "query": query},
        timeout=120.0,   # pipeline can take 30-60s
    )
    response.raise_for_status()
    return response.json()


def call_health() -> dict:
    """Calls GET /health to check if the API is up."""
    try:
        response = httpx.get(f"{API_URL}/health", timeout=5.0)
        return response.json()
    except Exception:
        return {"status": "unreachable"}


def submit_feedback(run_id: str, rating: str) -> None:
    """Calls POST /feedback on the FastAPI backend."""
    try:
        httpx.post(
            f"{API_URL}/feedback",
            json={"run_id": run_id, "rating": rating},
            timeout=5.0,
        )
        st.session_state.feedback[run_id] = rating
    except Exception:
        pass


# ── Report rendering ──────────────────────────────────────────────────

def parse_metrics_table(report_text: str) -> pd.DataFrame | None:
    """
    Extracts ## Key Metrics section and converts bullet lines
    into a two-column DataFrame for st.dataframe display.
    """
    match = re.search(
        r"## Key Metrics\n(.*?)(?=\n##|\Z)", report_text, re.DOTALL
    )
    if not match:
        return None

    rows = []
    for line in match.group(1).strip().split("\n"):
        line = line.strip().lstrip("- ").strip()
        if line and ":" in line:
            metric, values = line.split(":", 1)
            rows.append({"Metric": metric.strip(), "Values": values.strip()})

    return pd.DataFrame(rows) if rows else None


def render_report(report_text: str) -> None:
    """Renders report with Key Metrics as a table, other sections as markdown."""
    sections = re.split(r"(## \w[\w\s&()/]*)", report_text)

    i = 0
    while i < len(sections):
        chunk = sections[i].strip()
        if not chunk:
            i += 1
            continue

        if chunk.startswith("## "):
            heading = chunk
            body = sections[i + 1].strip() if i + 1 < len(sections) else ""
            i += 2
            st.markdown(f"### {heading[3:]}")

            if "Key Metrics" in heading:
                df = parse_metrics_table(f"{heading}\n{body}")
                if df is not None:
                    st.dataframe(
                        df,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Metric": st.column_config.TextColumn("Metric", width="medium"),
                            "Values": st.column_config.TextColumn("Values", width="large"),
                        },
                    )
                else:
                    st.markdown(body)
            else:
                st.markdown(body)
        else:
            st.markdown(chunk)
            i += 1


# ── Sidebar ───────────────────────────────────────────────────────────
with st.sidebar:
    st.title("KIRA")
    st.caption("Knowledge Intelligence & Reporting Agent")
    st.divider()

    # API health check
    health = call_health()
    if health.get("status") == "ok":
        st.success("API connected")
        st.caption(f"Model: `{health.get('model', '—')}`")
        st.caption(f"Backend: `{health.get('vector_backend', '—')}`")
    else:
        st.error("API unreachable")
        st.caption(f"Endpoint: `{API_URL}`")

    st.divider()

    client_id = st.selectbox(
        "Client",
        options=["natura", "vitalblend", "peakgear", "lumina"],
    )

    query = st.text_area(
        "Query",
        value="Generate a weekly performance report with anomaly analysis",
        height=80,
    )

    run_btn = st.button(
        "Run pipeline",
        type="primary",
        use_container_width=True,
        disabled=st.session_state.running or health.get("status") != "ok",
    )

    st.divider()
    st.caption(f"API: `{API_URL}`")


# ── Main area ─────────────────────────────────────────────────────────
st.title("KIRA — Client Reports")

if run_btn:
    st.session_state.running = True
    with st.spinner(f"Running pipeline for {client_id}..."):
        start = time.time()
        try:
            result = call_pipeline(client_id=client_id, query=query)
            elapsed = round(time.time() - start, 1)
            st.session_state.history.insert(0, {
                "result": result,
                "elapsed": elapsed,
            })
        except Exception as e:
            st.error(f"Pipeline failed: {e}")
    st.session_state.running = False
    st.rerun()


# ── Report display ────────────────────────────────────────────────────
if not st.session_state.history:
    st.info("Select a client and click **Run pipeline** to generate a report.")
else:
    for entry in st.session_state.history:
        result = entry["result"]
        elapsed = entry["elapsed"]
        run_id = result["run_id"]
        anomalies = result.get("anomalies", [])

        with st.container(border=True):
            col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
            with col1:
                st.subheader(f"{result['client_id'].title()} — {run_id}")
            with col2:
                st.metric("Time", f"{elapsed}s")
            with col3:
                st.metric("Agents", f"{len(result.get('completed_agents', []))}/4")
            with col4:
                st.metric("Anomalies", len(anomalies))

            # Compliance
            passed = result.get("compliance_passed")
            confidence = result.get("compliance_confidence", 0)
            if passed is True:
                st.success(f"Compliance PASSED (confidence: {confidence:.0%})")
            elif passed is False:
                st.error("Compliance FAILED")

            # Anomalies table
            if anomalies:
                with st.expander(
                    f"{len(anomalies)} anomaly/anomalies detected",
                    expanded=True,
                ):
                    st.dataframe(
                        pd.DataFrame([
                            {
                                "Metric": a["metric"],
                                "Description": a["description"],
                                "Deviation": f"{a['deviation_pct']:.1f}%",
                            }
                            for a in anomalies
                        ]),
                        use_container_width=True,
                        hide_index=True,
                    )

            # Errors
            errors = result.get("errors", [])
            if errors:
                with st.expander(f"{len(errors)} error(s)"):
                    for e in errors:
                        st.error(e)

            # Report body
            final_report = result.get("final_report", "")
            if final_report:
                st.divider()
                render_report(final_report)
            else:
                st.warning("No final report — compliance failed or pipeline errored.")
                draft = result.get("report_draft", "")
                if draft:
                    with st.expander("View unapproved draft"):
                        render_report(draft)

            # Feedback
            st.divider()
            existing_rating = st.session_state.feedback.get(run_id)
            if existing_rating:
                emoji = "👍" if existing_rating == "up" else "👎"
                st.caption(f"Feedback submitted: {emoji}")
            else:
                st.caption("Was this report useful?")
                fb1, fb2, _ = st.columns([1, 1, 6])
                with fb1:
                    if st.button("👍", key=f"up_{run_id}"):
                        submit_feedback(run_id, "up")
                        st.rerun()
                with fb2:
                    if st.button("👎", key=f"down_{run_id}"):
                        submit_feedback(run_id, "down")
                        st.rerun()