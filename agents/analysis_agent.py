"""
agents/analysis_agent.py

Second agent in the pipeline.
Job: query performance metrics, detect anomalies, write
     a plain English analysis narrative.

Data source: PostgreSQL (primary) → CSV fallback if DB unavailable.

Input:  PipelineState with retrieved_contexts
Output: PipelineState with metrics_summary, anomalies, analysis_narrative
"""

import json
import pandas as pd

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents import PipelineState, AnomalyFlag
from intelligence.llm_client import get_llm
from config.settings import get_settings

settings = get_settings()

THRESHOLDS = {
    "revenue_usd": 40,
    "sessions": 40,
    "buy_box_pct": 30,
    "acos": 20,
    "conversion_rate": 25,
}


def _get_metrics(client_id: str, days: int = 90) -> list[dict]:
    """
    Fetches metrics from PostgreSQL.
    Falls back to CSV via MCP tool if database is unavailable.
    Graceful fallback means the pipeline works in all environments.
    """
    try:
        from infra.database import get_session, get_client_metrics
        session = get_session()
        rows = get_client_metrics(session, client_id, days=days)
        session.close()
        if rows:
            return rows
        raise ValueError(f"No rows returned from database for {client_id}")
    except Exception as db_error:
        print(f"  [Analysis] DB unavailable ({db_error}) — falling back to CSV")
        from intelligence.mcp_server import query_metrics
        raw = query_metrics(client_id=client_id, days=days, metric="all")
        data = json.loads(raw)
        if "error" in data:
            raise RuntimeError(data["error"])
        return data.get("rows", [])


def _compute_summary(rows: list[dict]) -> dict:
    """Computes mean, min, max, latest for each numeric metric."""
    if not rows:
        return {}
    df = pd.DataFrame(rows)
    numeric_cols = [
        "revenue_usd", "ad_revenue_usd", "organic_revenue_usd",
        "units_sold", "sessions", "conversion_rate",
        "acos", "tacos", "buy_box_pct", "asp_usd",
    ]
    summary = {}
    for col in numeric_cols:
        if col not in df.columns:
            continue
        series = df[col].astype(float)
        summary[col] = {
            "mean": round(float(series.mean()), 4),
            "min": round(float(series.min()), 4),
            "max": round(float(series.max()), 4),
            "latest": round(float(series.iloc[-1]), 4),
        }
    return summary


def _detect_anomalies(df_rows: list[dict]) -> list[AnomalyFlag]:
    """
    Detects anomalies using a rolling baseline with per-metric thresholds.
    Rolling window excludes current day — baseline never contaminated
    by the anomaly it's measuring against.
    """
    if not df_rows:
        return []

    df = pd.DataFrame(df_rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    flags = []
    for metric, threshold in THRESHOLDS.items():
        if metric not in df.columns:
            continue
        series = df[metric].astype(float)
        baseline = series.shift(1).rolling(window=14, min_periods=5).mean()
        for i, (val, base) in enumerate(zip(series, baseline)):
            if pd.isna(base) or base == 0:
                continue
            deviation_pct = abs((val - base) / base) * 100
            if deviation_pct > threshold:
                direction = "above" if val > base else "below"
                date_str = df["date"].iloc[i].strftime("%Y-%m-%d")
                flags.append(AnomalyFlag(
                    metric=metric,
                    current_value=round(float(val), 4),
                    mean_value=round(float(base), 4),
                    deviation_pct=round(deviation_pct, 1),
                    description=(
                        f"{date_str}: {metric} was {deviation_pct:.0f}% {direction} "
                        f"its 14-day baseline ({val:.2f} vs {base:.2f})"
                    ),
                ))

    seen: dict[str, AnomalyFlag] = {}
    for flag in flags:
        if flag.metric not in seen or flag.deviation_pct > seen[flag.metric].deviation_pct:
            seen[flag.metric] = flag
    return list(seen.values())


def _save_anomalies_to_db(client_id: str, run_id: str, anomalies: list[AnomalyFlag]) -> None:
    """Persists detected anomalies to the database for historical tracking."""
    if not anomalies:
        return
    try:
        from infra.database import get_session, AnomalyRecord
        session = get_session()
        for a in anomalies:
            record = AnomalyRecord(
                run_id=run_id,
                client_id=client_id,
                metric=a.metric,
                current_value=a.current_value,
                baseline_value=a.mean_value,
                deviation_pct=a.deviation_pct,
                description=a.description,
            )
            session.add(record)
        session.commit()
        session.close()
    except Exception:
        pass   # DB write failure never breaks the pipeline


def run(state: PipelineState) -> PipelineState:
    """
    Queries 90 days of metrics from PostgreSQL, detects anomalies,
    generates plain English analysis narrative.
    """
    print(f"  [Analysis] Querying metrics for {state.client_id}...")

    if not state.is_healthy:
        print(f"  [Analysis] Skipping — pipeline has errors.")
        return state

    try:
        rows = _get_metrics(state.client_id, days=90)

        if not rows:
            state.add_error("analysis_agent", f"No metrics found for {state.client_id}")
            return state

        state.metrics_summary = _compute_summary(rows)
        anomalies = _detect_anomalies(rows)
        state.anomalies = anomalies

        _save_anomalies_to_db(state.client_id, state.run_id, anomalies)

        anomaly_text = (
            "\n".join(f"- {a.description}" for a in anomalies)
            if anomalies
            else "No significant anomalies detected in the 90-day window."
        )

        metrics_text = "\n".join(
            f"- {metric}: mean={stats['mean']}, latest={stats['latest']}, "
            f"min={stats['min']}, max={stats['max']}"
            for metric, stats in state.metrics_summary.items()
            if isinstance(stats, dict)
        )

        llm = get_llm(agent_name="analysis_agent")
        response = llm.chat(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an e-commerce analyst. Write a concise, factual "
                        "analysis narrative (3-5 sentences) based on the metrics "
                        "and context provided. Quantify everything. No filler phrases."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Client: {state.client_id}\n"
                        f"Query: {state.query}\n\n"
                        f"90-day metrics:\n{metrics_text}\n\n"
                        f"Anomalies detected:\n{anomaly_text}\n\n"
                        f"Relevant context:\n{state.context_block[:1500]}"
                    ),
                },
            ],
            temperature=0,
        )
        state.analysis_narrative = response.content

        state.mark_complete("analysis_agent")
        print(f"  [Analysis] Done. {len(anomalies)} anomalies detected.")

    except Exception as e:
        state.add_error("analysis_agent", str(e))
        print(f"  [Analysis] Error: {e}")

    return state