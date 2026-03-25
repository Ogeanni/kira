"""
agents/analysis_agent.py

Second agent in the pipeline.
Job: query performance metrics, detect anomalies, write
     a plain English analysis narrative.

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
from intelligence.mcp_server import query_metrics
from intelligence.llm_client import get_llm
from config.settings import get_settings

settings = get_settings()

# Per-metric deviation thresholds (%)
# Different metrics have different normal variance ranges
THRESHOLDS = {
    "revenue_usd": 40,
    "sessions": 40,
    "buy_box_pct": 30,
    "acos": 20,           # tight — 20% ACOS deviation is significant
    "conversion_rate": 25,
}


def _detect_anomalies(df_rows: list[dict]) -> list[AnomalyFlag]:
    """
    Detects anomalies using a rolling baseline with per-metric thresholds.

    For each day, compares value against mean of preceding 14 days.
    Rolling window excludes current day so baseline is never
    contaminated by the anomaly it's trying to detect.
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

    # Deduplicate — keep worst deviation per metric
    seen: dict[str, AnomalyFlag] = {}
    for flag in flags:
        if flag.metric not in seen or flag.deviation_pct > seen[flag.metric].deviation_pct:
            seen[flag.metric] = flag

    return list(seen.values())


def run(state: PipelineState) -> PipelineState:
    """
    Queries 90 days of metrics, detects anomalies using rolling baseline,
    then generates a plain English analysis narrative.
    """
    print(f"  [Analysis] Querying metrics for {state.client_id}...")

    if not state.is_healthy:
        print(f"  [Analysis] Skipping — pipeline has errors.")
        return state

    try:
        raw = query_metrics(client_id=state.client_id, days=90, metric="all")
        data = json.loads(raw)

        if "error" in data:
            state.add_error("analysis_agent", data["error"])
            return state

        state.metrics_summary = data.get("summary", {})
        anomalies = _detect_anomalies(data.get("rows", []))
        state.anomalies = anomalies

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