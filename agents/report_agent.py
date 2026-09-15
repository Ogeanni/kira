"""
agents/report_agent.py

Third agent in the pipeline.
Job: generate a structured client-ready report.

Input:  PipelineState with analysis_narrative, metrics_summary, anomalies
Output: PipelineState with report_draft
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
    
from agents import PipelineState
from intelligence.llm_client import get_llm
from config.settings import get_settings

settings = get_settings()

REPORT_SYSTEM_PROMPT = """
You are a senior e-commerce analyst writing a weekly performance report.
Write in a professional but direct tone. No filler phrases.
Quantify everything. Structure your report exactly as follows:

## Executive Summary
3-5 sentences. Lead with the most important metric movement.
State direction, magnitude, and one causal explanation.

## Key Metrics
Bullet list of the most important metrics with values and WoW change if available.

## Anomalies
List any detected anomalies with root cause hypothesis.
If none, write: No anomalies detected this period.

## Recommended Actions
Maximum 3 action items. Each must have: what, why, expected impact.

Keep the total report under 400 words.

GROUNDING RULES — these override everything else:
1. Recommended actions must be grounded in the supporting context provided.
   If the context contains specific client rules, thresholds, or procedures,
   use them. Do not give generic advice when specific guidance exists.
2. If the context does not contain enough information to make a specific
   recommendation, say explicitly: "Insufficient context to recommend a
   specific action for [topic]."
3. Never use general Amazon best practices as a substitute for specific
   retrieved context. If it is not in the context, do not recommend it.
""".strip()


def run(state: PipelineState) -> PipelineState:
    print(f"  [Report] Generating report for {state.client_id}...")

    if not state.is_healthy:
        print(f"  [Report] Skipping — pipeline has errors.")
        return state

    # Handle uncertainty before checking analysis narrative
    # When the assembler signals low confidence, the system must
    # explicitly state what it cannot answer rather than generating
    # from insufficient evidence or failing silently
    if (state.assembled_context and
            state.assembled_context.uncertainty and
            not state.analysis_narrative):
        state.report_draft = (
            f"## Insufficient Evidence\n\n"
            f"I was unable to generate a report for this query.\n\n"
            f"**Reason:** {state.assembled_context.missing_context}\n\n"
            f"**Query:** {state.query}\n\n"
            f"The knowledge base does not contain information about this topic. "
            f"Please verify the query is within KIRA's scope — Amazon agency "
            f"performance reporting, ACOS management, compliance, and client "
            f"onboarding context."
        )
        state.mark_complete("report_agent")
        print(f"  [Report] Uncertainty response generated.")
        return state

    if not state.analysis_narrative:
        state.add_error("report_agent", "No analysis narrative available.")
        return state

    try:
        anomaly_text = (
            "\n".join(f"- {a.metric}: {a.description}" for a in state.anomalies)
            if state.anomalies else "None detected."
        )

        metrics_text = "\n".join(
            f"- {metric}: latest={stats['latest']}, mean={stats['mean']}"
            for metric, stats in state.metrics_summary.items()
            if isinstance(stats, dict)
        )

        llm = get_llm(agent_name="report_agent")
        response = llm.chat(
            messages=[
                {"role": "system", "content": REPORT_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Client: {state.client_id}\n"
                        f"Report query: {state.query}\n\n"
                        f"Analysis:\n{state.analysis_narrative}\n\n"
                        f"Metrics (90-day):\n{metrics_text}\n\n"
                        f"Anomalies:\n{anomaly_text}\n\n"
                        f"Supporting context:\n{state.assembled_context.context_window if state.assembled_context else state.context_block}"
                    ),
                },
            ],
            temperature=0.1,
        )

        state.report_draft = response.content
        state.mark_complete("report_agent")
        print(f"  [Report] Done. {len(state.report_draft)} chars generated.")

    except Exception as e:
        state.add_error("report_agent", str(e))
        print(f"  [Report] Error: {e}")

    return state