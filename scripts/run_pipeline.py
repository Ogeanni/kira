"""
scripts/run_pipeline.py

Runs the full KIRA agent pipeline for a client.

Agent order:
  Research → Analysis → Report → Compliance → Memory save

Memory is saved at the end of every run so future runs
have context about past anomalies, compliance issues, and patterns.

Run:
    python scripts/run_pipeline.py --client natura
    python scripts/run_pipeline.py --client lumina
"""

import argparse

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents import PipelineState
from agents.research_agent import run as research
from agents.analysis_agent import run as analysis
from agents.report_agent import run as report
from agents.compliance_agent import run as compliance
from intelligence.memory import save_pipeline_memories, MemoryStore
from intelligence.llm_client import reset_global_tracker, get_global_tracker
from config.settings import get_settings

settings = get_settings()


def run_pipeline(client_id: str, query: str) -> PipelineState:
    reset_global_tracker()   # fresh cost tracker per run

    print(f"\n{'─' * 50}")
    print(f"KIRA Pipeline — {client_id}")
    print(f"Query: {query}")
    print(f"{'─' * 50}")

    # Show relevant past memories before running
    if settings.mem0_enabled:
        store = MemoryStore()
        past = store.search(client_id=client_id, query=query, top_k=2)
        if past:
            print(f"\n── Past memory ({len(past)} relevant) ──")
            for m in past:
                print(f"  [{m['score']}] {m['memory'][:80]}...")

    state = PipelineState(client_id=client_id, query=query)
    state = research(state)
    state = analysis(state)
    state = report(state)
    state = compliance(state)

    # Save findings to memory for future runs
    if settings.mem0_enabled:
        compliance_issues = (
            state.compliance.issues if state.compliance else []
        )
        save_pipeline_memories(
            client_id=client_id,
            run_id=state.run_id,
            anomalies=state.anomalies,
            compliance_issues=compliance_issues,
            analysis_narrative=state.analysis_narrative,
        )
        print(f"\n  [Memory] Saved {len(state.anomalies)} anomalies + analysis to memory.")

    return state


def print_results(state: PipelineState) -> None:
    print(f"\n{'─' * 50}")
    print(f"Pipeline complete — run_id: {state.run_id}")
    print(f"Agents completed : {', '.join(state.completed_agents)}")

    if state.errors:
        print(f"\nErrors:")
        for e in state.errors:
            print(f"  {e}")

    if state.anomalies:
        print(f"\nAnomalies detected ({len(state.anomalies)}):")
        for a in state.anomalies:
            print(f"  {a.description}")

    if state.compliance:
        status = "PASSED" if state.compliance.passed else "FAILED"
        print(f"\nCompliance: {status} "
              f"(confidence: {state.compliance.confidence:.2f})")
        if state.compliance.issues:
            for issue in state.compliance.issues:
                print(f"  - {issue}")

    # Print cost summary
    get_global_tracker().print_summary()

    if state.final_report:
        print(f"\n{'─' * 50}")
        print("FINAL REPORT")
        print(f"{'─' * 50}")
        print(state.final_report)
    else:
        print("\nNo final report — compliance failed or pipeline errored.")
        if state.report_draft:
            print("\nDraft (not approved):")
            print(state.report_draft[:500] + "...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--client",
        default="natura",
        choices=["natura", "vitalblend", "peakgear", "lumina"],
    )
    parser.add_argument(
        "--query",
        default="Generate a weekly performance report with anomaly analysis",
    )
    args = parser.parse_args()

    state = run_pipeline(client_id=args.client, query=args.query)
    print_results(state)