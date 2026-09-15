"""
pipeline_runner.py

Production entry point for KIRA.

Orchestrates the four-agent pipeline:
    Research → Analysis → Report → Compliance

Saves memories at the end of each run so KIRA compounds
knowledge across pipeline runs per client.

Usage:
    python pipeline_runner.py --client natura --query "Weekly performance report"
    python pipeline_runner.py --client vitalblend --query "ACOS anomaly investigation"
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
import time

from dotenv import load_dotenv
load_dotenv()

import sys
ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents import PipelineState
from agents.research_agent import run as research
from agents.analysis_agent import run as analysis
from agents.report_agent import run as report
from agents.compliance_agent import run as compliance
from config.settings import get_settings

settings = get_settings()

from intelligence.llm_client import reset_global_tracker, get_global_tracker
reset_global_tracker()

def run_pipeline(
    client_id: str,
    query: str,
    save_memory: bool = True,
    verbose: bool = True,
) -> PipelineState:
    """
    Runs the full KIRA pipeline for a client query.

    Args:
        client_id:    Client to run for (natura, vitalblend, peakgear, lumina)
        query:        The query or report trigger
        save_memory:  Whether to persist findings to memory (default True)
        verbose:      Whether to print progress (default True)

    Returns:
        Completed PipelineState with all agent outputs
    """
    started_at = datetime.now()

    if verbose:
        print(f"\n{'='*60}")
        print(f"KIRA Pipeline")
        print(f"Client : {client_id}")
        print(f"Query  : {query[:60]}")
        print(f"Started: {started_at.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}\n")

    state = PipelineState(
        client_id=client_id,
        query=query,
    )

    # ── Agent pipeline ────────────────────────────────────────────
    agent_times = {}

    t = time.time()
    state = research(state)
    agent_times['research'] = round(time.time() - t, 2)
    if not state.is_healthy:
        _print_errors(state, verbose)
        _log_run(state, elapsed=time.time() - started_at.timestamp(), agent_times=agent_times)
        return state

    t = time.time()
    state = analysis(state)
    agent_times['analysis'] = round(time.time() - t, 2)

    t = time.time()
    state = report(state)
    agent_times['report'] = round(time.time() - t, 2)

    if not state.report_draft:
        if verbose:
            print("\n  Pipeline stopped — no report generated.")
        _print_errors(state, verbose)
        _log_run(state, elapsed=time.time() - started_at.timestamp(), agent_times=agent_times)
        return state

    t = time.time()
    state = compliance(state)
    agent_times['compliance'] = round(time.time() - t, 2)

    # ── Log run to database ───────────────────────────────────────
    total_elapsed = (datetime.now() - started_at).total_seconds()
    _log_run(state, elapsed=total_elapsed, agent_times=agent_times)

    # ── Memory persistence ────────────────────────────────────────
    if save_memory and settings.mem0_enabled:
        _save_memories(state, verbose)

    # ── Summary ───────────────────────────────────────────────────
    elapsed = (datetime.now() - started_at).total_seconds()

    if verbose:
        print(f"\n{'='*60}")
        print(f"Pipeline complete in {elapsed:.1f}s")
        print(f"  research:   {agent_times.get('research', 0)}s")
        print(f"  analysis:   {agent_times.get('analysis', 0)}s")
        print(f"  report:     {agent_times.get('report', 0)}s")
        print(f"  compliance: {agent_times.get('compliance', 0)}s")
        print(f"Agents completed : {', '.join(state.completed_agents)}")
        print(f"Compliance       : {'PASSED' if state.compliance and state.compliance.passed else 'FAILED'}")
        print(f"Final report     : {'SET' if state.final_report else 'NOT SET'}")
        tracker = get_global_tracker()
        s = tracker.summary()
        if s['total_calls'] > 0:
            print(f"LLM calls        : {s['total_calls']}")
            print(f"Total tokens     : {s['total_input_tokens'] + s['total_output_tokens']:,}")
            print(f"Cost             : ${s['total_cost_usd']:.6f}")
            
        if state.errors:
            print(f"Errors           : {len(state.errors)}")
            for e in state.errors:
                print(f"  {e}")
        print(f"{'='*60}\n")

        if state.final_report:
            print("FINAL REPORT")
            print("─"*60)
            print(state.final_report)

    return state


def _save_memories(state: PipelineState, verbose: bool) -> None:
    """
    Persists key findings from this pipeline run to memory.
    Called only when all agents complete and mem0 is enabled.
    """
    try:
        from intelligence.memory import save_pipeline_memories
        save_pipeline_memories(
            client_id=state.client_id,
            run_id=state.run_id,
            anomalies=state.anomalies,
            compliance_issues=(
                state.compliance.issues
                if state.compliance and not state.compliance.passed
                else []
            ),
            analysis_narrative=state.analysis_narrative,
        )
        if verbose:
            print(f"\n  [Memory] Findings saved for {state.client_id}")
    except Exception as e:
        if verbose:
            print(f"\n  [Memory] Save failed (non-critical): {e}")


def _log_run(state: PipelineState, elapsed: float, agent_times: dict,) -> None:
    """
    Logs the pipeline run to the pipeline_runs table.
    Called after every run regardless of outcome.
    Non-blocking — failures are logged but don't raise.
    """
    try:
        tracker = get_global_tracker()
        summary = tracker.summary()

        from infra.database import get_session, save_pipeline_run
        session = get_session()
        save_pipeline_run(session, {
            "run_id":               state.run_id,
            "client_id":            state.client_id,
            "query":                state.query,
            "completed_agents":     state.completed_agents,
            "anomalies_detected":   len(state.anomalies),
            "compliance_passed":    state.compliance.passed if state.compliance else None,
            "compliance_confidence":state.compliance.confidence if state.compliance else None,
            "llm_calls":            summary.get("total_calls", 0),
            "total_tokens":         summary.get("total_input_tokens", 0) + summary.get("total_output_tokens", 0),
            "cost_usd":             summary.get("total_cost_usd", 0.0),
            "latency_seconds":      round(elapsed, 2),
            "final_report":         state.final_report,
            "errors":               state.errors,
            "s3_key":               "",
        })
        session.close()
    except Exception as e:
        print(f"  [Monitor] Failed to log run (non-critical): {e}")


def _print_errors(state: PipelineState, verbose: bool) -> None:
    if verbose and state.errors:
        print(f"\nErrors:")
        for e in state.errors:
            print(f"  {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KIRA Pipeline Runner")
    parser.add_argument("--client", required=True,
                        choices=["natura", "vitalblend", "peakgear", "lumina"],
                        help="Client to run pipeline for")
    parser.add_argument("--query", required=True,
                        help="Query or report trigger")
    parser.add_argument("--no-memory", action="store_true",
                        help="Skip memory persistence")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress progress output")
    args = parser.parse_args()

    state = run_pipeline(
        client_id=args.client,
        query=args.query,
        save_memory=not args.no_memory,
        verbose=not args.quiet,
    )

    sys.exit(0 if state.is_healthy else 1)
