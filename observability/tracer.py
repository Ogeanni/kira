"""
observability/tracer.py

LangSmith tracing wrapper for KIRA.

Wraps agent runs with trace context so every pipeline run
is recorded in LangSmith with:
  - Inputs and outputs per agent
  - Token cost per agent
  - Latency per agent hop
  - Errors and retries

Degrades gracefully when LANGSMITH_ENABLED=false —
agents run normally, nothing is traced.

Setup:
  1. Sign up at smith.langchain.com
  2. Add to .env:
       LANGSMITH_ENABLED=true
       LANGSMITH_API_KEY=your_key
       LANGSMITH_PROJECT=kira
"""

import functools
import time
from typing import Callable, Any
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import get_settings

settings = get_settings()


def _langsmith_available() -> bool:
    """Check if LangSmith is configured and enabled."""
    return (
        settings.langsmith_enabled
        and bool(settings.langsmith_api_key)
    )


def trace_agent(agent_name: str) -> Callable:
    """
    Decorator that wraps an agent run function with LangSmith tracing.

    Usage:
        @trace_agent("research_agent")
        def run(state: PipelineState) -> PipelineState:
            ...

    When LANGSMITH_ENABLED=false, the decorator is a no-op —
    the function runs exactly as if the decorator wasn't there.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            if not _langsmith_available():
                return func(*args, **kwargs)

            try:
                from langsmith import traceable
                traced_func = traceable(
                    name=agent_name,
                    run_type="chain",
                    project_name=settings.langsmith_project,
                )(func)
                return traced_func(*args, **kwargs)
            except Exception:
                # Never let tracing break the pipeline
                return func(*args, **kwargs)

        return wrapper
    return decorator


class PipelineTracer:
    """
    Context manager for tracing a full pipeline run.

    Records the entire run as a parent trace in LangSmith,
    with each agent as a child span.

    Usage:
        with PipelineTracer(client_id="natura", run_id="20260325") as tracer:
            state = research(state)
            tracer.log_step("research", tokens=150, cost=0.001)
            state = analysis(state)
            tracer.log_step("analysis", tokens=300, cost=0.002)
    """

    def __init__(self, client_id: str, run_id: str):
        self.client_id = client_id
        self.run_id = run_id
        self.start_time = None
        self.steps: list[dict] = []
        self.enabled = _langsmith_available()

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = round(time.time() - self.start_time, 2)
        if self.enabled:
            self._flush(elapsed, error=str(exc_val) if exc_val else None)
        return False   # don't suppress exceptions

    def log_step(
        self,
        step_name: str,
        tokens: int = 0,
        cost_usd: float = 0.0,
        latency: float = 0.0,
        metadata: dict | None = None,
    ) -> None:
        """Records a single agent step."""
        self.steps.append({
            "step": step_name,
            "tokens": tokens,
            "cost_usd": cost_usd,
            "latency": latency,
            "metadata": metadata or {},
        })

    def _flush(self, total_elapsed: float, error: str | None = None) -> None:
        """Sends the accumulated trace to LangSmith."""
        try:
            from langsmith import Client
            ls_client = Client(api_key=settings.langsmith_api_key)

            total_tokens = sum(s["tokens"] for s in self.steps)
            total_cost = sum(s["cost_usd"] for s in self.steps)

            ls_client.create_run(
                name=f"kira_pipeline_{self.client_id}",
                run_type="chain",
                project_name=settings.langsmith_project,
                inputs={
                    "client_id": self.client_id,
                    "run_id": self.run_id,
                },
                outputs={
                    "steps": self.steps,
                    "total_tokens": total_tokens,
                    "total_cost_usd": round(total_cost, 6),
                    "total_latency_seconds": total_elapsed,
                },
                error=error,
            )
        except Exception:
            pass   # tracing failure never breaks the pipeline


def log_retrieval_quality(
    strategy: str,
    faithfulness: float,
    context_precision: float,
    run_id: str = "",
) -> None:
    """
    Logs retrieval quality metrics to LangSmith.
    Called after eval_retrieval.py runs so scores are visible
    alongside pipeline traces.
    """
    if not _langsmith_available():
        return

    try:
        from langsmith import Client
        ls_client = Client(api_key=settings.langsmith_api_key)
        ls_client.create_run(
            name="retrieval_eval",
            run_type="eval",
            project_name=settings.langsmith_project,
            inputs={"strategy": strategy, "run_id": run_id},
            outputs={
                "faithfulness": faithfulness,
                "context_precision": context_precision,
                "faithfulness_threshold": settings.ragas_faithfulness_threshold,
                "context_precision_threshold": settings.ragas_context_precision_threshold,
                "passed": (
                    faithfulness >= settings.ragas_faithfulness_threshold
                    and context_precision >= settings.ragas_context_precision_threshold
                ),
            },
        )
    except Exception:
        pass


if __name__ == "__main__":
    print(f"LangSmith enabled : {settings.langsmith_enabled}")
    print(f"LangSmith available: {_langsmith_available()}")

    if not _langsmith_available():
        print("\nTo enable tracing:")
        print("  1. Sign up at smith.langchain.com")
        print("  2. Add to .env:")
        print("       LANGSMITH_ENABLED=true")
        print("       LANGSMITH_API_KEY=your_key")
        print("       LANGSMITH_PROJECT=kira")
    else:
        print(f"Project: {settings.langsmith_project}")
        print("Tracing is active.")