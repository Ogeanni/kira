"""
agents/__init__.py

Defines PipelineState — the Pydantic model that carries context
between agents.

Every agent receives a PipelineState, does its work, and returns
an updated PipelineState. Nothing else is shared between agents.

Why Pydantic?
  - Validation at every handoff boundary
  - If the Research Agent returns malformed data, the Analysis Agent
    refuses to run — the error surfaces immediately, not 3 agents later
  - Clear schema makes debugging trivial — you always know what's in
    the packet at any point
"""

from typing import Optional, Any
from pydantic import BaseModel, Field
from datetime import datetime
from pathlib import Path
import sys

import sys
ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from intelligence.llm_client import UsageTracker


class RetrievedContext(BaseModel):
    """A single piece of retrieved context with its source."""
    context: str
    source_file: str
    doc_type: str
    score: float
    window_context: str = ""
    section_title: str = ""
    document_title: str = ""


class AnomalyFlag(BaseModel):
    """A detected anomaly in the metrics data."""
    metric: str
    current_value: float
    mean_value: float
    deviation_pct: float
    description: str


class ComplianceResult(BaseModel):
    """Result of the compliance check."""
    passed: bool
    confidence: float
    issues: list[str] = Field(default_factory=list)
    checked_by: str = "gpt-4o"   # or "lora" when Layer 5 is built


class PipelineState(BaseModel):
    """
    The context packet passed between every agent.

    Think of it as a baton in a relay race — each agent picks it up,
    adds their contribution, and passes it to the next.

    Mandatory fields are set at pipeline start.
    Optional fields are populated as agents run.
    """

    # ── Set at pipeline start ─────────────────────────────────────
    client_id: str
    run_id: str = Field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))
    query: str                          # what the pipeline was triggered with
    started_at: str = Field(default_factory=lambda: datetime.now().isoformat())

    # ── Populated by Research Agent ───────────────────────────────
    retrieved_contexts: list[RetrievedContext] = Field(default_factory=list)
    research_summary: str = ""          # brief summary of what was retrieved
    assembled_context: Optional[Any] = None   # AssembledContext from context_assembler

    # ── Populated by Analysis Agent ───────────────────────────────
    metrics_summary: dict = Field(default_factory=dict)
    anomalies: list[AnomalyFlag] = Field(default_factory=list)
    analysis_narrative: str = ""        # plain English analysis

    # ── Populated by Report Agent ─────────────────────────────────
    report_draft: str = ""              # full generated report
    report_sections: dict = Field(default_factory=dict)

    # ── Populated by Compliance Agent ─────────────────────────────
    compliance: Optional[ComplianceResult] = None
    final_report: str = ""              # compliance-approved report

    # ── Pipeline metadata ─────────────────────────────────────────
    errors: list[str] = Field(default_factory=list)
    completed_agents: list[str] = Field(default_factory=list)

    usage_tracker: Optional[Any] = Field(default_factory=UsageTracker)

    faithfulness: Optional[dict] = None

    def mark_complete(self, agent_name: str) -> None:
        """Records that an agent finished successfully."""
        self.completed_agents.append(agent_name)

    def add_error(self, agent_name: str, error: str) -> None:
        """Records an agent error without crashing the pipeline."""
        self.errors.append(f"[{agent_name}] {error}")

    @property
    def context_block(self) -> str:
        """Returns all retrieved contexts as a single formatted string."""
        if not self.retrieved_contexts:
            return "No context retrieved."
        return "\n\n---\n\n".join(
            f"[{c.source_file} | {c.doc_type}]\n{c.context}"
            for c in self.retrieved_contexts
        )

    @property
    def is_healthy(self) -> bool:
        """True if no errors have been recorded."""
        return len(self.errors) == 0