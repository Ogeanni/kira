"""
agents/compliance_agent.py

Fourth and final agent in the pipeline.
Job: check the report draft against compliance rules.

Input:  PipelineState with report_draft
Output: PipelineState with compliance result and final_report

Two-stage check:
  Stage 1 — LoRA classifier (Layer 5 — stub until trained)
  Stage 2 — GPT-4o with compliance context from vector store
"""

import json

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents import PipelineState, ComplianceResult
from intelligence.mcp_server import retrieve_context
from intelligence.llm_client import get_llm
from config.settings import get_settings

settings = get_settings()

COMPLIANCE_SYSTEM_PROMPT = """
You are a compliance checker for Amazon marketplace listings and reports.
Review the provided report against the compliance rules.

Check for:
- Prohibited terms: best, #1, cure, treat, prevent, guaranteed, chemical-free
- Unsubstantiated claims (medical, safety, superlative)
- Competitor brand mentions
- Pricing language in copy (sale, discount, % off)
- Any client-specific restrictions mentioned in the context

Respond ONLY with valid JSON in this exact format:
{
  "passed": true or false,
  "confidence": 0.0 to 1.0,
  "issues": ["issue 1", "issue 2"]
}

If no issues found, issues should be an empty list and passed should be true.
""".strip()


def _check_with_gpt4o(state: PipelineState) -> ComplianceResult:
    raw = retrieve_context(
        question="compliance rules prohibited terms restricted language",
        client_id=state.client_id,
        doc_types="compliance",
        top_k=3,
    )
    compliance_context = json.loads(raw)
    context_text = "\n\n".join(
        r["context"] for r in compliance_context.get("results", [])
    )

    llm = get_llm(agent_name="compliance_agent")
    response = llm.chat(
        messages=[
            {"role": "system", "content": COMPLIANCE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Compliance rules and context:\n{context_text}\n\n"
                    f"Report to check:\n{state.report_draft}"
                ),
            },
        ],
        temperature=0,
    )

    raw_response = response.content.strip()
    try:
        result = json.loads(raw_response)
        return ComplianceResult(
            passed=result.get("passed", False),
            confidence=result.get("confidence", 0.0),
            issues=result.get("issues", []),
            checked_by="gpt-4o",
        )
    except json.JSONDecodeError:
        return ComplianceResult(
            passed=False,
            confidence=0.0,
            issues=["Compliance check failed to parse — manual review required."],
            checked_by="gpt-4o",
        )


def run(state: PipelineState) -> PipelineState:
    print(f"  [Compliance] Checking report for {state.client_id}...")

    if not state.is_healthy:
        print(f"  [Compliance] Skipping — pipeline has errors.")
        return state

    if not state.report_draft:
        state.add_error("compliance_agent", "No report draft to check.")
        return state

    try:
        if settings.has_lora:
            pass  # LoRA classifier — implemented in Layer 5

        print(f"  [Compliance] Running GPT-4o compliance check...")
        result = _check_with_gpt4o(state)
        state.compliance = result

        if result.passed:
            state.final_report = state.report_draft
            print(f"  [Compliance] Passed (confidence: {result.confidence:.2f})")
        else:
            issues_text = "\n".join(f"- {i}" for i in result.issues)
            print(f"  [Compliance] Failed. Issues:\n{issues_text}")
            state.final_report = ""
            state.add_error(
                "compliance_agent",
                f"Report failed compliance: {'; '.join(result.issues)}"
            )

        state.mark_complete("compliance_agent")

    except Exception as e:
        state.add_error("compliance_agent", str(e))
        print(f"  [Compliance] Error: {e}")

    return state