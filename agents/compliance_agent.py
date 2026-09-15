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
You are a compliance checker for Amazon marketplace listings and agency reports.
Review the provided report against the compliance rules and context.

Your job is to find violations — places where the report RECOMMENDS or USES
prohibited terms in a way that would appear in client-facing content.

A recommendation to AVOID a term, REMOVE a term, or ENSURE TERMS ARE NOT USED
is never a violation — it is correct compliance advice.
Only flag recommendations to USE, ADD, INCLUDE, or APPLY prohibited terms.

IMPORTANT DISTINCTION:
- A report that WARNS about a prohibited term is NOT a violation.
  Example: "Avoid using anti-aging in listing copy" — this is correct advice.
- A report that RECOMMENDS using a prohibited term IS a violation.
  Example: "Add anti-aging to your listing title" — this is a violation.
- A report that MENTIONS a prohibited term in a compliance context is NOT a violation.
  Example: "Prohibited terms such as chemical-free must be avoided" — correct.

Check for genuine violations only:
1. Prohibited terms recommended for use in listing copy, A+ content, or ads
   (cure, treat, prevent, diagnose, heal, therapeutic, clinically proven,
   best, #1, greatest, finest, top-rated, guaranteed, chemical-free,
   anti-aging when recommended for use — not when warned against)
2. Unsubstantiated medical or health claims being recommended
3. Competitor brand names recommended for use in copy
4. Pricing language recommended for listing copy (sale, discount, % off)
5. Actions requiring account manager approval that are recommended without
   flagging the approval requirement:
   - Pausing an entire campaign
   - Changing campaign structure across the board
   - Increasing total budget by more than 50% in a single change

When you find NO violations:
  - passed must be true
  - issues must be an empty list []
  - Do not explain correct compliance advice in the issues list
  - Do not list things the report is doing correctly

When you find genuine violations:
  - passed must be false
  - issues must list each violation specifically

Respond ONLY with valid JSON. No explanation outside the JSON.
{
  "passed": true or false,
  "confidence": 0.0 to 1.0,
  "issues": []
}

Example of a PASSING report response:
{"passed": true, "confidence": 0.95, "issues": []}

Example of a FAILING report response:
{"passed": false, "confidence": 0.9, "issues": ["Recommendation to use 'guaranteed' in listing copy"]}
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
    # Strip markdown code fences if present
    # gpt-4o sometimes wraps JSON in ```json ... ``` blocks
    if raw_response.startswith("```"):
        lines = raw_response.split("\n")
        raw_response = "\n".join(lines[1:-1])

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