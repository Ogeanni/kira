"""
intelligence/faithfulness.py

Lightweight faithfulness checker for generated reports.

Faithfulness measures whether claims in the generated report
are supported by the retrieved context. A report that scores
low on faithfulness is generating from parametric knowledge
rather than retrieved evidence — hallucination risk.

This is not a full Ragas evaluation. It is a single LLM-as-judge
call designed to run after every pipeline run as an automated
quality gate. It costs ~$0.001 per check and adds ~3s latency.

Scoring:
    1.0 — all claims supported by context
    0.8 — most claims supported, minor gaps
    0.6 — some claims unsupported
    below 0.6 — significant hallucination risk, flag for review

The check is non-blocking — a low score logs a warning but
does not prevent the report from being delivered. A human
reviewer should assess flagged reports.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from intelligence.llm_client import get_llm

FAITHFULNESS_THRESHOLD = 0.6

FAITHFULNESS_PROMPT = """
You are a faithfulness evaluator for AI-generated reports.

Your job: determine whether the claims in the REPORT are supported
by the CONTEXT provided to the system that generated it.

A claim is supported if the specific fact, number, threshold, or
recommendation text appears in the context — regardless of whether
applying it is currently appropriate given the metrics.

A claim is NOT supported if:
- It introduces facts, numbers, or recommendations with no basis in the context
- It references events, trends, or data not present in the context
- It uses general Amazon knowledge not retrieved from the knowledge base

IMPORTANT: Do not reason about whether an action is appropriate given
current metrics. Only check whether the claim text is grounded in the
context. "Reduce bids on keywords with ACOS above 60%" is supported if
that text appears in the context — even if current ACOS is 38%.

CONTEXT:
{context}

REPORT:
{report}

Respond ONLY with valid JSON:
{{
    "faithfulness_score": 0.0 to 1.0,
    "supported_claims": ["claim 1", "claim 2"],
    "unsupported_claims": ["claim 1", "claim 2"],
    "assessment": "one sentence summary"
}}

Be precise. Only list specific claims that are clearly unsupported.
If all claims are supported, unsupported_claims must be an empty list
and faithfulness_score must be 1.0.
""".strip()


def check_faithfulness(
    report: str,
    context_window: str,
    client_id: str = "unknown",
    run_id: str = "unknown",
) -> dict:
    """
    Checks whether the report is faithful to the retrieved context.

    Args:
        report:         The generated report text
        context_window: The assembled context passed to the Report Agent
        client_id:      For logging
        run_id:         For logging

    Returns:
        dict with faithfulness_score, supported_claims,
        unsupported_claims, assessment, and flagged (bool)
    """
    if not report or not context_window:
        return {
            "faithfulness_score": 0.0,
            "supported_claims": [],
            "unsupported_claims": ["Report or context is empty"],
            "assessment": "Cannot evaluate — missing report or context",
            "flagged": True,
            "error": "Missing input",
        }

    # Truncate context to avoid token limits
    # The faithfulness check needs the key facts, not the full window
    context_truncated = context_window[:3000]
    report_truncated = report[:2000]

    try:
        llm = get_llm(agent_name="faithfulness_checker")
        response = llm.chat(
            messages=[
                {
                    "role": "user",
                    "content": FAITHFULNESS_PROMPT.format(
                        context=context_truncated,
                        report=report_truncated,
                    ),
                }
            ],
            temperature=0,
        )

        raw = response.content.strip()
        result = json.loads(raw)

        score = float(result.get("faithfulness_score", 0.0))
        unsupported = result.get("unsupported_claims", [])
        supported = result.get("supported_claims", [])

        # If LLM returns 0.0 but only flagged 1-2 minor claims,
        # recalculate proportionally based on claim counts
        if score == 0.0 and supported and unsupported:
            total = len(supported) + len(unsupported)
            score = round(len(supported) / total, 3)

        flagged = score < FAITHFULNESS_THRESHOLD or len(unsupported) > 2

        return {
            "faithfulness_score": round(score, 3),
            "supported_claims": result.get("supported_claims", []),
            "unsupported_claims": unsupported,
            "assessment": result.get("assessment", ""),
            "flagged": flagged,
            "error": None,
        }

    except json.JSONDecodeError as e:
        return {
            "faithfulness_score": 0.0,
            "supported_claims": [],
            "unsupported_claims": [],
            "assessment": "Faithfulness check failed to parse LLM response",
            "flagged": True,
            "error": str(e),
        }
    except Exception as e:
        return {
            "faithfulness_score": 0.0,
            "supported_claims": [],
            "unsupported_claims": [],
            "assessment": f"Faithfulness check error: {e}",
            "flagged": True,
            "error": str(e),
        }


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    # Test with a clean report
    test_context = """
RETRIEVAL CONFIDENCE: HIGH
== AUTHORITATIVE SOURCES ==
[SOURCE 1] ACOS Management SOP
If ACOS exceeds target by more than 10% for 3 consecutive days:
Reduce bids on keywords with ACOS above 60% by 15%.

== CLIENT OPERATIONAL DATA ==
ACOS TARGETS BY ACCOUNT STAGE
  Growth (90-365): target 28%-35%, break-even 38%
"""

    test_report_clean = """
## Executive Summary
ACOS is currently at 38%, which is above the growth phase target of 28-35%.

## Recommended Actions
1. Reduce bids on keywords with ACOS above 60% by 15% per the ACOS management SOP.
"""

    test_report_hallucinated = """
## Executive Summary
ACOS is at 38%, above target. Competitor activity has increased by 23% this quarter.

## Recommended Actions
1. Launch a new Sponsored Display campaign targeting similar ASINs.
2. Reduce price by 10% to improve buy box percentage.
3. File a case with Amazon Seller Support for the suppression issue.
"""

    print("=== Test 1 — Clean report ===")
    result = check_faithfulness(test_report_clean, test_context)
    print(f"Score    : {result['faithfulness_score']}")
    print(f"Flagged  : {result['flagged']}")
    print(f"Assessment: {result['assessment']}")
    if result['unsupported_claims']:
        print(f"Unsupported: {result['unsupported_claims']}")

    print("\n=== Test 2 — Hallucinated report ===")
    result2 = check_faithfulness(test_report_hallucinated, test_context)
    print(f"Score    : {result2['faithfulness_score']}")
    print(f"Flagged  : {result2['flagged']}")
    print(f"Assessment: {result2['assessment']}")
    if result2['unsupported_claims']:
        print("Unsupported:")
        for c in result2['unsupported_claims']:
            print(f"  - {c}")
