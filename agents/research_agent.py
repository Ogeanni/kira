"""
agents/research_agent.py

First agent in the pipeline.
Job: retrieve relevant context from the knowledge store and
     summarise what was found.

Input:  PipelineState with client_id and query
Output: PipelineState with retrieved_contexts and research_summary
"""

from openai import OpenAI

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents import PipelineState, RetrievedContext
from intelligence.mcp_server import retrieve_context
from config.settings import get_settings

settings = get_settings()
client = OpenAI(api_key=settings.openai_api_key)


def _get_client_operational_context(client_id: str) -> str:
    """
    Retrieves client-specific operational data from PostgreSQL.

    Returns a formatted string containing:
    - ACOS targets by account stage
    - Product catalogue with current status and notes

    This data closes the grounding gap in recommendations — without it
    the LLM generates generic advice. With it, it can reference specific
    thresholds and known product issues.

    Returns empty string if database is unavailable or client not found.
    """
    if client_id == "global":
        return ""

    try:
        from infra.database import get_session
        from infra.database import AcosTarget, ProductCatalogue

        session = get_session()
        lines = []

        # ACOS targets by stage
        targets = (
            session.query(AcosTarget)
            .filter(AcosTarget.client_id == client_id)
            .all()
        )
        if targets:
            lines.append("ACOS TARGETS BY ACCOUNT STAGE")
            for t in targets:
                line = (
                    f"  {t.account_stage} ({t.days_range}): "
                    f"target {t.acos_target_min*100:.0f}%-{t.acos_target_max*100:.0f}%, "
                    f"break-even {t.break_even_acos*100:.0f}%, "
                    f"objective: {t.primary_objective}"
                )
                if t.notes:
                    line += f" — {t.notes}"
                lines.append(line)

        # Product catalogue — status and notes
        products = (
            session.query(ProductCatalogue)
            .filter(ProductCatalogue.client_id == client_id)
            .all()
        )
        if products:
            lines.append("")
            lines.append("PRODUCT CATALOGUE")
            for p in products:
                line = (
                    f"  ASIN {p.asin}: {p.product_title[:60]} | "
                    f"Buy Box: {p.buy_box_status} ({p.buy_box_pct}%) | "
                    f"ACOS target: {p.acos_target}%"
                )
                if p.notes:
                    line += f" | Note: {p.notes}"
                lines.append(line)

        session.close()
        return "\n".join(lines)

    except Exception as e:
        # Database unavailable — degrade gracefully
        # Research Agent logs this but does not fail the pipeline
        print(f"  [Research] DB context unavailable: {e}")
        return ""


def run(state: PipelineState) -> PipelineState:
    """
    Retrieves context relevant to the query and client,
    then assembles a structured context window for generation.

    Two retrieval calls:
    1. Vector store — SOPs, compliance, frameworks, transcripts
    2. Database — client ACOS targets, product catalogue

    Both are passed to the context assembler which structures them
    into a context window with source separation, confidence signalling,
    and uncertainty notices.
    """
    print(f"  [Research] Retrieving context for '{state.query[:50]}...'")

    try:
        import json

        # ── 1. Vector store retrieval ─────────────────────────────
        raw = retrieve_context(
            question=state.query,
            client_id=state.client_id,
            doc_types="",
            top_k=settings.retrieval_top_k,
        )
        data = json.loads(raw)

        contexts = [
            RetrievedContext(
                context=r["context"],
                source_file=r["source_file"],
                doc_type=r["doc_type"],
                score=r["score"],
                window_context=r.get("window_context", ""),
                section_title=r.get("section_title", ""),
                document_title=r.get("document_title", ""),
            )
            for r in data.get("results", [])
        ]
        state.retrieved_contexts = contexts

        if not contexts:
            state.add_error("research_agent", "No relevant context found.")
            return state

        # ── 2. Database context retrieval ─────────────────────────
        operational_context = _get_client_operational_context(state.client_id)
        if operational_context:
            print(f"  [Research] Operational context retrieved for {state.client_id}")

        # ── 3. Source summary ─────────────────────────────────────
        source_summary = {}
        for c in contexts:
            doc = c.source_file
            if doc not in source_summary:
                source_summary[doc] = []
            source_summary[doc].append(c.doc_type)

        sources_str = ", ".join(
            f"{doc} ({types[0]})" for doc, types in source_summary.items()
        )
        state.research_summary = (
            f"Retrieved {len(contexts)} context chunks from: {sources_str}."
        )
        if operational_context:
            state.research_summary += " Operational context from database included."

        state.mark_complete("research_agent")

        # ── 4. Context assembly ───────────────────────────────────
        from agents.context_assembler import assemble
        state.assembled_context = assemble(
            retrieved_contexts=contexts,
            query=state.query,
            client_id=state.client_id,
            operational_context=operational_context,
        )
        print(
            f"  [Research] Context assembled. "
            f"Confidence: {state.assembled_context.confidence}"
        )
        print(
            f"  [Research] Done. {len(contexts)} chunks from "
            f"{len(source_summary)} sources."
        )

    except Exception as e:
        state.add_error("research_agent", str(e))
        print(f"  [Research] Error: {e}")

    return state