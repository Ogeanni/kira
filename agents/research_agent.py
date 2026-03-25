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


def run(state: PipelineState) -> PipelineState:
    """
    Retrieves context relevant to the query and client,
    then generates a brief summary of what was found.

    Calls retrieve_context MCP tool — the agent never touches
    the vector store directly.
    """
    print(f"  [Research] Retrieving context for '{state.query[:50]}...'")

    try:
        import json

        # Call MCP tool
        raw = retrieve_context(
            question=state.query,
            client_id=state.client_id,
            doc_types="",       # search all doc types
            top_k=settings.retrieval_top_k,
        )
        data = json.loads(raw)

        # Convert to RetrievedContext objects
        contexts = [
            RetrievedContext(
                context=r["context"],
                source_file=r["source_file"],
                doc_type=r["doc_type"],
                score=r["score"],
            )
            for r in data.get("results", [])
        ]
        state.retrieved_contexts = contexts

        if not contexts:
            state.add_error("research_agent", "No relevant context found.")
            return state

        # Summarise what was retrieved
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

        state.mark_complete("research_agent")
        print(f"  [Research] Done. {len(contexts)} chunks from {len(source_summary)} sources.")

    except Exception as e:
        state.add_error("research_agent", str(e))
        print(f"  [Research] Error: {e}")

    return state