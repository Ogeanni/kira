"""
intelligence/mcp_server.py

MCP server exposing three tools to KIRA agents:
  retrieve_context — semantic search over knowledge store
  query_metrics    — structured performance data from CSV warehouse
  read_memory      — episodic memory reads per client

Agents never import retriever.py or pandas directly.
They call tools through this server.

Run standalone to test tools directly:
    python intelligence/mcp_server.py

Agents connect via stdio transport.
"""

import json
import numpy as np
import pandas as pd
from mcp.server.fastmcp import FastMCP

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import get_settings
from knowledge.retriever import Retriever, RetrievalQuery

settings = get_settings()

mcp = FastMCP("kira-tools")
retriever = Retriever()


class NumpyEncoder(json.JSONEncoder):
    """
    Converts numpy int64/float64/bool to native Python types.
    Pandas reads CSV columns as numpy types — json.dumps fails
    without this encoder.
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


@mcp.tool()
def retrieve_context(
    question: str,
    client_id: str = "global",
    doc_types: str = "",
    top_k: int = 5,
) -> str:
    """
    Retrieves relevant context from the knowledge store.

    Searches both global documents (SOPs, compliance, frameworks)
    and client-specific documents (transcripts, brand guides).
    Returns top_k most relevant chunks as a JSON string.

    Args:
        question:  The question or topic to retrieve context for.
        client_id: Client namespace to search. Use 'global' for
                   agency-wide documents only.
        doc_types: Comma-separated filter e.g. 'sop,compliance'.
                   Empty string searches all document types.
        top_k:     Number of chunks to return (default 5).
    """
    doc_type_list = [d.strip() for d in doc_types.split(",") if d.strip()]

    if client_id == "global":
        query = RetrievalQuery(
            question=question,
            namespace="global",
            doc_types=doc_type_list,
            top_k=top_k,
        )
        response = retriever.retrieve(query)
    else:
        response = retriever.retrieve_for_client(
            question=question,
            client_id=client_id,
            doc_types=doc_type_list if doc_type_list else None,
            top_k=top_k,
        )

    return json.dumps({
        "question": question,
        "client_id": client_id,
        "total_found": response.total_found,
        "results": [
            {
                "chunk_id": r.chunk_id,
                "context": r.context,
                "score": r.score,
                "source_file": r.metadata.get("source_file", ""),
                "doc_type": r.metadata.get("doc_type", ""),
            }
            for r in response.results
        ],
    })


@mcp.tool()
def query_metrics(
    client_id: str,
    days: int = 7,
    metric: str = "all",
) -> str:
    """
    Returns performance metrics for a client from the data warehouse.

    Reads from data/metrics/daily_metrics.csv.

    Args:
        client_id: Client to query e.g. 'natura', 'vitalblend'.
        days:      Number of recent days to return (default 7).
        metric:    Specific metric or 'all'. Options: revenue_usd,
                   acos, buy_box_pct, sessions, conversion_rate.
    """
    metrics_path = settings.metrics_dir / "daily_metrics.csv"

    if not metrics_path.exists():
        return json.dumps({"error": "Metrics file not found. Run scripts/generate_data.py"})

    df = pd.read_csv(metrics_path)
    client_df = df[df["client_id"] == client_id].copy()

    if client_df.empty:
        return json.dumps({"error": f"No data found for client_id: {client_id}"})

    client_df = client_df.sort_values("date", ascending=False).head(days)

    base_cols = ["date", "client_id", "asin", "is_anomaly", "anomaly_type"]
    metric_cols = {
        "revenue_usd": ["revenue_usd", "ad_revenue_usd", "organic_revenue_usd"],
        "acos": ["acos", "tacos"],
        "buy_box_pct": ["buy_box_pct"],
        "sessions": ["sessions", "conversion_rate"],
        "conversion_rate": ["conversion_rate", "sessions"],
        "all": [
            "revenue_usd", "ad_revenue_usd", "organic_revenue_usd",
            "units_sold", "sessions", "conversion_rate",
            "acos", "tacos", "buy_box_pct", "asp_usd",
        ],
    }

    selected = metric_cols.get(metric, metric_cols["all"])
    cols = base_cols + [c for c in selected if c in client_df.columns]
    result_df = client_df[cols].sort_values("date")

    numeric_cols = result_df.select_dtypes(include="number").columns.tolist()
    summary = {}
    for col in numeric_cols:
        if col == "is_anomaly":
            continue
        summary[col] = {
            "mean": round(float(result_df[col].mean()), 4),
            "min": round(float(result_df[col].min()), 4),
            "max": round(float(result_df[col].max()), 4),
            "latest": round(float(result_df[col].iloc[-1]), 4),
        }

    anomalies = result_df[result_df["is_anomaly"] == True]

    return json.dumps({
        "client_id": client_id,
        "days_returned": len(result_df),
        "date_range": {
            "from": result_df["date"].iloc[0],
            "to": result_df["date"].iloc[-1],
        },
        "summary": summary,
        "anomalies_detected": int(len(anomalies)),
        "anomaly_types": anomalies["anomaly_type"].dropna().unique().tolist(),
        "rows": result_df.to_dict(orient="records"),
    }, cls=NumpyEncoder)


@mcp.tool()
def read_memory(
    client_id: str,
    query: str = "",
    top_k: int = 3,
) -> str:
    """
    Reads episodic memory for a client.

    Returns what KIRA has learned about this client across
    previous pipeline runs — past anomalies, decisions made,
    client preferences surfaced during analysis.

    Args:
        client_id: Client to read memory for.
        query:     Optional topic to focus memory retrieval.
        top_k:     Number of memory entries to return.
    """
    if not settings.mem0_enabled:
        return json.dumps({
            "client_id": client_id,
            "memories": [],
            "note": "Memory disabled in settings.",
        })

    try:
        from intelligence.memory import MemoryStore
        store = MemoryStore()
        memories = store.search(client_id=client_id, query=query, top_k=top_k)
        return json.dumps({
            "client_id": client_id,
            "memories": memories,
        })
    except ImportError:
        return json.dumps({
            "client_id": client_id,
            "memories": [],
            "note": "Memory module not yet built (Layer 5). No past context available.",
        })


if __name__ == "__main__":
    print("\nTesting MCP tools directly (without protocol overhead)...\n")

    print("── retrieve_context ──")
    result = retrieve_context(
        question="What are the ACOS thresholds by account stage?",
        client_id="global",
        doc_types="sop",
        top_k=2,
    )
    data = json.loads(result)
    print(f"  Found: {data['total_found']} results")
    for r in data["results"]:
        print(f"  [{r['score']}] {r['source_file']} — {r['context'][:60]}...")

    print("\n── query_metrics ──")
    result = query_metrics(client_id="natura", days=7, metric="acos")
    data = json.loads(result)
    if "error" not in data:
        print(f"  Client      : {data['client_id']}")
        print(f"  Days        : {data['days_returned']}")
        print(f"  ACOS mean   : {data['summary'].get('acos', {}).get('mean', 'n/a')}")
        print(f"  ACOS latest : {data['summary'].get('acos', {}).get('latest', 'n/a')}")
        print(f"  Anomalies   : {data['anomalies_detected']}")
    else:
        print(f"  Error: {data['error']}")

    print("\n── read_memory ──")
    result = read_memory(client_id="natura", query="ACOS performance")
    data = json.loads(result)
    print(f"  Memories: {len(data['memories'])}")
    if data.get("note"):
        print(f"  Note: {data['note']}")

    print("\nAll tools tested. MCP server ready.")