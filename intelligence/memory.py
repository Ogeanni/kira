"""
intelligence/memory.py

Cross-session memory for KIRA agents using Mem0.

What memory does:
  Each time the pipeline runs for a client, key findings are stored.
  Next run, the agent retrieves relevant past context — anomalies
  detected, client preferences, decisions made, patterns observed.

  Without memory: every run starts from scratch.
  With memory: the system compounds knowledge over time.

Memory types stored:
  - Anomalies detected and their root causes
  - Compliance issues flagged per client
  - Client-specific patterns
  - Analysis summaries per run

Storage: Qdrant persisted to data/memory/ on disk.

Requires: mem0ai==1.0.x
API note: messages must be a list of dicts with role/content,
not a plain string — same format as OpenAI chat messages.

Run directly to test:
    python intelligence/memory.py
"""

from dotenv import load_dotenv
load_dotenv()

from mem0 import Memory

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
    
from config.settings import get_settings

settings = get_settings()

# Mem0 config — persists to disk between pipeline runs
MEMORY_CONFIG = {
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "collection_name": "kira_memory",
            "path": str(settings.data_dir / "memory"),
        },
    },
}


class MemoryStore:
    """
    Wraps Mem0 with KIRA-specific methods.

    Mem0 handles:
      - Embedding memories into a local Qdrant vector store
      - Deduplication — similar memories are merged
      - Semantic search over stored memories
      - Automatic summarisation of older memories

    API note (mem0ai 1.0.x):
      messages parameter must be a list of dicts:
        [{"role": "user", "content": "..."}]
      Plain strings are not supported in this version.
    """

    def __init__(self):
        self.memory = Memory.from_config(MEMORY_CONFIG)

    def add(
        self,
        client_id: str,
        content: str,
        agent_name: str = "pipeline",
        metadata: dict | None = None,
    ) -> str:
        """
        Stores a memory for a client.

        Args:
            client_id:  The client this memory belongs to.
            content:    The memory text to store.
            agent_name: Which agent created this memory.
            metadata:   Optional extra context (run_id, type, etc.)

        Returns:
            The memory ID assigned by Mem0.
        """
        meta = metadata or {}
        meta["client_id"] = client_id
        meta["agent"] = agent_name

        # mem0ai 1.0.x requires messages as list of dicts
        result = self.memory.add(
            messages=[{"role": "user", "content": content}],
            user_id=client_id,
            metadata=meta,
        )

        if isinstance(result, dict):
            results = result.get("results", [])
            if results:
                return results[0].get("id", "unknown")
        if isinstance(result, list) and result:
            return result[0].get("id", "unknown")
        return "unknown"

    def search(
        self,
        client_id: str,
        query: str,
        top_k: int = 3,
    ) -> list[dict]:
        """
        Searches memories for a client using semantic similarity.

        Returns list of dicts:
          {"id": "...", "memory": "...", "score": 0.87}

        Empty list if no memories exist — agents handle this gracefully.
        """
        try:
            results = self.memory.search(
                query=query,
                user_id=client_id,
                limit=top_k,
            )
            if isinstance(results, dict):
                results = results.get("results", [])
            return [
                {
                    "id": r.get("id", ""),
                    "memory": r.get("memory", ""),
                    "score": round(r.get("score", 0.0), 4),
                }
                for r in (results or [])
            ]
        except Exception:
            return []

    def get_all(self, client_id: str) -> list[dict]:
        """Returns all stored memories for a client."""
        try:
            results = self.memory.get_all(user_id=client_id)
            if isinstance(results, dict):
                results = results.get("results", [])
            return [
                {
                    "id": r.get("id", ""),
                    "memory": r.get("memory", ""),
                }
                for r in (results or [])
            ]
        except Exception:
            return []

    def delete_all(self, client_id: str) -> None:
        """Clears all memories for a client."""
        try:
            self.memory.delete_all(user_id=client_id)
        except Exception:
            pass


def save_pipeline_memories(
    client_id: str,
    run_id: str,
    anomalies: list,
    compliance_issues: list[str],
    analysis_narrative: str,
) -> None:
    """
    Called at the end of each pipeline run to persist key findings.
    This is what makes KIRA smarter over time.

    Stores three memory types:
      1. Anomalies — so future runs know the history
      2. Compliance issues — so the compliance agent learns patterns
      3. Analysis summary — high-level context for future retrieval
    """
    store = MemoryStore()

    if anomalies:
        anomaly_text = "; ".join(a.description for a in anomalies)
        store.add(
            client_id=client_id,
            content=f"Run {run_id}: Anomalies detected — {anomaly_text}",
            agent_name="analysis_agent",
            metadata={"run_id": run_id, "type": "anomaly"},
        )

    if compliance_issues:
        issues_text = "; ".join(compliance_issues)
        store.add(
            client_id=client_id,
            content=f"Run {run_id}: Compliance issues — {issues_text}",
            agent_name="compliance_agent",
            metadata={"run_id": run_id, "type": "compliance"},
        )

    if analysis_narrative:
        store.add(
            client_id=client_id,
            content=f"Run {run_id} summary: {analysis_narrative[:300]}",
            agent_name="analysis_agent",
            metadata={"run_id": run_id, "type": "summary"},
        )


if __name__ == "__main__":
    store = MemoryStore()

    # Clean slate for test
    store.delete_all("lumina_test")

    print("── Memory write test ──")
    mid = store.add(
        client_id="lumina_test",
        content="On 2024-01-21 Lumina experienced listing suppression. Sessions dropped 89% and revenue fell 84%. Root cause: off-white background on main image.",
        agent_name="analysis_agent",
        metadata={"type": "anomaly"},
    )
    print(f"  Stored memory ID: {mid}")

    mid2 = store.add(
        client_id="lumina_test",
        content="Lumina ACOS target is 30%. Current blended ACOS is 41%. Priority: reduce pendant light ACOS from 48% to under 30%.",
        agent_name="research_agent",
        metadata={"type": "client_context"},
    )
    print(f"  Stored memory ID: {mid2}")

    print("\n── Memory search test ──")
    results = store.search(
        client_id="lumina_test",
        query="listing suppression sessions drop",
        top_k=2,
    )
    for r in results:
        print(f"  [{r['score']}] {r['memory'][:80]}...")

    print("\n── Get all memories ──")
    all_memories = store.get_all(client_id="lumina_test")
    print(f"  Total memories: {len(all_memories)}")
    for m in all_memories:
        print(f"  - {m['memory'][:70]}...")

    print("\n── Persistence test ──")
    print("  Creating second MemoryStore instance (simulates new process)...")
    store2 = MemoryStore()
    persisted = store2.get_all(client_id="lumina_test")
    print(f"  Memories visible in new instance: {len(persisted)}")

    print("\n── Cleanup ──")
    store.delete_all("lumina_test")
    print(f"  Memories after delete_all: {len(store.get_all('lumina_test'))}")