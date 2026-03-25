"""
knowledge/pinecone_store.py

Pinecone implementation of VectorStore.
Used in production — managed, scalable, namespace-isolated per client.

Key differences from ChromaDB:
  - Namespaces are native Pinecone concepts, not separate collections
  - Metadata filtering uses Pinecone filter syntax
  - Upsert is batched at 100 vectors (Pinecone limit per request)
  - count() returns approximate vector count from index stats
"""

from pinecone import Pinecone

from pathlib import Path
import sys

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import get_settings
from knowledge.vector_store import VectorStore, SearchResult

settings = get_settings()

UPSERT_BATCH_SIZE = 100   # Pinecone limit per upsert request


class PineconeStore(VectorStore):

    def __init__(self):
        if not settings.has_pinecone:
            raise RuntimeError(
                "Pinecone API key not set. "
                "Add PINECONE_API_KEY to .env"
            )
        pc = Pinecone(api_key=settings.pinecone_api_key)
        self.index = pc.Index(settings.pinecone_index_name)

    def upsert(self, chunks: list[dict], embeddings: list[dict]) -> int:
        """
        Upserts vectors into Pinecone with namespace isolation per client.

        Each vector carries metadata for filtering:
          doc_type    — sop, compliance, framework, transcript
          client      — global, natura, vitalblend, peakgear, lumina
          source_file — original filename
          text        — the embedded sentence (for display)
          context     — window_context (for LLM generation)

        Batched at 100 to respect Pinecone's per-request limit.
        """
        emb_map = {e["chunk_id"]: e["embedding"] for e in embeddings}

        # Group by namespace
        by_namespace: dict[str, list] = {}
        for chunk in chunks:
            ns = chunk.get("client", settings.pinecone_default_namespace)
            if ns not in by_namespace:
                by_namespace[ns] = []
            by_namespace[ns].append(chunk)

        total = 0
        for namespace, ns_chunks in by_namespace.items():
            vectors = [
                {
                    "id": chunk["chunk_id"],
                    "values": emb_map[chunk["chunk_id"]],
                    "metadata": {
                        "text": chunk["text"],
                        "context": chunk.get("window_context") or chunk["text"],
                        "source_file": chunk["source_file"],
                        "doc_type": chunk["doc_type"],
                        "client": chunk["client"],
                        "chunk_index": chunk["chunk_index"],
                    },
                }
                for chunk in ns_chunks
                if chunk["chunk_id"] in emb_map
            ]

            # Batch upsert
            for i in range(0, len(vectors), UPSERT_BATCH_SIZE):
                batch = vectors[i: i + UPSERT_BATCH_SIZE]
                self.index.upsert(vectors=batch, namespace=namespace)
                total += len(batch)

        return total

    def query(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filters: dict | None = None,
        namespace: str = "global",
    ) -> list[SearchResult]:
        """
        Queries Pinecone with optional metadata filtering.

        Example filters:
          {"doc_type": {"$eq": "sop"}}
          {"client": {"$eq": "natura"}}
          {"doc_type": {"$in": ["sop", "compliance"]}}

        Pinecone filter syntax uses $eq, $in, $gt etc.
        These are set per query — no index rebuild needed.
        """
        response = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            namespace=namespace,
            filter=filters,
            include_metadata=True,
        )

        results = []
        for match in response.matches:
            meta = match.metadata or {}
            results.append(SearchResult(
                chunk_id=match.id,
                text=meta.get("text", ""),
                context=meta.get("context", meta.get("text", "")),
                score=round(match.score, 4),
                metadata=meta,
            ))

        return results

    def delete(self, chunk_ids: list[str], namespace: str = "global") -> None:
        self.index.delete(ids=chunk_ids, namespace=namespace)

    def count(self, namespace: str = "global") -> int:
        stats = self.index.describe_index_stats()
        ns_stats = stats.namespaces.get(namespace)
        return ns_stats.vector_count if ns_stats else 0