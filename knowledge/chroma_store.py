"""
knowledge/chroma_store.py

ChromaDB implementation of VectorStore.
Used in development — no account needed, persists to disk at data/chroma/.
One collection per namespace (client).
"""

import chromadb
from chromadb.config import Settings as ChromaSettings

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import get_settings
from knowledge.vector_store import VectorStore, SearchResult

settings = get_settings()


class ChromaStore(VectorStore):

    def __init__(self):
        self.client = chromadb.PersistentClient(
            path=str(settings.chroma_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )

    def _get_collection(self, namespace: str):
        return self.client.get_or_create_collection(
            name=f"kira_{namespace}",
            metadata={"hnsw:space": "cosine"},
        )

    def upsert(self, chunks: list[dict], embeddings: list[dict]) -> int:
        by_namespace: dict[str, list] = {}
        emb_map = {e["chunk_id"]: e["embedding"] for e in embeddings}

        for chunk in chunks:
            ns = chunk.get("client", "global")
            if ns not in by_namespace:
                by_namespace[ns] = []
            by_namespace[ns].append(chunk)

        total = 0
        for namespace, ns_chunks in by_namespace.items():
            collection = self._get_collection(namespace)
            ids = [c["chunk_id"] for c in ns_chunks]
            docs = [c.get("window_context") or c["text"] for c in ns_chunks]
            embeds = [emb_map[c["chunk_id"]] for c in ns_chunks]
            metas = [
                {
                    "text": c["text"],
                    "source_file": c["source_file"],
                    "doc_type": c["doc_type"],
                    "client": c["client"],
                    "chunk_index": c["chunk_index"],
                }
                for c in ns_chunks
            ]
            collection.upsert(
                ids=ids,
                documents=docs,
                embeddings=embeds,
                metadatas=metas,
            )
            total += len(ns_chunks)

        return total

    def query(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filters: dict | None = None,
        namespace: str = "global",
    ) -> list[SearchResult]:
        collection = self._get_collection(namespace)

        # Return empty list if collection has no vectors.
        # Avoids ChromaDB error: "Number of requested results 0,
        # cannot be negative or zero" when namespace is empty.
        if collection.count() == 0:
            return []

        where = filters if filters else None
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, collection.count()),
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        search_results = []
        for i, chunk_id in enumerate(results["ids"][0]):
            distance = results["distances"][0][i]
            score = 1 - (distance / 2)
            meta = results["metadatas"][0][i]
            context = results["documents"][0][i]
            search_results.append(SearchResult(
                chunk_id=chunk_id,
                text=meta.get("text", context),
                context=context,
                score=round(score, 4),
                metadata=meta,
            ))

        return search_results

    def delete(self, chunk_ids: list[str], namespace: str = "global") -> None:
        collection = self._get_collection(namespace)
        collection.delete(ids=chunk_ids)

    def count(self, namespace: str = "global") -> int:
        collection = self._get_collection(namespace)
        return collection.count()