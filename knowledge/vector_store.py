"""
knowledge/vector_store.py

Abstract base class for all vector store backends.
Nothing outside knowledge/ ever imports ChromaDB or Pinecone directly.
Everything talks to VectorStore.

This pattern means:
  - Swapping backends = one line in .env
  - Testing = easy to mock
  - Adding a new backend (Weaviate, pgvector) = implement 3 methods
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class SearchResult:
    """
    A single result returned from a vector search.

    chunk_id    — matches the chunk_id from ingestion
    text        — the embedded text (the sentence)
    context     — window_context — what the LLM will read
    score       — cosine similarity 0.0 to 1.0, higher = more similar
    metadata    — doc_type, client, source_file etc
    """
    chunk_id: str
    text: str
    context: str
    score: float
    metadata: dict


class VectorStore(ABC):
    """
    Every vector store backend implements these three methods.
    Nothing else is required at the interface level.
    """

    @abstractmethod
    def upsert(self, chunks: list[dict], embeddings: list[dict]) -> int:
        """
        Stores chunks and their embeddings.
        Returns the number of vectors upserted.

        chunks     — list of Chunk dicts from ingestion/chunker.py
        embeddings — list of EmbeddingResult dicts, same order as chunks
        """
        ...

    @abstractmethod
    def query(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filters: dict | None = None,
        namespace: str = "global",
    ) -> list[SearchResult]:
        """
        Returns the top_k most similar chunks to the query embedding.

        filters   — e.g. {"doc_type": "sop"} to restrict results
        namespace — client isolation in Pinecone, collection in ChromaDB
        """
        ...

    @abstractmethod
    def delete(self, chunk_ids: list[str], namespace: str = "global") -> None:
        """Removes specific chunks by ID."""
        ...

    @abstractmethod
    def count(self, namespace: str = "global") -> int:
        """Returns the number of vectors stored."""
        ...