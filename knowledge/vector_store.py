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
    chunk_id: str
    text: str
    context: str
    score: float
    metadata: dict

    # Promoted from metadata for convenient access
    # These are the fields every consumer needs as attributes
    doc_type: str = "unknown"
    source_file: str = ""
    client_id: str = ""

    def __post_init__(self):
        """
        Promotes metadata fields to top-level attributes after init.

        Why __post_init__ rather than setting them in __init__?
        @dataclass generates __init__ automatically from field
        definitions. __post_init__ runs after that generated __init__,
        giving us a hook to do secondary initialisation — in this case,
        extracting fields from the metadata dict that was just set.

        Why not just access metadata directly everywhere?
        Two reasons:
        1. Callers shouldn't need to know that doc_type lives in metadata.
           That's an implementation detail of how Pinecone stores things.
           The public interface of SearchResult should be flat and explicit.
        2. If the metadata key name changes (e.g., "doc_type" → "document_type"),
           fix it in one place here, not everywhere metadata is accessed.
        """
        if self.metadata:
            self.doc_type = self.metadata.get("doc_type", "unknown")
            self.source_file = self.metadata.get("source_file", "")
            self.client_id = self.metadata.get(
                "client_id",
                self.metadata.get("client", "")
            )
            # Populate context from window_context if context is empty
            # New pipeline chunks use 'window_context' key
            # Old chunks used 'context' key directly
            if not self.context:
                self.context = self.metadata.get(
                    "window_context",
                    self.metadata.get("text", "")
                )


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