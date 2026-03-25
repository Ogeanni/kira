"""
knowledge/retriever.py

Sits on top of the vector store and provides a clean interface
for agents to retrieve context.

Agents don't call the vector store directly — they call the retriever.
This keeps retrieval logic (filtering, top_k, namespace selection)
in one place rather than scattered across four agent files.

Two retrieval modes:
  semantic  — pure dense vector search (what we have now)
  filtered  — dense search + metadata pre-filter by doc_type or client
"""

from dataclasses import dataclass, field
from openai import OpenAI

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import get_settings
from knowledge import get_vector_store
from knowledge.vector_store import SearchResult

settings = get_settings()
client = OpenAI(api_key=settings.openai_api_key)

@dataclass
class RetrievalQuery:
    """
    What an agent sends to the retriever.

    question  — the natural language question to answer
    namespace — which client's data to search (default: global)
    doc_types — optional filter: only return these document types
                e.g. ["sop", "compliance"] excludes transcripts
    top_k     — how many chunks to return
    """
    question: str
    namespace: str = "global"
    doc_types: list[str] = field(default_factory=list)
    top_k: int = 5


@dataclass
class RetrievalResponse:
    """
    What the retriever returns to the agent.

    results      — ranked list of SearchResult objects
    query        — the original query (for tracing)
    total_found  — how many results came back
    contexts     — convenience: just the context strings for LLM prompts
    """
    results: list[SearchResult]
    query: RetrievalQuery
    total_found: int

    @property
    def contexts(self) -> list[str]:
        """Returns just the context strings — ready to paste into a prompt."""
        return [r.context for r in self.results]

    @property
    def top_context(self) -> str:
        """Returns the single best context — for simple lookups."""
        return self.results[0].context if self.results else ""
    

class Retriever:
    """
    Clean retrieval interface for agents.
    Handles embedding the query, filtering, and returning
    a structured RetrievalResponse.
    """

    def __init__(self):
        self.store = get_vector_store()

    def _embed_query(self, question: str) -> list[float]:
        """
        Embeds the query question.
        Uses the same model as ingestion — critical.
        If ingestion used text-embedding-3-small, retrieval
        must also use text-embedding-3-small. Mixing models
        produces meaningless similarity scores.
        """
        response = client.embeddings.create(
            model=settings.openai_embedding_model,
            input=question,
        )
        return response.data[0].embedding

    def _build_filters(self, doc_types: list[str]) -> dict | None:
        """
        Builds metadata filter from doc_types list.

        ChromaDB syntax  : {"doc_type": {"$in": ["sop", "compliance"]}}
        Pinecone syntax  : {"doc_type": {"$in": ["sop", "compliance"]}}

        Both use the same filter syntax for $in — one of the few
        cases where ChromaDB and Pinecone are compatible.
        Single doc_type uses $eq for slightly better performance.
        """
        if not doc_types:
            return None
        if len(doc_types) == 1:
            return {"doc_type": {"$eq": doc_types[0]}}
        return {"doc_type": {"$in": doc_types}}

    def retrieve(self, query: RetrievalQuery) -> RetrievalResponse:
        """
        Main retrieval method. Called by agents.

        Flow:
          1. Embed the question
          2. Build metadata filters if doc_types specified
          3. Query the vector store
          4. Return structured RetrievalResponse
        """
        query_embedding = self._embed_query(query.question)
        filters = self._build_filters(query.doc_types)

        results = self.store.query(
            query_embedding=query_embedding,
            top_k=query.top_k,
            filters=filters,
            namespace=query.namespace,
        )

        return RetrievalResponse(
            results=results,
            query=query,
            total_found=len(results),
        )

    def retrieve_for_client(
        self,
        question: str,
        client_id: str,
        doc_types: list[str] | None = None,
        top_k: int | None = None,
    ) -> RetrievalResponse:
        """
        Convenience method for client-specific retrieval.
        Searches both global and client namespaces, merges,
        re-ranks by score, returns top_k.

        Why both namespaces?
        Global documents (SOPs, compliance) apply to all clients.
        Client documents (transcripts) contain client-specific rules.
        An agent asking about Natura needs both.
        """
        top_k = top_k or settings.retrieval_top_k

        query_embedding = self._embed_query(question)
        filters = self._build_filters(doc_types or [])

        # Search global namespace
        global_results = self.store.query(
            query_embedding=query_embedding,
            top_k=top_k,
            filters=filters,
            namespace="global",
        )

        # Search client namespace
        client_results = self.store.query(
            query_embedding=query_embedding,
            top_k=top_k,
            filters=filters,
            namespace=client_id,
        )

        # Merge and re-rank by score
        all_results = global_results + client_results
        all_results.sort(key=lambda r: r.score, reverse=True)
        top_results = all_results[:top_k]

        query = RetrievalQuery(
            question=question,
            namespace=client_id,
            doc_types=doc_types or [],
            top_k=top_k,
        )

        return RetrievalResponse(
            results=top_results,
            query=query,
            total_found=len(top_results),
        )


if __name__ == "__main__":
    retriever = Retriever()
    backend = type(retriever.store).__name__
    print(f"\nRetriever ready — backend: {backend}\n")

    # Test 1: global semantic search
    print("── Test 1: global search ──")
    response = retriever.retrieve(RetrievalQuery(
        question="What happens when a client loses the Buy Box?",
        namespace="global",
        top_k=3,
    ))
    for i, r in enumerate(response.results, 1):
        print(f"  {i}. [{r.score}] {r.text[:70]}...")

    # Test 2: filtered search — SOPs only
    print("\n── Test 2: filtered — SOPs only ──")
    response = retriever.retrieve(RetrievalQuery(
        question="What are the steps for keyword research?",
        namespace="global",
        doc_types=["sop"],
        top_k=3,
    ))
    for i, r in enumerate(response.results, 1):
        print(f"  {i}. [{r.score}] {r.metadata.get('doc_type')} | {r.text[:60]}...")

    # Test 3: client-specific — merges global + natura
    print("\n── Test 3: client-specific — natura ──")
    response = retriever.retrieve_for_client(
        question="What are Natura's brand restrictions?",
        client_id="natura",
        top_k=4,
    )
    for i, r in enumerate(response.results, 1):
        ns = r.metadata.get("client", "unknown")
        print(f"  {i}. [{r.score}] ({ns}) {r.text[:60]}...")

    print(f"\ncontexts property: {len(response.contexts)} strings ready for LLM prompt")