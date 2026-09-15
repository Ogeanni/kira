"""
knowledge/retriever.py

Sits on top of the vector store and provides a clean interface
for agents to retrieve context.

Agents don't call the vector store directly — they call the retriever.
This keeps retrieval logic (filtering, top_k, namespace selection)
in one place rather than scattered across four agent files.

Agent-facing retrieval with optional cross-encoder reranking.

Architecture:
  Stage 1 — Bi-encoder retrieval (fast, approximate)
    Embed the query, query the vector store, retrieve candidates.
    Speed: O(1) — vector similarity is a single matrix operation.
    Weakness: evaluates query and chunks independently, misses
    semantic relationships that only appear when evaluated together.

  Stage 2 — Cross-encoder reranking (accurate, exhaustive)
    Score each query-chunk pair jointly using a transformer model.
    Speed: O(n) — one forward pass per candidate chunk.
    Strength: understands semantic relationships, not just similarity.
    Cost: 100-500ms on CPU for 15 candidates.

The two-stage pattern is the production standard because:
  - Running a cross-encoder on the entire index is too slow
  - Running a bi-encoder alone misses vocabulary mismatches
  - Combining them gives accuracy close to cross-encoder at
    speed close to bi-encoder
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import lru_cache
from typing import TYPE_CHECKING
import sys
from pathlib import Path

from openai import OpenAI


ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import get_settings
from knowledge import get_vector_store
from knowledge.vector_store import SearchResult

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)
settings = get_settings()



# ── Data contracts ────────────────────────────────────────────────────

@dataclass
class RetrievalQuery:
    """
    What an agent sends to the retriever.

    question  — the natural language question to answer
        namespace — which client's data to search (default: global)
        doc_types — optional filter: only return these document types
                    e.g. ["sop", "compliance"] excludes transcripts
        top_k     — how many chunks to return

    Every field has a default so callers only specify what they need.
    """
    question: str
    namespace: str = "global"
    doc_types: list[str] = field(default_factory=list)
    top_k: int = field(default_factory=lambda: settings.retrieval_top_k)

@dataclass
class RetrievalResponse:
    """
    What the retriever returns to the agent.
    
    results      — ranked list of SearchResult objects
    query        — the original query (for tracing)
    total_found  — how many results came back
    contexts     — convenience: just the context strings for LLM prompts
    
    results are always sorted by relevance score, highest first.
    contexts is a convenience property for agent consumption.
    """
    results: list[SearchResult]
    query: RetrievalQuery
    total_found: int
    reranked: bool = False   # flag so callers know if reranking was applied
    candidates: list[SearchResult] = field(default_factory=list)


    @property
    def contexts(self) -> list[str]:
        """
        Returns the text content of each result.
        Uses window_context if available — that's what gets
        passed to the LLM, not just the embedded sentence.

        Why window_context over text?
        text is the embedded sentence — the single sentence
        that determined retrieval score.
        window_context is the surrounding sentences — what
        the LLM actually uses to generate an answer.
        Returning text here would mean the LLM only sees one
        sentence per retrieved chunk instead of the full window.
        """
        return [
            getattr(r, "window_context", None) or r.text
            for r in self.results
        ]
    
# ── Reranker ──────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_reranker():
    """
    Loads the cross-encoder model. Cached with lru_cache so the model
    is only loaded once per process — loading takes 2-3 seconds.

    Why lru_cache(maxsize=1)?
    We only ever need one reranker instance. lru_cache memoizes
    the result of the first call and returns it on subsequent calls.
    This is the lazy singleton pattern — the model isn't loaded
    until the first retrieval that needs it, and it's never loaded
    twice in the same process.

    Why not load at module import time?
    Two reasons:
    1. Loading a transformer model at import time slows down every
       script that imports the retriever — even scripts that don't
       use retrieval. Lazy loading means you only pay the cost when
       you actually retrieve.
    2. Tests that mock the retriever don't have to deal with a
       model download on import.
    """
    from sentence_transformers import CrossEncoder
    logger.info(f"Loading reranker: {settings.reranker_model}")
    return CrossEncoder(settings.reranker_model)


def _rerank(
    question: str,
    candidates: list[SearchResult],
    top_k: int,
) -> list[SearchResult]:
    """
    Reranks candidates using a cross-encoder.

    The cross-encoder takes (question, chunk_text) pairs and scores
    how well each chunk answers the question. Unlike the bi-encoder
    which scores question and chunk independently, the cross-encoder
    reads both together — it can understand semantic relationships
    that only appear when the two are considered jointly.

    Why use text not window_context for reranking?
    The cross-encoder should score the embedded sentence — the
    specific content unit. The window provides context for the LLM
    but the relevance signal comes from the sentence itself.
    Using the full window would give the cross-encoder too much
    text and dilute the relevance signal.

    Args:
        question:   the original query string
        candidates: results from bi-encoder retrieval
        top_k:      how many to return after reranking

    Returns:
        top_k results sorted by cross-encoder score, highest first
    """
    reranker = _load_reranker()

    # Build (question, chunk_text) pairs for the cross-encoder
    pairs = [(question, r.text) for r in candidates]

    # Score all pairs — returns a list of floats
    # Higher score = more relevant
    scores = reranker.predict(pairs)

    # Attach cross-encoder scores to results and sort
    scored = sorted(
        zip(scores, candidates),
        key=lambda x: x[0],
        reverse=True,   # highest score first
    )

    # Return top_k results with updated scores
    reranked = []
    for score, result in scored[:top_k]:
        # Create a new SearchResult with the cross-encoder score
        # We replace the bi-encoder similarity score with the
        # cross-encoder relevance score — they're on different scales
        # but both higher-is-better, so comparison is meaningful
        reranked.append(SearchResult(
            chunk_id=result.chunk_id,
            text=result.text,
            context=result.context,
            score=float(score),
            metadata=result.metadata,
            # Preserve window_context — it's still what the LLM uses
        ))

    return reranked


# ── Retriever ─────────────────────────────────────────────────────────

class Retriever:
    """
    Agent-facing retrieval with optional cross-encoder reranking.

    Agents call retrieve() and receive ranked results.
    Whether reranking is applied is controlled by settings.reranker_enabled.
    Agents are unaware of this — their interface is unchanged.

    This is the open/closed principle: the Retriever is open for
    extension (reranking can be added/removed via config) but closed
    for modification — agent code doesn't change when reranking is toggled.
    """

    def __init__(self):
        self.store = get_vector_store()
        self._openai = OpenAI(api_key=settings.openai_api_key)

    def _embed_query(self, question: str) -> list[float]:
        """
        Embeds the query question.
        Uses the same model as ingestion — critical.
        If ingestion used text-embedding-3-small, retrieval
        must also use text-embedding-3-small. Mixing models
        produces meaningless similarity scores.
        """
        response = self._openai.embeddings.create(
            model=settings.openai_embedding_model,
            input=question,
        )
        return response.data[0].embedding

    def _build_filters(self, doc_types: list[str]) -> dict | None:
        """
        Builds metadata filters for doc_type filtering.

        ChromaDB syntax  : {"doc_type": {"$in": ["sop", "compliance"]}}
        Pinecone syntax  : {"doc_type": {"$in": ["sop", "compliance"]}}

        Both use the same filter syntax for $in — one of the few
        cases where ChromaDB and Pinecone are compatible.
        Single doc_type uses $eq for slightly better performance.
        """
        # if not doc_types:
        #     return None
        # if len(doc_types) == 1:
        #     return {"doc_type": {"$eq": doc_types[0]}}
        # return {"doc_type": {"$in": doc_types}}

        if not doc_types:
            return {}
        return {"doc_type": {"$in": doc_types}}

    def _candidate_count(self, top_k: int) -> int:
        """
        Returns how many candidates to retrieve before reranking.
 
        Reranking disabled: retrieve exactly top_k.
        Reranking enabled: retrieve top_k * multiplier so the
        reranker has enough candidates to promote the correct chunk
        even if the bi-encoder ranked it low.
        """
        if settings.reranker_enabled:
            return top_k * settings.reranker_candidate_multiplier
        return top_k

    def retrieve(self, query: RetrievalQuery) -> RetrievalResponse:
        """
         Main retrieval method. Called by agents.

        Flow:
          1. Embed the question
          2. Build metadata filters if doc_types specified
          3. Query the vector store for candidates
             (more candidates than needed if reranking is enabled)
          4. Rerank candidates if reranker is enabled
          5. Return structured RetrievalResponse

        The candidate count expansion (top_k * multiplier) is the
        key design decision. If I retrieved only top_k=5 and then
        reranked, I'd rerank the same 5 bi-encoder results.
        The reranker can only promote chunks that were retrieved.
        Retrieving more candidates gives the reranker more to work
        with — it can find the correct chunk even if the bi-encoder
        ranked it 8th.
        """
        query_embedding = self._embed_query(query.question)
        filters = self._build_filters(query.doc_types)
 
        results = self.store.query(
            query_embedding=query_embedding,
            top_k=self._candidate_count(query.top_k),
            filters=filters,
            namespace=query.namespace,
        )

        candidates = results  # bi-encoder output before reranking
        reranked = False
        if settings.reranker_enabled and results:
            results = _rerank(
                question=query.question,
                candidates=results,
                top_k=query.top_k,
            )
            reranked = True
 
        return RetrievalResponse(
            results=results,
            query=query,
            total_found=len(results),
            reranked=reranked,
            candidates=candidates,
        )


    def retrieve_for_client(
        self,
        question: str,
        client_id: str,
        doc_types: list[str] | None = None,
        top_k: int | None = None,
    ) -> RetrievalResponse:
        """
        Client-specific retrieval — merges global and client namespaces.
 
        Why both namespaces?
        Global documents (SOPs, compliance) apply to all clients.
        Client documents (transcripts) contain client-specific rules.
        An agent querying for Natura needs both.
 
        Why rerank after merge rather than sort by bi-encoder score?
        Bi-encoder scores from two namespace queries are not directly
        comparable. Global namespace returns scores of 0.75-0.80,
        client namespace returns 0.55-0.65 — simply because global
        has more semantically similar documents. Sorting by bi-encoder
        score after merge systematically favours global results.
 
        The cross-encoder fixes this: scores all candidates against
        the query jointly regardless of source namespace. A client-
        specific chunk that directly answers the question will score
        higher than a tangentially related global chunk, even if the
        global chunk had a higher bi-encoder similarity score.
 
        Flow:
          1. Embed the question once — shared across both queries
          2. Query global namespace for candidates
          3. Query client namespace for candidates
          4. Merge all candidates
          5. Rerank merged candidates if enabled
          6. Return top_k results
        """
        top_k = top_k or settings.retrieval_top_k
        filters = self._build_filters(doc_types or [])
        query_embedding = self._embed_query(question)
        candidate_count = self._candidate_count(top_k)
 
        global_results = self.store.query(
            query_embedding=query_embedding,
            top_k=candidate_count,
            filters=filters,
            namespace="global",
        )
 
        client_results = self.store.query(
            query_embedding=query_embedding,
            top_k=candidate_count,
            filters=filters,
            namespace=client_id,
        )
 
        all_candidates = global_results + client_results

        candidates = all_candidates
        # reranked = False
        # if settings.reranker_enabled and all_candidates:
        #     top_results = _rerank(
        #         question=question,
        #         candidates=all_candidates,
        #         top_k=top_k,
        #     )
        #     reranked = True
        # else:
        #     # Known limitation without reranker: systematically
        #     # favours global results due to score incompatibility
        #     all_candidates.sort(key=lambda r: r.score, reverse=True)
        #     top_results = all_candidates[:top_k]

        # Reranker is disabled for client-specific queries.
        # Client transcript chunks use conversational vocabulary that
        # systematically mismatches MS MARCO cross-encoder training data.
        # The bi-encoder produces better ranking for transcript-heavy
        # client queries — bi-encoder similarity is more robust to
        # informal vocabulary than cross-encoder lexical co-occurrence.
        # Reranker remains active for single-namespace global queries
        # via retrieve() where it demonstrably improves procedural queries.
        all_candidates.sort(key=lambda r: r.score, reverse=True)
        top_results = all_candidates[:top_k]
        reranked = False
 
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
            reranked=reranked,
            candidates=candidates,
        )

if __name__ == "__main__":
    retriever = Retriever()
    backend = type(retriever.store).__name__
    reranker_status = "enabled" if settings.reranker_enabled else "disabled"
    print(f"\nRetriever ready — backend: {backend} | reranker: {reranker_status}\n")
 
    # Test 1: previously failing Buy Box query
    print("── Test 1: Buy Box recovery steps (previously failing) ──")
    response = retriever.retrieve(RetrievalQuery(
        question="What steps should be taken when a product loses Buy Box?",
        namespace="global",
        top_k=5,
    ))
    print(f"Reranked: {response.reranked}")
    for i, r in enumerate(response.results, 1):
        print(f"  {i}. [score: {r.score:.4f}] {r.chunk_id}")
        print(f"     {r.text[:80]}...")
 
    # Test 2: ACOS threshold — should still rank chunk_7 first
    print("\n── Test 2: ACOS threshold ──")
    response = retriever.retrieve(RetrievalQuery(
        question="What ACOS threshold triggers bid reduction on keywords?",
        namespace="global",
        top_k=5,
    ))
    print(f"Reranked: {response.reranked}")
    for i, r in enumerate(response.results, 1):
        print(f"  {i}. [score: {r.score:.4f}] {r.chunk_id}")
 
    # Test 3: client-specific — merges global + natura
    print("\n── Test 3: client-specific — natura ──")
    response = retriever.retrieve_for_client(
        question="What are Natura's brand restrictions?",
        client_id="natura",
        top_k=4,
    )
    print(f"Reranked: {response.reranked}")
    for i, r in enumerate(response.results, 1):
        ns = r.metadata.get("client", "unknown")
        print(f"  {i}. [score: {r.score:.4f}] ({ns}) {r.text[:60]}...")
 
    print(f"\ncontexts property: {len(response.contexts)} strings ready for LLM")
 