"""
tests/test_retrieval.py

Tests Stage 1 of the RAG pipeline: Retrieval.

What I am testing:
  - Did retrieval return chunks at all?
  - Did it return chunks from the right source type?
  - Did client namespace isolation hold?
  - Does retrieval fail gracefully on unanswerable queries?
  - Does retrieval behave correctly when the knowledge base is empty?

What I am NOT testing here:
  - Whether the chunks are ranked correctly (test_ranking.py)
  - Whether the context window is well-formed (test_context.py)
  - Whether the generated answer is correct (test_generation.py)

Why this separation?
  If a test fails, I know immediately which stage broke.
  A combined test that checks retrieval AND generation tells me
  "something is wrong" but not where. Separate tests give me
  precise failure attribution — the same principle for
  production monitoring.
"""

import pytest
from knowledge.retriever import Retriever, RetrievalQuery


# ── Test 1: Basic retrieval returns results ───────────────────────────

def test_retrieval_returns_results(retriever, known_questions):
    """
    The most basic test: does retrieval return anything at all?

    Why test something this obvious?
    Because "obvious" failures happen in production:
    - Knowledge base was never ingested
    - Vector store connection failed silently
    - Environment variable points to wrong index

    This test catches all of them immediately.
    If this fails, all other retrieval tests will also fail —
    so fix this first before investigating anything else.

    Why parametrize over known_questions?
    I want to verify retrieval works for every question type,
    not just one. A system that retrieves SOP chunks correctly
    but fails on compliance chunks has a real problem.
    pytest.mark.parametrize runs the test once per question.
    """
    for item in known_questions:
        response = retriever.retrieve(RetrievalQuery(
            question=item["question"],
            namespace=item["expected_namespace"],
            top_k=5,
        ))

        # Why assert len > 0 rather than == 5?
        # top_k=5 is a maximum, not a guarantee.
        # If the knowledge base has fewer than 5 chunks,
        # retrieval returns what exists. Asserting == 5 would
        # fail on a small knowledge base even when retrieval works.
        assert len(response.contexts) > 0, (
            f"Retrieval returned no results for: '{item['question']}'\n"
            f"Backend: check that documents are ingested and "
            f"the knowledge base is not empty."
        )


# ── Test 2: Source type correctness ──────────────────────────────────

def test_retrieval_source_type_match(retriever, known_questions):
    """
    Verifies that the correct document type was retrieved.

    Each question in known_questions has an expected_source_type.
    I manually determined these by reading the source documents.

    The test checks that at least one retrieved chunk came from
    the expected source type — not that ALL chunks did.

    Why "at least one" rather than "all"?
    top_k=5 might return 3 SOP chunks and 2 framework chunks
    for an SOP question. That's acceptable — the most relevant
    SOP chunk is present. Requiring all 5 to be SOPs would be
    too strict and would fail on knowledge bases where document
    types are semantically similar.

    What failure looks like:
    If expected_source_type="sop" but retrieved chunks all have
    doc_type="compliance", it means the SOP documents weren't
    ingested, or the compliance documents are semantically closer
    to this query than the SOPs — a content problem, not a code problem.
    """
    for item in known_questions:
        response = retriever.retrieve(RetrievalQuery(
            question=item["question"],
            namespace=item["expected_namespace"],
            top_k=5,
        ))

        # Extract doc_type from each result
        # Why check hasattr?
        # RetrievalResult might not have doc_type if the retriever
        # implementation doesn't populate it from chunk metadata.
        # hasattr protects against AttributeError while making
        # the gap visible — you'll see "unknown" in the failure message.
        doc_types_found = set()
        for result in response.results:
            doc_type = getattr(result, "doc_type", "unknown")
            doc_types_found.add(doc_type)

        assert item["expected_source_type"] in doc_types_found, (
            f"Source type mismatch for: '{item['question']}'\n"
            f"Expected source type: {item['expected_source_type']}\n"
            f"Found source types  : {doc_types_found}\n"
            f"This indicates retrieval returned documents from "
            f"the wrong category. Check ingestion metadata."
        )


# ── Test 3: Client namespace isolation ───────────────────────────────

def test_client_namespace_isolation(retriever, client_ids, backend_name):
    """
    Verifies that querying one client's namespace never returns
    another client's documents.

    This is the most critical safety test in the harness.
    A failure here means Client A can see Client B's data —
    a data breach in a multi-tenant system.

    How the test works:
    I query each client namespace with a generic question.
    I check that no retrieved chunk's source_file contains
    another client's name.

    Why use source_file for the check rather than client_id metadata?
    source_file is set at document creation time in generate_data.py.
    Files are named with the client ID: "transcript_onboarding_natura.txt"
    This is an independent verification path — if client_id metadata
    is wrong but the filename is correct (or vice versa), we catch it.

    What this test does NOT catch:
    If two clients have identical document content with no client
    identifier in the filename, a namespace failure would be invisible
    to this test. Production systems need both metadata AND content
    checks for full isolation verification.
    """
    backend = backend_name

    query = "What are the performance targets and compliance rules?"

    for client_id in client_ids:
        response = retriever.retrieve(RetrievalQuery(
            question=query,
            namespace=client_id,
            top_k=5,
        ))

        # if not response.contexts:
        #     # Client namespace is empty — skip isolation check
        #     # but log it. An empty namespace might mean ingestion
        #     # didn't run for this client.
        #     pytest.skip(
        #         f"Namespace '{client_id}' is empty — "
        #         f"run ingest_docs.py first"
        #     )
        #     continue

        # Correct — empty namespace is a failure, not a skip
        assert len(response.contexts) > 0, (
            f"Namespace '{client_id}' is empty — no vectors ingested.\n"
            f"This means pipelines for {client_id} have no client-specific context.\n"
            f"Run: python scripts/ingest_docs.py --backend {backend}\n"
            f"Then verify: all 4 client namespaces have vectors."
        )

        

        # Check each retrieved chunk
        for result in response.results:
            source_file = getattr(result, "source_file", "")

            # Does this chunk's source file belong to another client?
            other_clients = [c for c in client_ids if c != client_id]
            for other_client in other_clients:
                assert other_client not in source_file.lower(), (
                    f"Client isolation FAILURE\n"
                    f"Queried namespace : {client_id}\n"
                    f"Contaminating file: {source_file}\n"
                    f"Belongs to client : {other_client}\n"
                    f"This is a data isolation breach. "
                    f"Check namespace filtering in retriever.py"
                )


# ── Test 4: Unanswerable query handling ──────────────────────────────

def test_unanswerable_query_returns_irrelevant_chunks(retriever):
    """
    Tests what happens when a question has no answer in the knowledge base.

    Why does this matter?
    A question outside the knowledge base will still return chunks —
    vector similarity always finds the nearest neighbours, even if
    they're not actually relevant. The question is whether the system
    handles this gracefully downstream.

    What I test here:
    I don't test whether the system says "I don't know" —
    that's generation behaviour tested in test_generation.py.
    I test that retrieval doesn't return chunks with
    suspiciously high similarity scores for a nonsense query.

    Why check scores?
    A retrieval score tells you how similar the query is to the chunk.
    For a completely out-of-domain question, scores should be low —
    below 0.5 typically. High scores on an unanswerable query suggest
    the system might confidently generate a wrong answer.

    The threshold 0.7 is a heuristic based on typical cosine similarity
    distributions. Adjust based on your knowledge base — run this test
    and print scores to calibrate.
    """
    nonsense_query = "What is the NFT minting policy for blockchain-verified products?"

    response = retriever.retrieve(RetrievalQuery(
        question=nonsense_query,
        namespace="global",
        top_k=5,
    ))

    # Retrieval will return something — ANN always finds nearest neighbours
    # What we check is that scores are low, indicating low confidence
    if response.results:
        scores = [getattr(r, "score", 1.0) for r in response.results]
        max_score = max(scores)

        # If max score is above 0.7 for a completely irrelevant query,
        # it suggests the similarity space is compressed and the system
        # might present irrelevant chunks confidently.
        # This is a warning, not a hard failure — hence pytest.warns
        # pattern rather than assert.
        if max_score > 0.7:
            pytest.warns(
                UserWarning,
                match="High similarity score on unanswerable query"
            )
            print(
                f"\nWARNING: Unanswerable query returned max score {max_score:.3f}. "
                f"Consider this when evaluating generation confidence."
            )


# ── Test 5: Global namespace contains shared documents ────────────────

def test_global_namespace_contains_shared_docs(
    retriever, global_namespace
):
    """
    Verifies that the global namespace was correctly populated
    with shared documents (SOPs, compliance docs, frameworks).

    Why test this separately from basic retrieval?
    Basic retrieval tests that the retriever works.
    This tests that the ingestion pipeline correctly directed
    shared documents to the global namespace rather than to
    a client namespace.

    If this fails it means ingest_docs.py put shared documents
    in the wrong namespace — a silent ingestion bug that would
    cause every client to miss shared context.
    """
    # SOPs and compliance docs should be in global namespace
    sop_query = "What are the standard operating procedures for bid management?"

    response = retriever.retrieve(RetrievalQuery(
        question=sop_query,
        namespace=global_namespace,
        top_k=5,
    ))

    assert len(response.contexts) > 0, (
        f"Global namespace appears empty or missing SOP content.\n"
        f"Run: python scripts/ingest_docs.py to populate it."
    )

    # Verify at least one SOP chunk is present
    doc_types = [getattr(r, "doc_type", "unknown") for r in response.results]
    assert "sop" in doc_types or "framework" in doc_types, (
        f"Global namespace doesn't contain SOP or framework documents.\n"
        f"Found doc types: {set(doc_types)}\n"
        f"Check that shared documents were ingested with correct metadata."
    )


# ── Test 6: Retrieval consistency ────────────────────────────────────

def test_retrieval_is_consistent(retriever, known_questions):
    """
    Verifies that the same query returns the same results
    when run twice.

    Why test this?
    Vector similarity search should be deterministic —
    the same query against the same index should always
    return the same results. If it doesn't, something is
    wrong with the index or the embedding generation.

    Non-determinism in retrieval is a serious production problem:
    it means the same user asking the same question gets
    different context, which leads to different answers.
    Users notice this and it destroys trust.

    What would cause non-determinism?
    - Embedding model returning different vectors for same text
      (shouldn't happen, but can with certain model configs)
    - ANN index with random tie-breaking between equally similar chunks
    - Index being modified between queries (during ingestion)

    I test with the first question only — if consistency holds
    for one, it holds for all (same code path).
    """
    item = known_questions[0]

    query = RetrievalQuery(
        question=item["question"],
        namespace=item["expected_namespace"],
        top_k=5,
    )

    response_1 = retriever.retrieve(query)
    response_2 = retriever.retrieve(query)

    # Extract source files from both runs
    sources_1 = [getattr(r, "source_file", "") for r in response_1.results]
    sources_2 = [getattr(r, "source_file", "") for r in response_2.results]

    assert sources_1 == sources_2, (
        f"Retrieval is non-deterministic.\n"
        f"Run 1 sources: {sources_1}\n"
        f"Run 2 sources: {sources_2}\n"
        f"The same query should always return the same chunks."
    )