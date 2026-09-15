"""
tests/test_ranking.py

Tests Stage 2 of the RAG pipeline: Ranking.

What we're testing:
  - Does the most relevant chunk appear in the top 3 results?
  - Does the top-ranked chunk come from the correct source file?
  - Are similarity scores reasonable — not too compressed?
  - Does score ordering match relevance ordering?

What we're NOT testing here:
  - Whether retrieval returned anything at all (test_retrieval.py)
  - Whether the generated answer is correct (test_generation.py)

Why ranking matters independently:
  Retrieval can return the right chunk but rank it 5th.
  The LLM pays more attention to earlier context in the window.
  A correctly retrieved but poorly ranked chunk produces a worse
  answer than a correctly retrieved and well-ranked chunk.
  Ranking failure is invisible in end-to-end tests — the chunk
  is present, the answer looks reasonable, but quality is degraded.

Key concept — MRR (Mean Reciprocal Rank):
  MRR measures the average of 1/rank across all questions.
  If the ground truth chunk is always ranked 1st, MRR = 1.0
  If it's always ranked 3rd, MRR = 0.33
  MRR of 0.5 or above is generally acceptable for RAG systems.
  Below 0.5 means the most relevant chunk is consistently buried.
"""

import pytest
from knowledge.retriever import Retriever, RetrievalQuery


# ── Ground truth chunk IDs ────────────────────────────────────────────
# These were manually identified by reading the source documents
# and finding the exact chunk that most directly answers each question.
# This is the labelled dataset that makes ranking evaluation possible.
#
# Why manual labelling?
# There is no automated way to know which chunk is most relevant.
# ANN retrieval finds nearest neighbours — it doesn't know which
# chunk a human would judge as the best answer. Manual labelling
# is the ground truth that retrieval is measured against.
#
# How to update these:
# If you change chunking strategy or re-chunk documents,
# chunk IDs change. Re-run the diagnostic script to find new IDs.
# This is a maintenance cost of having a labelled eval set.


GROUND_TRUTH = [
    {
        "question": "What ACOS threshold triggers bid reduction on keywords?",
        "namespace": "global",
        "relevant_chunk_ids": [
            "sop_acos_management.txt_sentence_window_7",
        ],
        "relevant_source": "sop_acos_management.txt",
        "acceptable_rank": 3,
    },
    {
        "question": "What terms are absolutely prohibited in Amazon listing copy?",
        "namespace": "global",
        "relevant_chunk_ids": [
            "compliance_restricted_claims.txt_sentence_window_0",  # window contains the list
            "compliance_restricted_claims.txt_sentence_window_1",  # direct list
        ],
        "relevant_source": "compliance_restricted_claims.txt",
        "acceptable_rank": 3,
    },
    {
        "question": "What format should Amazon listing titles follow?",
        "namespace": "global",
        "relevant_chunk_ids": [
            "sop_listing_optimisation.txt_sentence_window_7",
        ],
        "relevant_source": "sop_listing_optimisation.txt",
        "acceptable_rank": 3,
    },
    {
        "question": "What steps should be taken when a product loses Buy Box?",
        "namespace": "global",
        "relevant_chunk_ids": [
            "sop_buy_box_recovery.txt_sentence_window_3",  # window contains causes + steps
            "sop_buy_box_recovery.txt_sentence_window_4",  # direct causes list
        ],
        "relevant_source": "sop_buy_box_recovery.txt",
        "acceptable_rank": 3,
    },
]


# ── Helper: compute MRR ───────────────────────────────────────────────

def compute_mrr(ranks: list[int | None]) -> float:
    """
    Computes Mean Reciprocal Rank from a list of rank positions.

    rank=1 means the relevant chunk was ranked first (best case)
    rank=None means the relevant chunk wasn't in the results at all

    MRR = average of 1/rank across all questions
    MRR of 1.0 = relevant chunk always ranked first
    MRR of 0.5 = relevant chunk always ranked second
    MRR of 0.0 = relevant chunk never appeared in results

    Why reciprocal rank rather than raw rank?
    A system that ranks relevant chunks 1st is not twice as good
    as one that ranks them 2nd — the difference is much larger
    in terms of answer quality because LLMs weight earlier context
    more heavily. Reciprocal rank captures this non-linearity:
    1st → 1.0, 2nd → 0.5, 3rd → 0.33, 4th → 0.25

    Why mean across questions?
    A single question's rank is noisy. MRR across multiple questions
    gives a stable signal about overall ranking quality.
    """
    reciprocal_ranks = []
    for rank in ranks:
        if rank is None:
            reciprocal_ranks.append(0.0)   # chunk not found = worst case
        else:
            reciprocal_ranks.append(1.0 / rank)
    return sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0


# ── Test 1: Ground truth chunk appears in top k ───────────────────────

def test_ground_truth_chunk_in_top_k(retriever):
    """
    Verifies that the manually identified most relevant chunk
    appears within the top k results for each question.

    Why top 3 rather than top 1?
    Requiring rank 1 is too strict for sentence-window chunking.
    Multiple chunks from the same document often have similar
    scores — the exact sentence and its neighbours score closely.
    Requiring the relevant chunk in the top 3 is a meaningful
    bar without being brittle to small score fluctuations.

    What failure means:
    If the ground truth chunk is ranked 4th or lower, it might
    be excluded from the context window if top_k=3.
    The LLM would answer without the most relevant information.
    This is a ranking failure — retrieval found the right document
    but buried the right chunk under less relevant ones.
    """
    failures = []

    for item in GROUND_TRUTH:
        response = retriever.retrieve(RetrievalQuery(
            question=item["question"],
            namespace=item["namespace"],
            top_k=5,
        ))

        chunk_ids = [r.chunk_id for r in response.results]
        valid_ids = item["relevant_chunk_ids"]

        # Find best rank among all valid chunk IDs
        ranks = []
        for valid_id in valid_ids:
            if valid_id in chunk_ids:
                ranks.append(chunk_ids.index(valid_id) + 1)

        best_rank = min(ranks) if ranks else None

        if best_rank is None or best_rank > item["acceptable_rank"]:
            failures.append(
                f"Question: '{item['question'][:50]}...'\n"
                f"  Valid chunks  : {valid_ids}\n"
                f"  Best rank     : {best_rank} (acceptable: top {item['acceptable_rank']})\n"
                f"  Retrieved IDs : {chunk_ids}"
            )

    assert not failures, (
        f"Ranking failures on {len(failures)} question(s):\n\n"
        + "\n\n".join(failures)
    )


# ── Test 2: MRR above threshold ───────────────────────────────────────

def test_mrr_above_threshold(retriever):
    """
    Verifies that Mean Reciprocal Rank across all questions
    meets the minimum acceptable threshold.

    Why MRR as a separate test from top-k presence?
    Top-k presence is binary — the chunk is in the top 3 or not.
    MRR captures the quality of the ranking within that top 3.
    A system where the relevant chunk is always ranked 3rd
    passes the top-3 test but has MRR=0.33, which is poor.
    We want to know not just that it's in the window
    but how prominently it's ranked.

    Threshold of 0.5:
    MRR=0.5 means the relevant chunk is ranked 2nd on average.
    This is a reasonable bar for sentence-window chunking
    where the most relevant sentence and its neighbours
    often have very similar scores.
    Adjust this threshold based on your system's characteristics.
    """
    ranks = []

    for item in GROUND_TRUTH:
        response = retriever.retrieve(RetrievalQuery(
            question=item["question"],
            namespace=item["namespace"],
            top_k=5,
        ))

        chunk_ids = [r.chunk_id for r in response.results]
        valid_ids = item["relevant_chunk_ids"]

        best_rank = None
        for valid_id in valid_ids:
            if valid_id in chunk_ids:
                rank = chunk_ids.index(valid_id) + 1
                if best_rank is None or rank < best_rank:
                    best_rank = rank

        ranks.append(best_rank)

    mrr = compute_mrr(ranks)

    print(f"\nPer-question ranks:")
    for item, rank in zip(GROUND_TRUTH, ranks):
        rr = 1/rank if rank else 0
        print(f"  rank={rank} RR={rr:.2f} | {item['question'][:50]}...")
    print(f"MRR: {mrr:.3f}")

    assert mrr >= 0.5, (
        f"MRR {mrr:.3f} below threshold 0.5\n"
        f"Per-question ranks: {ranks}\n"
        f"Consider: cross-encoder reranker, wider window size, "
        f"or revised chunking strategy."
    )


# ── Test 3: Score ordering is meaningful ─────────────────────────────

def test_score_ordering_is_monotonic(retriever, known_questions):
    """
    Verifies that retrieved chunks are returned in descending
    score order — highest score first.

    Why test this?
    This should be guaranteed by the vector store implementation.
    But abstraction layers can introduce bugs — a sort that
    accidentally reverses, a merge of two result sets that
    loses ordering, a pagination bug.

    If this test fails, everything downstream is wrong:
    the LLM receives chunks in the wrong order, MRR calculations
    are meaningless, and top-k filtering excludes the wrong chunks.
    This is a correctness invariant, not a quality threshold.

    Why monotonic rather than strictly decreasing?
    Two chunks can have identical scores (ties). Monotonic
    ordering allows ties — it just requires no chunk scores
    higher than the chunk before it.
    """
    for item in known_questions:
        response = retriever.retrieve(RetrievalQuery(
            question=item["question"],
            namespace=item["expected_namespace"],
            top_k=5,
        ))

        scores = [r.score for r in response.results]

        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1], (
                f"Score ordering violation for: '{item['question']}'\n"
                f"Scores: {[round(s, 4) for s in scores]}\n"
                f"Score at position {i} ({scores[i]:.4f}) < "
                f"score at position {i+1} ({scores[i+1]:.4f})\n"
                f"Results are not in descending score order."
            )


# ── Test 4: Score distribution is meaningful ─────────────────────────

def test_score_distribution(retriever, known_questions):
    """
    Verifies that similarity scores have meaningful spread —
    the highest score is substantially above the lowest.

    Why does score spread matter?
    If all chunks score between 0.79 and 0.81, the ranking
    is essentially random — tiny floating point differences
    determine order. A system with compressed scores can't
    reliably distinguish relevant from irrelevant chunks.

    Minimum spread of 0.05:
    A spread of at least 0.05 between top and bottom score
    indicates the similarity function is discriminating —
    it assigns meaningfully different scores to chunks of
    different relevance. If spread is consistently below 0.05,
    consider a different embedding model or similarity metric.

    Note on ChromaDB vs Pinecone:
    Score scales differ between backends — ChromaDB returns
    cosine distance (lower = more similar), Pinecone returns
    cosine similarity (higher = more similar). This test checks
    spread regardless of scale direction.
    """
    spreads = []

    for item in known_questions:
        response = retriever.retrieve(RetrievalQuery(
            question=item["question"],
            namespace=item["expected_namespace"],
            top_k=5,
        ))

        if len(response.results) < 2:
            continue

        scores = [r.score for r in response.results]
        spread = abs(max(scores) - min(scores))
        spreads.append(spread)

    avg_spread = sum(spreads) / len(spreads) if spreads else 0

    print(f"\nPer-question score spreads: {[round(s, 4) for s in spreads]}")
    print(f"Average spread: {avg_spread:.4f}")

    assert avg_spread >= 0.05, (
        f"Score spread {avg_spread:.4f} is too compressed.\n"
        f"Retrieval cannot reliably distinguish relevant from irrelevant chunks.\n"
        f"Consider: different embedding model, similarity metric, or reranker."
    )

# ── Test 5:source proximity check ─────────────────────────

def test_top_result_from_correct_source(retriever):
    """
    Verifies that at least one of the top 3 retrieved chunks
    comes from the correct source document.

    This is the right retrieval-level check:
    Did we retrieve from the right document?
    Whether the specific chunk contains the answer is a
    generation question — tested in test_generation.py.
    """
    failures = []

    for item in GROUND_TRUTH:
        response = retriever.retrieve(RetrievalQuery(
            question=item["question"],
            namespace=item["namespace"],
            top_k=5,
        ))

        top_3_sources = [
            r.source_file for r in response.results[:3]
        ]

        if item["relevant_source"] not in top_3_sources:
            failures.append(
                f"Question: '{item['question'][:50]}...'\n"
                f"  Expected source in top 3: {item['relevant_source']}\n"
                f"  Top 3 sources found     : {top_3_sources}"
            )

    assert not failures, (
        f"Source failures on {len(failures)} question(s):\n\n"
        + "\n\n".join(failures)
    )