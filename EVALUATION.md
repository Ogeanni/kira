# How I Evaluate KIRA

Most RAG systems are evaluated by asking "does it seem to work?"
KIRA is evaluated by measuring where it fails.

This document covers the evaluation methodology, results, what changed,
and what remains unresolved. It exists because a system you can't measure
is a system you can't trust in production.

---

## Why evaluation first

KIRA is a production system for Amazon e-commerce agencies. A wrong answer
about ACOS thresholds or a hallucinated compliance rule causes real business
harm. Before building context assembly or generation, retrieval had to be
measured and validated.

The principle: **get retrieval right before anything else. If the wrong
chunks reach the LLM, no amount of prompt engineering fixes the output.**

---

## A note on terminology

This document uses terms that need a brief explanation before they appear
in context.

**RAG (Retrieval-Augmented Generation):** An AI architecture where the
system retrieves relevant documents from a knowledge base before generating
an answer. Instead of relying on what the model memorised during training,
it looks up current, specific information first — then generates from what
it found.

**Chunk:** A small piece of a document stored in the knowledge base.
Long documents are split into chunks so the system can retrieve the
specific paragraph that answers a question, not the entire document.

**Embedding:** A numerical representation of text. Sentences with similar
meanings produce similar numbers, which is how the system finds relevant
chunks — it converts the query to numbers and finds chunks whose numbers
are closest.

**Bi-encoder (dense retrieval):** The first retrieval stage. Converts both
the query and every chunk to embeddings and finds the closest matches by
mathematical similarity. Fast but approximate.

**Cross-encoder (reranker):** The second retrieval stage. Takes the top
candidates from the bi-encoder and re-scores each query-chunk pair
together, which is more accurate but slower. Only runs on a small
candidate pool (15-30 chunks), not the entire knowledge base.

**MRR (Mean Reciprocal Rank):** A retrieval quality metric. If the correct
chunk is ranked 1st, MRR = 1.0. If it's ranked 2nd, MRR = 0.5. If 3rd,
MRR = 0.33. Averaged across all queries. Higher is better. An MRR of 0.740
means on average the most relevant chunk is ranked approximately 1.4th.

**R@5 (Recall at 5):** The fraction of queries where the correct chunk
appears somewhere in the top 5 results. R@5 of 0.868 means 87% of queries
have the right chunk in the top 5.

**nDCG@5 (Normalised Discounted Cumulative Gain at 5):** A ranking quality
metric that rewards finding the right chunk at higher positions more than
lower ones. Used to evaluate the reranker's ordering quality.

**Strict vs lenient evaluation:** Strict scoring counts a query as correct
only if the single most relevant chunk (the primary ground truth) is
retrieved. Lenient scoring counts it correct if any relevant chunk is
retrieved. Both are reported — strict is the primary metric.

---

## The benchmark

### Why I built a custom benchmark instead of using an existing one

Generic RAG benchmarks don't cover Amazon agency knowledge. A benchmark
that doesn't reflect the actual query distribution tells you nothing about
production performance.

The benchmark consists of **38 manually constructed queries** covering:
- 7 source documents (SOPs, compliance docs, frameworks, transcripts)
- 29 distinct content sections
- 9 sections with dual queries (same section, different ways of asking)
- Manually verified ground truth — the specific chunk that correctly answers each query

Every ground truth label was established by reading the source document and
the chunk text against the query. No retrieval system was run to help determine
what the right answer should be. The benchmark is independent of what the
system returns.

### Query type distribution

| Query type | Count | Why it matters |
|---|---|---|
| Direct factual | 16 | Baseline — the query words appear in the source text |
| Threshold factual | 5 | Specific numbers — ACOS targets, percentages, thresholds |
| Action-oriented | 6 | "What should I do when X happens" — common user pattern |
| Procedural | 4 | Multi-step processes — hardest for retrieval |
| Diagnostic | 3 | "Why is X happening" — reasoning required |
| Compliance | 3 | Policy lookup — zero tolerance for wrong answers |
| Paraphrased | 1 | Different words, same meaning as the source |

### Vocabulary mismatch — the core retrieval challenge

Each query is pre-labelled with its expected failure mode:

**Vocabulary overlap** (22 queries): the words in the query appear in
the source chunk. Example: "What ACOS threshold triggers bid reduction?"
— the chunk contains the phrase "ACOS" and "bid reduction" directly.
These queries are easier for embedding-based retrieval.

**Vocabulary mismatch** (16 queries): the query uses different words than
the source. Example: "When a product has been on Amazon for over a year,
what should the advertising cost target be?" — the chunk says "Mature phase
(365 days and above): Target ACOS 15 to 25%." The query says "over a year"
and "advertising cost target." Neither phrase appears in the chunk.

This distinction matters because the fix for each failure mode is different.
Vocabulary mismatch requires either adding context to the chunk before
embedding (so the embedding carries more meaning) or expanding the query
before retrieval.

---

## The failure attribution framework

Before running experiments, I established a four-stage failure attribution
framework. When a query fails, which stage caused it?

```
Stage 1 — Retrieval:   did the right chunks come back at all?
Stage 2 — Ranking:     are the best chunks ranked first?
Stage 3 — Context:     is the context window well-formed?
Stage 4 — Generation:  did the LLM use the context correctly?
```

This matters because the fix for a Stage 1 failure (the right chunk is not
in the index at all) is completely different from a Stage 2 failure (the
right chunk is in the index but ranked 8th instead of 1st). Conflating them
produces experiments that solve the wrong problem.

For the original four failing queries, investigation confirmed both were
Stage 2 failures — the right chunk was in the candidate pool but ranked too
low. The bi-encoder found it; the reranker demoted it.

---

## Retrieval experiments

### What I was trying to improve

The core hypothesis: the bi-encoder embeds a short sentence without any
surrounding context. "Common causes ranked by frequency" — the ground truth
chunk for the Buy Box query — has almost no semantic overlap with "What steps
should be taken when a product loses Buy Box?" The embedding similarity is low
because the chunk text doesn't carry enough information about what section
it belongs to or what document it came from.

**The proposed fix:** Before embedding a chunk, prepend its document title
and section heading to the chunk text. Instead of embedding:

> "Common causes ranked by frequency: a) Price undercut by competitor..."

Embed:

> "STANDARD OPERATING PROCEDURE — Buy Box Loss and Recovery
> IMMEDIATE RESPONSE — within 2 hours of detection
> Common causes ranked by frequency: a) Price undercut by competitor..."

This is called **contextual enrichment** — giving the embedding model
structural context so the resulting vector carries information about where
the chunk came from, not just what it says.

### Three configurations tested

| Configuration | What gets embedded | What the reranker scores |
|---|---|---|
| Baseline | Raw chunk text only | Raw chunk text only |
| Embed-enriched | Title + section + chunk text | Raw chunk text only |
| Fully-enriched | Title + section + chunk text | Title + section + chunk text |

The reranker is the cross-encoder that re-scores the top 15 candidates
after the bi-encoder retrieves them. The question is whether giving it
the same enriched context — not just the raw sentence — improves its
ability to rank the right chunk first.

### Results on the 38-query benchmark

| Metric | Baseline | Embed-enriched | Fully-enriched |
|---|---|---|---|
| Dense retrieval R@5 | 0.763 | **0.868** | **0.868** |
| Dense retrieval MRR | 0.597 | **0.740** | **0.740** |
| Reranker MRR | 0.630 | 0.661 | **0.664** |
| Reranker nDCG@5 | 0.629 | 0.662 | **0.702** |

**What this means:**

Contextual enrichment at the embedding stage produces a 24% relative
improvement in dense retrieval MRR (0.597 → 0.740). The right chunk is
now in the top 5 for 87% of queries compared to 76% before.

The reranker also improves with enrichment, but more modestly (+5% MRR).
The reason is that enrichment helps some query types significantly while
slightly hurting others — they average out at the aggregate level.

### Where enrichment helps most vs least

| Query type | Baseline reranker MRR | Fully-enriched reranker MRR | Change |
|---|---|---|---|
| Procedural | 0.309 | **0.625** | +102% |
| Diagnostic | 0.556 | **0.778** | +40% |
| Threshold factual | 0.833 | **0.917** | +10% |
| Direct factual | **0.783** | 0.724 | -8% |
| Action-oriented | **0.618** | 0.582 | -6% |
| Compliance | 0.137 | 0.135 | ~0% |
| Paraphrased | 0.000 | 0.000 | 0% |

Enrichment helps most for **procedural queries** (+102%) — questions like
"What steps should be taken when X happens?" where the chunk text is an
action list that doesn't contain the word "steps." The document title and
section heading provide the context the embedding was missing.

Enrichment slightly hurts **direct factual queries** (-8%) — questions
where the query vocabulary already overlaps with the chunk text. Adding
context introduces noise for these queries because the embedding is already
finding the right chunk.

This nuance is why aggregate metrics are insufficient. A single MRR number
hides the fact that enrichment makes the system significantly better for
one query type and slightly worse for another.

---

## The reranker: when it helps and when it hurts

The cross-encoder reranker was trained on the MS MARCO dataset — web search
queries matched to web documents. Amazon agency SOPs, compliance documents,
and onboarding transcripts use different vocabulary patterns than web search.

**Specific failure observed:** The Buy Box recovery SOP chunk that correctly
answers "What steps should be taken when a product loses Buy Box?" scored
-7.13 on the reranker in baseline configuration (correctly answered by the
bi-encoder at rank 1). The reranker demoted it to rank 8.

After adding structural enrichment to the reranker input, the same chunk
scored +5.42 — a 12-point swing. The section heading "IMMEDIATE RESPONSE —
within 2 hours of detection" gave the reranker the context it needed to
recognise the chunk as relevant.

**Client transcript queries:** Onboarding transcripts are conversational.
"We are not a budget brand. Do not compete on price." doesn't match web
search vocabulary patterns at all. The reranker systematically demotes
transcript chunks even when the bi-encoder correctly ranks them first.

**Decision:** The reranker is disabled for client-specific queries.
The bi-encoder alone produces better results for transcript-heavy queries.
The reranker remains active for global knowledge base queries (SOPs,
compliance, frameworks) where it demonstrably improves ranking.

---

## What changed because of measurement

| What the measurement showed | What changed |
|---|---|
| Sparse retrieval (BM25) MRR = 0.063 — worse than random | Sparse retrieval rejected; hybrid approach deprioritised |
| Increasing candidate pool from 15 to 50 had no effect on MRR | Pool size confirmed not the bottleneck; focus moved to ranking quality |
| Buy Box and prohibited terms queries were ranking failures, not retrieval failures | Fixed ranking signal rather than adding more documents |
| Reranker demoted the correct chunk for client transcript queries | Reranker disabled for client namespace queries |
| ACOS SOP produced 1 chunk instead of ~10 (wrong chunking strategy) | Structure detector thresholds calibrated from measured signal values across 6 documents |
| Compliance PDF produced 4 large chunks instead of 9 semantic units | Root cause identified (PDF extraction strips blank lines); post-extraction paragraph reconstruction added |
| Paraphrased queries: MRR = 0.000 across all configurations | Documented as known limitation; not fixable with current embedding approach |

---

## What is still failing and why

### 1. Compliance document chunking — MRR 0.268

The compliance rules document has 4 chunks instead of the 9 it should have.
Chunk 2 contains three separate categories — guarantee language, pricing
restrictions, and conditional terms — as one 722-character block.

**Root cause:** The PDF extraction library (pdfplumber) strips blank lines
when extracting text. The blank lines between term categories are what the
paragraph splitter uses to identify boundaries. Without them, three categories
merge into one chunk.

**Why it matters:** A query for "what guarantee language is prohibited?"
retrieves the 722-character chunk that also contains pricing terms and
conditional terms. Retrieval is less precise — the right content is there
but buried in a larger block.

**Fix identified:** Replace pdfplumber with the `unstructured` library, which
uses the PDF's layout coordinates rather than plain text extraction to detect
paragraph boundaries. It would correctly identify where one term category ends
and the next begins regardless of blank lines.

**Status:** Deferred. The information is still retrievable, just at reduced
precision.

### 2. Paraphrased queries — MRR 0.000

The query "When a product has been on Amazon for over a year, what should
the advertising cost target be?" fails completely. The source chunk says
"Mature phase (365 days and above): Target ACOS 15 to 25%."

"Over a year" and "365 days and above" mean the same thing. The embedding
model doesn't know this. It measures word similarity, not semantic equivalence
of temporal expressions.

**Why it matters:** Real users paraphrase. They won't always use the exact
vocabulary in the source documents.

**Fix identified:** Query expansion — before retrieval, use the LLM to
rewrite the query in multiple ways ("over a year" → "365 days", "12 months",
"mature phase"). Retrieve for all variants, merge the results. This is a
standard technique for vocabulary mismatch.

**Status:** Documented. Implementation scheduled for next iteration.

### 3. Anomaly investigation framework — regression under enrichment

The anomaly investigation framework document shows worse retrieval with
enrichment (MRR 0.704 → 0.492) compared to baseline.

**Root cause:** Adding the same document title to every chunk from this
document makes all 15 chunks more similar to each other in embedding space.
The bi-encoder finds the right document easily but loses the ability to
discriminate between chunks within the same document.

**Why it matters:** This is a real trade-off. Enrichment helps cross-document
retrieval (finding the right document) but can hurt within-document retrieval
(finding the right section).

**Fix identified:** Apply enrichment selectively — only for query types where
it demonstrably helps (procedural, diagnostic). Apply baseline embedding for
direct factual queries where vocabulary already matches.

**Status:** Documented. Enrichment accepted as the overall best configuration
despite this regression — the aggregate improvement outweighs this specific
document's regression.

---

## The benchmark audit

The 38-query benchmark was audited twice:

**Initial construction:** Ground truth assigned by reading the source document
section by section and identifying which chunk directly answers each query.

**Independent audit:** After completing initial construction, each ground truth
entry was re-verified by reading the actual chunk text (not the source document)
and confirming it contained the answer. Chunk IDs and section memberships were
checked against the raw extracted content.

Three ground truth corrections made during the audit:
- One query had an extra chunk listed as relevant that was contextual but
  not answer-bearing — removed
- One query had the wrong chunk listed as primary (the answer was in the
  adjacent chunk, not the one originally labelled)
- One query had the wrong section title — the content was in a different
  section than initially recorded

No queries were removed, added, or modified to make retrieval metrics look
better. The corrections made metrics slightly harder to achieve, not easier.

---

## Evaluation infrastructure

All evaluation code and results are in the repository:

```
data/eval_set_38_audited.json       — 38-query benchmark with ground truth
scripts/eval_38_query_experiment.py — Runs all three configurations and
                                      produces stratified results
data/eval_38_results.json           — Full per-query results including
                                      dense rank, reranker rank, and scores
                                      for all 38 queries × 3 configurations
```

To reproduce the benchmark results:

```bash
python scripts/eval_38_query_experiment.py
```

---

## The honest summary

Dense retrieval MRR improved from 0.597 to 0.740 through contextual
enrichment — a 24% relative improvement. The right chunks are in the
top 5 results for 87% of queries.

The reranker improves ranking precision for procedural and diagnostic
queries significantly (+102% and +40% MRR respectively) but slightly hurts
direct factual queries. It is disabled for client-specific queries where it
demonstrably hurts performance.

Three failure modes remain unresolved: compliance document chunking,
paraphrased vocabulary, and anomaly investigation regression. All three
have specific root causes identified and fixes designed. None are blocking
for current use cases.

The system is not perfect. It is measured, understood, and honest about
where it fails. That is a stronger foundation for production than a system
that appears to work but has never been tested to failure.
