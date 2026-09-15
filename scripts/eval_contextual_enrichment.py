"""
scripts/eval_contextual_enrichment.py

Ablation experiment: does contextual chunk enrichment improve retrieval?

Hypothesis:
    The ground truth chunk for "What steps should be taken when a product
    loses Buy Box?" ranks poorly because its text lacks the vocabulary
    present in the query. The chunk "Common causes ranked by frequency..."
    does not contain "Buy Box", "steps", or "loses".

    Adding parent context (document title, section title, or both) to the
    embedding representation may bridge this vocabulary gap and improve
    dense retrieval rank and/or cross-encoder reranker rank.

Four representations tested:
    A — chunk text only (current baseline)
    B — document title + chunk text
    C — section title + chunk text
    D — document title + section title + chunk text

Experiment design:
    - Source chunk text is NEVER modified
    - embedding_text changes per ablation
    - reranker_text is identical to embedding_text for this experiment
    - Dense recall and reranker rank recorded separately per query
    - Results written to data/eval_contextual_enrichment.json

Run:
    python scripts/eval_contextual_enrichment.py
"""

import json
import re
import math
from pathlib import Path
from dotenv import load_dotenv

import sys
from pathlib import Path


ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv()

from openai import OpenAI
from config.settings import get_settings
from knowledge.retriever import _load_reranker

settings = get_settings()
client = OpenAI(api_key=settings.openai_api_key)

# ── Ground truth ──────────────────────────────────────────────────────

EVAL_SET = [
    {
        "question": "What ACOS threshold triggers bid reduction on keywords?",
        "ground_truth_id": "sop_acos_management.txt_sentence_window_7",
        "namespace": "global",
    },
    {
        "question": "What terms are absolutely prohibited in Amazon listing copy?",
        "ground_truth_id": "compliance_restricted_claims.txt_sentence_window_1",
        "namespace": "global",
    },
    {
        "question": "What format should Amazon listing titles follow?",
        "ground_truth_id": "sop_listing_optimisation.txt_sentence_window_7",
        "namespace": "global",
    },
    {
        "question": "What steps should be taken when a product loses Buy Box?",
        "ground_truth_id": "sop_buy_box_recovery.txt_sentence_window_4",
        "namespace": "global",
    },
]


# ── Section detector ──────────────────────────────────────────────────

def is_section_header(line: str) -> bool:
    """
    Identifies section headers in KIRA SOP and compliance documents.

    Validated against full corpus — no false positives observed.
    Transcripts return no section headers (correct — they have none).

    Rules derived from corpus structure inventory, not formatting assumptions.
    """
    line = line.strip()
    if not line:
        return False
    # Metadata lines contain pipe separators
    if '|' in line:
        return False
    # Transcript timestamps: [HH:MM]
    if line.startswith('[') and ']' in line[:10]:
        return False
    # Sub-list category headers end with colon
    if line.endswith(':'):
        return False
    # Complete sentences end with terminal punctuation
    if line.endswith(('.', '?', '!')):
        return False
    # Step headers within sections (not section boundaries)
    if re.match(r'^Step\s+\d+\s*—', line):
        return False
    # List items
    if re.match(r'^[a-z]\)', line):
        return False
    # Indented content
    if line != line.lstrip():
        return False
    # Too long to be a header (longest observed: 58 chars)
    if len(line) > 100:
        return False
    return True


def extract_document_structure(text: str) -> dict:
    """
    Extracts document title and maps each line to its parent section.

    Returns:
        {
            "document_title": str,
            "line_to_section": {line_number: section_title}
        }

    The document title is the first non-empty, non-metadata line.
    Section titles are assigned to all lines that follow them until
    the next section header.
    """
    lines = text.split('\n')

    # Document title: first non-empty line
    document_title = ""
    for line in lines:
        stripped = line.strip()
        if stripped and '|' not in stripped:
            document_title = stripped
            break

    # Map each line to its current section
    line_to_section = {}
    current_section = ""

    for i, line in enumerate(lines):
        stripped = line.strip()
        if is_section_header(stripped):
            current_section = stripped
        line_to_section[i] = current_section

    return {
        "document_title": document_title,
        "line_to_section": line_to_section,
    }


def get_chunk_section(chunk: dict, raw_dir: Path) -> dict:
    """
    Finds the section title for a given chunk by locating its text
    in the source document and checking which section header precedes it.

    Returns:
        {
            "document_title": str,
            "section_title": str,
        }
    """
    source_file = chunk.get("source_file", "")
    chunk_text = chunk.get("text", "")

    doc_path = raw_dir / source_file
    if not doc_path.exists():
        return {"document_title": "", "section_title": ""}

    text = doc_path.read_text()
    structure = extract_document_structure(text)

    lines = text.split('\n')

    # Find the line in the document that contains the start of this chunk
    # We search for the first ~50 chars of the chunk text
    search_text = chunk_text[:50].strip()
    target_line = None

    for i, line in enumerate(lines):
        if search_text in line or (len(search_text) > 20 and search_text[:20] in line):
            target_line = i
            break

    if target_line is None:
        return {
            "document_title": structure["document_title"],
            "section_title": "",
        }

    section_title = structure["line_to_section"].get(target_line, "")

    return {
        "document_title": structure["document_title"],
        "section_title": section_title,
    }


# ── Embedding ─────────────────────────────────────────────────────────

def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embeds a list of texts using OpenAI embeddings.
    Batches to stay within API limits.
    """
    embeddings = []
    batch_size = 100

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = client.embeddings.create(
            model=settings.openai_embedding_model,
            input=batch,
        )
        embeddings.extend([r.embedding for r in response.data])

    return embeddings


# ── Metrics ───────────────────────────────────────────────────────────

def compute_recall_at_k(ranks: list[int | None], k: int) -> float:
    """Fraction of queries where ground truth appears in top k."""
    hits = sum(1 for r in ranks if r is not None and r <= k)
    return hits / len(ranks) if ranks else 0.0


def compute_mrr(ranks: list[int | None]) -> float:
    """Mean Reciprocal Rank."""
    rr = [1/r if r is not None else 0.0 for r in ranks]
    return sum(rr) / len(rr) if rr else 0.0


def compute_ndcg_at_k(ranks: list[int | None], k: int) -> float:
    """
    Normalised Discounted Cumulative Gain at k.

    For binary relevance (one ground truth chunk per query):
    DCG@k = 1/log2(rank+1) if ground truth in top k, else 0
    IDCG@k = 1/log2(2) = 1.0 (perfect ranking puts it at position 1)
    nDCG@k = DCG@k / IDCG@k = DCG@k
    """
    ndcg_values = []
    for rank in ranks:
        if rank is not None and rank <= k:
            ndcg_values.append(1.0 / math.log2(rank + 1))
        else:
            ndcg_values.append(0.0)
    return sum(ndcg_values) / len(ndcg_values) if ndcg_values else 0.0


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ── Experiment ────────────────────────────────────────────────────────

def run_ablation(
    chunks: list[dict],
    representation: str,
    raw_dir: Path,
) -> dict:
    """
    Runs one ablation — embeds all chunks with the given representation,
    then evaluates dense retrieval and reranker ranking for all queries.

    representation options:
        "A" — chunk text only
        "B" — document title + chunk text
        "C" — section title + chunk text
        "D" — document title + section title + chunk text
    """
    print(f"\n{'─' * 55}")
    print(f"Representation {representation}")
    print(f"{'─' * 55}")

    # Global chunks only — matching our eval namespace
    global_chunks = [c for c in chunks if c.get("client") == "global"]
    print(f"Chunks: {len(global_chunks)}")

    # Build embedding text per chunk
    print("Extracting section structure...")
    chunk_structures = {}
    for chunk in global_chunks:
        chunk_id = chunk["chunk_id"]
        structure = get_chunk_section(chunk, raw_dir)
        chunk_structures[chunk_id] = structure

    # Build embedding texts
    embedding_texts = []
    for chunk in global_chunks:
        structure = chunk_structures[chunk["chunk_id"]]
        doc_title = structure["document_title"]
        section_title = structure["section_title"]
        chunk_text = chunk["text"]

        if representation == "A":
            emb_text = chunk_text
        elif representation == "B":
            emb_text = f"{doc_title}\n{chunk_text}" if doc_title else chunk_text
        elif representation == "C":
            emb_text = f"{section_title}\n{chunk_text}" if section_title else chunk_text
        elif representation == "D":
            parts = [p for p in [doc_title, section_title, chunk_text] if p]
            emb_text = "\n".join(parts)
        else:
            emb_text = chunk_text

        embedding_texts.append(emb_text)

    # Embed all chunks
    print(f"Embedding {len(embedding_texts)} chunks...")
    chunk_embeddings = embed_texts(embedding_texts)

    # Load reranker
    reranker = _load_reranker()

    # Evaluate each query
    query_results = []
    dense_ranks = []
    reranker_ranks = []

    for item in EVAL_SET:
        question = item["question"]
        target_id = item["ground_truth_id"]

        # Embed query
        q_emb = embed_texts([question])[0]

        # Dense retrieval — score all global chunks
        scores = [
            cosine_similarity(q_emb, chunk_emb)
            for chunk_emb in chunk_embeddings
        ]

        # Sort by score descending
        scored = sorted(
            zip(scores, global_chunks),
            key=lambda x: x[0],
            reverse=True,
        )

        # Dense rank
        dense_ranked_ids = [c["chunk_id"] for _, c in scored]
        dense_rank = (
            dense_ranked_ids.index(target_id) + 1
            if target_id in dense_ranked_ids else None
        )
        dense_ranks.append(dense_rank)

        # Reranker — score top 15 candidates
        candidates = [c for _, c in scored[:15]]
        pairs = [(question, c["text"]) for c in candidates]
        reranker_scores = reranker.predict(pairs)

        reranker_scored = sorted(
            zip(reranker_scores, candidates),
            key=lambda x: x[0],
            reverse=True,
        )
        reranker_ranked_ids = [c["chunk_id"] for _, c in reranker_scored]
        reranker_rank = (
            reranker_ranked_ids.index(target_id) + 1
            if target_id in reranker_ranked_ids else None
        )
        reranker_ranks.append(reranker_rank)

        print(f"  Q: {question[:55]}...")
        print(f"     Dense rank: {dense_rank}  |  Reranker rank: {reranker_rank}")

        query_results.append({
            "question": question,
            "ground_truth_id": target_id,
            "embedding_text_preview": embedding_texts[
                global_chunks.index(
                    next(c for c in global_chunks if c["chunk_id"] == target_id)
                )
            ][:150] if target_id in [c["chunk_id"] for c in global_chunks] else "",
            "dense_rank": dense_rank,
            "reranker_rank": reranker_rank,
        })

    # Aggregate metrics
    metrics = {
        "dense_recall_at_5":  compute_recall_at_k(dense_ranks, 5),
        "dense_recall_at_10": compute_recall_at_k(dense_ranks, 10),
        "dense_recall_at_15": compute_recall_at_k(dense_ranks, 15),
        "dense_mrr":          compute_mrr(dense_ranks),
        "reranker_mrr":       compute_mrr(reranker_ranks),
        "ndcg_at_5":          compute_ndcg_at_k(reranker_ranks, 5),
        "ndcg_at_10":         compute_ndcg_at_k(reranker_ranks, 10),
    }

    print(f"\n  Dense  — Recall@5: {metrics['dense_recall_at_5']:.2f} | "
          f"Recall@10: {metrics['dense_recall_at_10']:.2f} | "
          f"MRR: {metrics['dense_mrr']:.3f}")
    print(f"  Reranker — MRR: {metrics['reranker_mrr']:.3f} | "
          f"nDCG@5: {metrics['ndcg_at_5']:.3f} | "
          f"nDCG@10: {metrics['ndcg_at_10']:.3f}")

    return {
        "representation": representation,
        "metrics": metrics,
        "per_query": query_results,
    }


# ── Main ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Load chunks
    chunks_path = settings.chunks_dir / "sentence_window.json"
    with open(chunks_path) as f:
        chunks = json.load(f)
    print(f"Loaded {len(chunks)} chunks")

    raw_dir = settings.raw_dir

    results = {}
    for rep in ["A", "B", "C", "D"]:
        results[rep] = run_ablation(chunks, rep, raw_dir)

    # Print comparison table
    print(f"\n{'=' * 65}")
    print("ABLATION RESULTS SUMMARY")
    print(f"{'=' * 65}")
    print(f"{'Rep':<6} {'Dense MRR':<12} {'R@5':<8} {'R@10':<8} "
          f"{'Reranker MRR':<14} {'nDCG@5':<10} {'nDCG@10'}")
    print(f"{'─' * 65}")
    for rep in ["A", "B", "C", "D"]:
        m = results[rep]["metrics"]
        print(
            f"{rep:<6} "
            f"{m['dense_mrr']:<12.3f} "
            f"{m['dense_recall_at_5']:<8.2f} "
            f"{m['dense_recall_at_10']:<8.2f} "
            f"{m['reranker_mrr']:<14.3f} "
            f"{m['ndcg_at_5']:<10.3f} "
            f"{m['ndcg_at_10']:.3f}"
        )

    print(f"\nPer-query dense ranks:")
    print(f"{'Rep':<6} {'Q1':<8} {'Q2':<8} {'Q3':<8} {'Q4'}")
    print(f"{'─' * 35}")
    for rep in ["A", "B", "C", "D"]:
        ranks = [
            str(q["dense_rank"]) if q["dense_rank"] else "None"
            for q in results[rep]["per_query"]
        ]
        print(f"{rep:<6} {ranks[0]:<8} {ranks[1]:<8} {ranks[2]:<8} {ranks[3]}")

    print(f"\nPer-query reranker ranks:")
    print(f"{'Rep':<6} {'Q1':<8} {'Q2':<8} {'Q3':<8} {'Q4'}")
    print(f"{'─' * 35}")
    for rep in ["A", "B", "C", "D"]:
        ranks = [
            str(q["reranker_rank"]) if q["reranker_rank"] else "None"
            for q in results[rep]["per_query"]
        ]
        print(f"{rep:<6} {ranks[0]:<8} {ranks[1]:<8} {ranks[2]:<8} {ranks[3]}")

    # Save full results
    out_path = Path("data/eval_contextual_enrichment.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved → {out_path}")