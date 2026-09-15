"""
scripts/eval_reranker_enrichment.py

Controlled experiment: does enriching the reranker input resolve
the ranking failure observed in the contextual enrichment ablation?

Experiment design:
    A  — Baseline
         embedding_text = chunk_text
         reranker_text  = chunk_text

    D1 — Contextualized dense retrieval only (already measured)
         embedding_text = document_title + section_title + chunk_text
         reranker_text  = chunk_text

    D2 — Contextualized dense retrieval + reranking
         embedding_text = document_title + section_title + chunk_text
         reranker_text  = document_title + section_title + chunk_text

Critical comparison: D1 → D2
If D2 improves reranker MRR over D1, the reranker was being deprived
of structural context. The cross-encoder model is not fundamentally
unsuitable — it lacked the information to make correct judgements.

If D2 does NOT improve over D1, missing context was not sufficient to
explain the reranker failure. Further investigation required.

Source truth is preserved throughout:
    chunk_text      — original source content, unchanged
    embedding_text  — contextual representation for dense retrieval
    reranker_text   — contextual representation for cross-encoder

chunk_text is never overwritten and is used for LLM delivery.

Run:
    python scripts/eval_reranker_enrichment.py
"""

import json
import math
import re
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

# ── Eval set ──────────────────────────────────────────────────────────

EVAL_SET = [
    {
        "id": "Q1",
        "question": "What ACOS threshold triggers bid reduction on keywords?",
        "ground_truth_id": "sop_acos_management.txt_sentence_window_7",
    },
    {
        "id": "Q2",
        "question": "What terms are absolutely prohibited in Amazon listing copy?",
        "ground_truth_id": "compliance_restricted_claims.txt_sentence_window_1",
    },
    {
        "id": "Q3",
        "question": "What format should Amazon listing titles follow?",
        "ground_truth_id": "sop_listing_optimisation.txt_sentence_window_7",
    },
    {
        "id": "Q4",
        "question": "What steps should be taken when a product loses Buy Box?",
        "ground_truth_id": "sop_buy_box_recovery.txt_sentence_window_4",
    },
]

# ── Section detector ──────────────────────────────────────────────────
# Identical to eval_contextual_enrichment.py — validated against full corpus

def is_section_header(line: str) -> bool:
    line = line.strip()
    if not line:
        return False
    if '|' in line:
        return False
    if line.startswith('[') and ']' in line[:10]:
        return False
    if line.endswith(':'):
        return False
    if line.endswith(('.', '?', '!')):
        return False
    if re.match(r'^Step\s+\d+\s*—', line):
        return False
    if re.match(r'^[a-z]\)', line):
        return False
    if line != line.lstrip():
        return False
    if len(line) > 100:
        return False
    # Bullet items starting with dash
    if line.startswith('- '):
        return False
    # Conditional term definitions: "term — explanation" pattern
    # Distinguished from section headers by containing lowercase after the dash
    if re.match(r'^[a-z].*—.*[a-z]', line):
        return False
    # Lines starting with "Example:"
    if line.startswith('Example:'):
        return False
    # Lines ending with comma — mid-sentence wrap
    if line.endswith(','):
        return False
    return True


def extract_document_structure(text: str) -> dict:
    lines = text.split('\n')
    document_title = ""
    for line in lines:
        stripped = line.strip()
        if stripped and '|' not in stripped:
            document_title = stripped
            break

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


def get_chunk_context(chunk: dict, raw_dir: Path) -> dict:
    """
    Returns document_title and section_title for a chunk.
    Verified against source document structure — not inferred.
    """
    source_file = chunk.get("source_file", "")
    chunk_text = chunk.get("text", "")
    doc_path = raw_dir / source_file

    if not doc_path.exists():
        return {"document_title": "", "section_title": ""}

    text = doc_path.read_text()
    structure = extract_document_structure(text)
    lines = text.split('\n')

    search_text = chunk_text[:50].strip()
    target_line = None
    for i, line in enumerate(lines):
        if search_text in line or (len(search_text) > 20 and search_text[:20] in line):
            target_line = i
            break

    if target_line is None:
        return {"document_title": structure["document_title"], "section_title": ""}

    section_title = structure["line_to_section"].get(target_line, "")
    return {
        "document_title": structure["document_title"],
        "section_title": section_title,
    }


def build_enriched_text(chunk: dict, ctx: dict) -> str:
    """
    Builds representation D: document_title + section_title + chunk_text.
    Omits empty components gracefully.
    """
    parts = [
        p for p in [
            ctx["document_title"],
            ctx["section_title"],
            chunk["text"],
        ]
        if p
    ]
    return "\n".join(parts)


# ── Metrics ───────────────────────────────────────────────────────────

def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def compute_recall_at_k(ranks: list, k: int) -> float:
    return sum(1 for r in ranks if r is not None and r <= k) / len(ranks)


def compute_mrr(ranks: list) -> float:
    return sum(1/r if r else 0 for r in ranks) / len(ranks)


def compute_ndcg_at_k(ranks: list, k: int) -> float:
    vals = [1/math.log2(r+1) if r and r <= k else 0 for r in ranks]
    return sum(vals) / len(vals)


def embed_texts(texts: list[str]) -> list[list[float]]:
    embeddings = []
    for i in range(0, len(texts), 100):
        batch = texts[i:i+100]
        response = client.embeddings.create(
            model=settings.openai_embedding_model,
            input=batch,
        )
        embeddings.extend([r.embedding for r in response.data])
    return embeddings


def extract_sections(text: str) -> list[tuple[int, str]]:
    lines = text.split('\n')
    sections = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not is_section_header(stripped):
            continue
        # True section headers are always preceded by a blank line
        preceded_by_blank = (i == 0) or (lines[i-1].strip() == '')
        if not preceded_by_blank:
            continue
        sections.append((i, stripped))
    return sections

# ── Single configuration runner ───────────────────────────────────────

def run_configuration(
    config_name: str,
    global_chunks: list[dict],
    chunk_contexts: dict,
    embedding_mode: str,
    reranker_mode: str,
) -> dict:
    """
    Runs one experimental configuration.

    embedding_mode: "chunk" | "enriched"
    reranker_mode:  "chunk" | "enriched"

    These are the only two variables. Everything else is held constant.
    """
    print(f"\n{'─' * 55}")
    print(f"Configuration {config_name}")
    print(f"  embedding: {embedding_mode}  |  reranker: {reranker_mode}")
    print(f"{'─' * 55}")

    # Build embedding texts
    embedding_texts = []
    for chunk in global_chunks:
        ctx = chunk_contexts[chunk["chunk_id"]]
        if embedding_mode == "enriched":
            embedding_texts.append(build_enriched_text(chunk, ctx))
        else:
            embedding_texts.append(chunk["text"])

    # Embed all chunks
    print(f"  Embedding {len(embedding_texts)} chunks ({embedding_mode} mode)...")
    chunk_embeddings = embed_texts(embedding_texts)

    # Load reranker
    reranker = _load_reranker()

    # Evaluate each query
    per_query = []
    dense_ranks = []
    reranker_ranks = []

    for item in EVAL_SET:
        question = item["question"]
        target_id = item["ground_truth_id"]

        # Embed query — always raw question, never enriched
        q_emb = embed_texts([question])[0]

        # Dense retrieval
        scores = [
            cosine_similarity(q_emb, emb)
            for emb in chunk_embeddings
        ]
        scored = sorted(
            zip(scores, global_chunks, chunk_embeddings),
            key=lambda x: x[0],
            reverse=True,
        )

        dense_ranked_ids = [c["chunk_id"] for _, c, _ in scored]
        dense_rank = (
            dense_ranked_ids.index(target_id) + 1
            if target_id in dense_ranked_ids else None
        )
        dense_score = scores[
            [c["chunk_id"] for c in global_chunks].index(target_id)
        ] if target_id in [c["chunk_id"] for c in global_chunks] else None

        dense_ranks.append(dense_rank)

        # Reranker — top 15 candidates
        candidates = [(score, chunk) for score, chunk, _ in scored[:15]]

        # Build reranker pairs based on reranker_mode
        pairs = []
        for _, chunk in candidates:
            ctx = chunk_contexts[chunk["chunk_id"]]
            if reranker_mode == "enriched":
                reranker_text = build_enriched_text(chunk, ctx)
            else:
                reranker_text = chunk["text"]
            pairs.append((question, reranker_text))

        reranker_scores = reranker.predict(pairs)
        reranker_scored = sorted(
            zip(reranker_scores, [c for _, c in candidates]),
            key=lambda x: x[0],
            reverse=True,
        )

        reranker_ranked_ids = [c["chunk_id"] for _, c in reranker_scored]
        reranker_rank = (
            reranker_ranked_ids.index(target_id) + 1
            if target_id in reranker_ranked_ids else None
        )

        # Reranker score for ground truth chunk
        target_reranker_score = None
        for rs, chunk in zip(reranker_scores, [c for _, c in candidates]):
            if chunk["chunk_id"] == target_id:
                target_reranker_score = float(rs)
                break

        dense_ranks_current = dense_rank
        reranker_ranks.append(reranker_rank)

        ds = f"{dense_score:.4f}" if dense_score is not None else "N/A"
        rs = f"{target_reranker_score:.4f}" if target_reranker_score is not None else "N/A"
        print(f"  {item['id']}: dense={dense_rank}  reranker={reranker_rank}  "
              f"[dense_score={ds}  reranker_score={rs}]")

        per_query.append({
            "id": item["id"],
            "question": question,
            "ground_truth_id": target_id,
            "dense_rank": dense_rank,
            "dense_score": round(dense_score, 4) if dense_score else None,
            "reranker_rank": reranker_rank,
            "reranker_score": round(target_reranker_score, 4)
                if target_reranker_score else None,
        })

    metrics = {
        "dense_recall_at_5":  compute_recall_at_k(dense_ranks, 5),
        "dense_recall_at_10": compute_recall_at_k(dense_ranks, 10),
        "dense_recall_at_15": compute_recall_at_k(dense_ranks, 15),
        "dense_mrr":          compute_mrr(dense_ranks),
        "reranker_mrr":       compute_mrr(reranker_ranks),
        "ndcg_at_5":          compute_ndcg_at_k(reranker_ranks, 5),
        "ndcg_at_10":         compute_ndcg_at_k(reranker_ranks, 10),
    }

    print(f"\n  Dense   — R@5: {metrics['dense_recall_at_5']:.2f} | "
          f"R@10: {metrics['dense_recall_at_10']:.2f} | "
          f"MRR: {metrics['dense_mrr']:.3f}")
    print(f"  Reranker — MRR: {metrics['reranker_mrr']:.3f} | "
          f"nDCG@5: {metrics['ndcg_at_5']:.3f} | "
          f"nDCG@10: {metrics['ndcg_at_10']:.3f}")

    return {
        "config": config_name,
        "embedding_mode": embedding_mode,
        "reranker_mode": reranker_mode,
        "metrics": metrics,
        "per_query": per_query,
    }


# ── Main ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Load chunks
    chunks_path = settings.chunks_dir / "sentence_window.json"
    with open(chunks_path) as f:
        all_chunks = json.load(f)

    global_chunks = [c for c in all_chunks if c.get("client") == "global"]
    raw_dir = settings.raw_dir

    print(f"Global chunks: {len(global_chunks)}")

    # Pre-compute section context for all chunks once
    print("Extracting section context for all chunks...")
    chunk_contexts = {}
    for chunk in global_chunks:
        chunk_contexts[chunk["chunk_id"]] = get_chunk_context(chunk, raw_dir)

    # Verify ground truth chunk context
    gt_chunk_id = "sop_buy_box_recovery.txt_sentence_window_4"
    gt_ctx = chunk_contexts.get(gt_chunk_id, {})
    print(f"\nVerified context for Q4 ground truth chunk:")
    print(f"  document_title : {gt_ctx.get('document_title', 'NOT FOUND')}")
    print(f"  section_title  : {gt_ctx.get('section_title', 'NOT FOUND')}")
    print(f"  chunk_text     : {next((c['text'][:80] for c in global_chunks if c['chunk_id'] == gt_chunk_id), 'NOT FOUND')}")

    # Run three configurations
    configurations = [
        ("A",  "chunk",    "chunk"),     # baseline
        ("D1", "enriched", "chunk"),     # dense enriched, reranker raw
        ("D2", "enriched", "enriched"),  # both enriched
    ]

    results = {}
    for config_name, emb_mode, rer_mode in configurations:
        results[config_name] = run_configuration(
            config_name=config_name,
            global_chunks=global_chunks,
            chunk_contexts=chunk_contexts,
            embedding_mode=emb_mode,
            reranker_mode=rer_mode,
        )

    # Summary table
    print(f"\n{'=' * 70}")
    print("EXPERIMENT SUMMARY: A vs D1 vs D2")
    print(f"{'=' * 70}")
    print(f"{'Config':<8} {'D.MRR':<10} {'R@5':<8} {'R@10':<8} "
          f"{'Re.MRR':<10} {'nDCG@5':<10} {'nDCG@10'}")
    print(f"{'─' * 70}")
    for name in ["A", "D1", "D2"]:
        m = results[name]["metrics"]
        print(
            f"{name:<8} "
            f"{m['dense_mrr']:<10.3f} "
            f"{m['dense_recall_at_5']:<8.2f} "
            f"{m['dense_recall_at_10']:<8.2f} "
            f"{m['reranker_mrr']:<10.3f} "
            f"{m['ndcg_at_5']:<10.3f} "
            f"{m['ndcg_at_10']:.3f}"
        )

    print(f"\nPer-query dense ranks:")
    print(f"{'Config':<8} {'Q1':<8} {'Q2':<8} {'Q3':<8} {'Q4'}")
    print(f"{'─' * 35}")
    for name in ["A", "D1", "D2"]:
        ranks = [
            str(q["dense_rank"]) if q["dense_rank"] else "None"
            for q in results[name]["per_query"]
        ]
        print(f"{name:<8} {ranks[0]:<8} {ranks[1]:<8} {ranks[2]:<8} {ranks[3]}")

    print(f"\nPer-query reranker ranks:")
    print(f"{'Config':<8} {'Q1':<8} {'Q2':<8} {'Q3':<8} {'Q4'}")
    print(f"{'─' * 35}")
    for name in ["A", "D1", "D2"]:
        ranks = [
            str(q["reranker_rank"]) if q["reranker_rank"] else "None"
            for q in results[name]["per_query"]
        ]
        print(f"{name:<8} {ranks[0]:<8} {ranks[1]:<8} {ranks[2]:<8} {ranks[3]}")

    # Q4 focused analysis — the primary failure case
    print(f"\nQ4 (Buy Box) — dense → reranker rank progression:")
    for name in ["A", "D1", "D2"]:
        q4 = next(q for q in results[name]["per_query"] if q["id"] == "Q4")
        print(f"  {name}: dense={q4['dense_rank']}  →  "
              f"reranker={q4['reranker_rank']}  "
              f"[dense_score={q4['dense_score']}  "
              f"reranker_score={q4['reranker_score']}]")

    # Q2 focused analysis — the other failure case
    print(f"\nQ2 (Prohibited terms) — dense → reranker rank progression:")
    for name in ["A", "D1", "D2"]:
        q2 = next(q for q in results[name]["per_query"] if q["id"] == "Q2")
        print(f"  {name}: dense={q2['dense_rank']}  →  "
              f"reranker={q2['reranker_rank']}  "
              f"[dense_score={q2['dense_score']}  "
              f"reranker_score={q2['reranker_score']}]")

    # Save results
    out_path = Path("data/eval_reranker_enrichment.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results → {out_path}")
