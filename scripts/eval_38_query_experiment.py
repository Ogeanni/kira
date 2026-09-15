"""
scripts/eval_38_query_experiment.py

A/D1/D2 experiment against the audited 38-query benchmark.

Configurations:
    A  — embedding: chunk text        | reranker: chunk text
    D1 — embedding: enriched text     | reranker: chunk text
    D2 — embedding: enriched text     | reranker: enriched text

Enriched text = document_title + section_title + chunk_text

Metrics per configuration:
    Dense  : Recall@5, Recall@10, Recall@15, MRR
    Reranker: MRR, nDCG@5, nDCG@10

Metrics stratified by:
    query_type
    expected_failure_mode
    document_id

Ground truth: primary_chunk_id (strict) and relevant_chunk_ids (lenient)
    Strict  = ground truth only if primary_chunk_id is in top k
    Lenient = ground truth if ANY relevant_chunk_id is in top k

Both are reported. Strict is the primary metric.
"""

import json
import math
import re
from pathlib import Path
from dotenv import load_dotenv
import sys


ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


load_dotenv()

from openai import OpenAI
from config.settings import get_settings
from knowledge.retriever import _load_reranker

settings = get_settings()
client = OpenAI(api_key=settings.openai_api_key)


# ── Section detector ──────────────────────────────────────────────────

def is_section_header(line: str) -> bool:
    line = line.strip()
    if not line: return False
    if '|' in line: return False
    if line.startswith('[') and ']' in line[:10]: return False
    if line.startswith('('): return False
    if line.endswith(':'): return False
    if line.endswith(('.', '?', '!')): return False
    if re.match(r'^Step\s+\d+\s*—', line): return False
    if re.match(r'^[a-z]\)', line): return False
    if line.startswith('- '): return False
    if re.match(r'^[a-z].+—.+[a-z]', line): return False
    if line.startswith('Example:'): return False
    if line.endswith(','): return False
    if line != line.lstrip(): return False
    if len(line) > 100: return False
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
    return {"document_title": document_title, "line_to_section": line_to_section}


def get_chunk_context(chunk: dict, raw_dir: Path) -> dict:
    source_file = chunk.get("source_file", "")
    chunk_text = chunk.get("text", "")[:50].strip()
    doc_path = raw_dir / source_file
    if not doc_path.exists():
        return {"document_title": "", "section_title": ""}
    text = doc_path.read_text()
    structure = extract_document_structure(text)
    lines = text.split('\n')
    target_line = None
    for i, line in enumerate(lines):
        if chunk_text[:30] in line:
            target_line = i
            break
    if target_line is None:
        return {"document_title": structure["document_title"], "section_title": ""}
    section_title = structure["line_to_section"].get(target_line, "")
    return {"document_title": structure["document_title"], "section_title": section_title}


def build_enriched_text(chunk: dict, ctx: dict) -> str:
    parts = [p for p in [ctx["document_title"], ctx["section_title"], chunk["text"]] if p]
    return "\n".join(parts)


# ── Metrics ───────────────────────────────────────────────────────────

def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


def recall_at_k(ranks, k):
    return sum(1 for r in ranks if r is not None and r <= k) / len(ranks)


def mrr(ranks):
    return sum(1 / r if r else 0 for r in ranks) / len(ranks)


def ndcg_at_k(ranks, k):
    vals = [1 / math.log2(r + 1) if r and r <= k else 0 for r in ranks]
    return sum(vals) / len(vals)


def embed_texts(texts):
    embeddings = []
    for i in range(0, len(texts), 100):
        batch = texts[i:i + 100]
        resp = client.embeddings.create(model=settings.openai_embedding_model, input=batch)
        embeddings.extend([r.embedding for r in resp.data])
    return embeddings


def find_rank(chunk_id, ranked_ids):
    return ranked_ids.index(chunk_id) + 1 if chunk_id in ranked_ids else None


def best_rank(chunk_ids, ranked_ids):
    ranks = [find_rank(cid, ranked_ids) for cid in chunk_ids if cid in ranked_ids]
    return min(ranks) if ranks else None


# ── Configuration runner ──────────────────────────────────────────────

def run_configuration(config_name, global_chunks, chunk_contexts,
                      eval_set, embedding_mode, reranker_mode):
    print(f"\n{'─' * 60}")
    print(f"Configuration {config_name}  |  embedding={embedding_mode}  reranker={reranker_mode}")
    print(f"{'─' * 60}")

    # Build embedding texts
    embedding_texts = []
    for chunk in global_chunks:
        ctx = chunk_contexts[chunk["chunk_id"]]
        if embedding_mode == "enriched":
            embedding_texts.append(build_enriched_text(chunk, ctx))
        else:
            embedding_texts.append(chunk["text"])

    print(f"  Embedding {len(embedding_texts)} chunks...")
    chunk_embeddings = embed_texts(embedding_texts)
    chunk_id_list = [c["chunk_id"] for c in global_chunks]

    reranker = _load_reranker()

    per_query = []

    for item in eval_set:
        question = item["query"]
        primary_id = item["primary_chunk_id"]
        relevant_ids = item["relevant_chunk_ids"]

        # Embed query
        q_emb = embed_texts([question])[0]

        # Dense retrieval — score all global chunks
        scores = [cosine_similarity(q_emb, emb) for emb in chunk_embeddings]
        scored = sorted(zip(scores, global_chunks), key=lambda x: x[0], reverse=True)
        dense_ranked_ids = [c["chunk_id"] for _, c in scored]

        dense_primary_rank = find_rank(primary_id, dense_ranked_ids)
        dense_lenient_rank = best_rank(relevant_ids, dense_ranked_ids)
        dense_primary_score = scores[chunk_id_list.index(primary_id)] if primary_id in chunk_id_list else None

        # Reranker — top 15 candidates
        candidates = [c for _, c in scored[:15]]
        pairs = []
        for c in candidates:
            ctx = chunk_contexts[c["chunk_id"]]
            if reranker_mode == "enriched":
                pairs.append((question, build_enriched_text(c, ctx)))
            else:
                pairs.append((question, c["text"]))

        reranker_scores_list = reranker.predict(pairs)
        reranker_scored = sorted(
            zip(reranker_scores_list, candidates),
            key=lambda x: x[0], reverse=True
        )
        reranker_ranked_ids = [c["chunk_id"] for _, c in reranker_scored]

        reranker_primary_rank = find_rank(primary_id, reranker_ranked_ids)
        reranker_lenient_rank = best_rank(relevant_ids, reranker_ranked_ids)

        # Primary reranker score
        primary_reranker_score = None
        for rs, c in zip(reranker_scores_list, candidates):
            if c["chunk_id"] == primary_id:
                primary_reranker_score = float(rs)
                break

        per_query.append({
            "query_id": item["query_id"],
            "query": question,
            "query_type": item["query_type"],
            "expected_failure_mode": item["expected_failure_mode"],
            "document_id": item["document_id"],
            "section_id": item["section_id"],
            "primary_chunk_id": primary_id,
            "dense_primary_rank": dense_primary_rank,
            "dense_lenient_rank": dense_lenient_rank,
            "dense_primary_score": round(dense_primary_score, 4) if dense_primary_score else None,
            "reranker_primary_rank": reranker_primary_rank,
            "reranker_lenient_rank": reranker_lenient_rank,
            "reranker_primary_score": round(primary_reranker_score, 4) if primary_reranker_score else None,
        })

    # Aggregate metrics
    dp_ranks = [q["dense_primary_rank"] for q in per_query]
    dl_ranks = [q["dense_lenient_rank"] for q in per_query]
    rp_ranks = [q["reranker_primary_rank"] for q in per_query]
    rl_ranks = [q["reranker_lenient_rank"] for q in per_query]

    metrics = {
        "dense_strict": {
            "recall_at_5":  recall_at_k(dp_ranks, 5),
            "recall_at_10": recall_at_k(dp_ranks, 10),
            "recall_at_15": recall_at_k(dp_ranks, 15),
            "mrr":          mrr(dp_ranks),
        },
        "dense_lenient": {
            "recall_at_5":  recall_at_k(dl_ranks, 5),
            "recall_at_10": recall_at_k(dl_ranks, 10),
            "recall_at_15": recall_at_k(dl_ranks, 15),
            "mrr":          mrr(dl_ranks),
        },
        "reranker_strict": {
            "mrr":     mrr(rp_ranks),
            "ndcg_5":  ndcg_at_k(rp_ranks, 5),
            "ndcg_10": ndcg_at_k(rp_ranks, 10),
        },
        "reranker_lenient": {
            "mrr":     mrr(rl_ranks),
            "ndcg_5":  ndcg_at_k(rl_ranks, 5),
            "ndcg_10": ndcg_at_k(rl_ranks, 10),
        },
    }

    print(f"\n  Dense (strict)   R@5={metrics['dense_strict']['recall_at_5']:.2f} "
          f"R@10={metrics['dense_strict']['recall_at_10']:.2f} "
          f"MRR={metrics['dense_strict']['mrr']:.3f}")
    print(f"  Dense (lenient)  R@5={metrics['dense_lenient']['recall_at_5']:.2f} "
          f"R@10={metrics['dense_lenient']['recall_at_10']:.2f} "
          f"MRR={metrics['dense_lenient']['mrr']:.3f}")
    print(f"  Reranker (strict)  MRR={metrics['reranker_strict']['mrr']:.3f} "
          f"nDCG@5={metrics['reranker_strict']['ndcg_5']:.3f}")
    print(f"  Reranker (lenient) MRR={metrics['reranker_lenient']['mrr']:.3f} "
          f"nDCG@5={metrics['reranker_lenient']['ndcg_5']:.3f}")

    return {"config": config_name, "embedding_mode": embedding_mode,
            "reranker_mode": reranker_mode, "metrics": metrics, "per_query": per_query}


# ── Stratified analysis ───────────────────────────────────────────────

def stratified_report(results, dimension):
    """Report reranker strict MRR by a dimension (query_type or expected_failure_mode)."""
    configs = ["A", "D1", "D2"]
    # Collect all unique values
    all_values = sorted(set(
        q[dimension] for r in results.values() for q in r["per_query"]
    ))

    print(f"\n  By {dimension}:")
    print(f"  {'Value':<30} {'A MRR':>8} {'D1 MRR':>8} {'D2 MRR':>8}")
    print(f"  {'─'*58}")
    for val in all_values:
        row = f"  {val:<30}"
        for cfg in configs:
            ranks = [
                q["reranker_primary_rank"]
                for q in results[cfg]["per_query"]
                if q[dimension] == val
            ]
            m = mrr(ranks) if ranks else 0
            row += f" {m:>8.3f}"
        print(row)


# ── Main ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Load data
    with open('data/chunks/sentence_window.json') as f:
        all_chunks = json.load(f)
    global_chunks = [c for c in all_chunks if c.get('client') == 'global']

    with open('data/eval_set_38_audited.json') as f:
        eval_set = json.load(f)

    raw_dir = Path('data/raw')

    print(f"Global chunks : {len(global_chunks)}")
    print(f"Eval queries  : {len(eval_set)}")

    # Pre-compute section context
    print("\nExtracting section context...")
    chunk_contexts = {}
    for chunk in global_chunks:
        chunk_contexts[chunk["chunk_id"]] = get_chunk_context(chunk, raw_dir)

    # Run three configurations
    configurations = [
        ("A",  "chunk",    "chunk"),
        ("D1", "enriched", "chunk"),
        ("D2", "enriched", "enriched"),
    ]

    results = {}
    for config_name, emb_mode, rer_mode in configurations:
        results[config_name] = run_configuration(
            config_name, global_chunks, chunk_contexts,
            eval_set, emb_mode, rer_mode
        )

    # Summary table
    print(f"\n{'='*70}")
    print("SUMMARY — A vs D1 vs D2 on 38-query audited benchmark")
    print(f"{'='*70}")
    print(f"\n{'Metric':<35} {'A':>8} {'D1':>8} {'D2':>8}")
    print(f"{'─'*60}")

    metric_labels = [
        ("Dense strict R@5",       "dense_strict",   "recall_at_5"),
        ("Dense strict R@10",      "dense_strict",   "recall_at_10"),
        ("Dense strict MRR",       "dense_strict",   "mrr"),
        ("Dense lenient R@5",      "dense_lenient",  "recall_at_5"),
        ("Dense lenient MRR",      "dense_lenient",  "mrr"),
        ("Reranker strict MRR",    "reranker_strict","mrr"),
        ("Reranker strict nDCG@5", "reranker_strict","ndcg_5"),
        ("Reranker strict nDCG@10","reranker_strict","ndcg_10"),
        ("Reranker lenient MRR",   "reranker_lenient","mrr"),
        ("Reranker lenient nDCG@5","reranker_lenient","ndcg_5"),
    ]

    for label, group, metric in metric_labels:
        row = f"{label:<35}"
        for cfg in ["A", "D1", "D2"]:
            val = results[cfg]["metrics"][group][metric]
            row += f" {val:>8.3f}"
        print(row)

    # Per-query rank table for original 4 queries
    original_ids = {"Q04", "Q10", "Q16", "Q20"}
    print(f"\nOriginal 4-query comparison (dense → reranker, strict):")
    print(f"{'QID':<6} {'Query':<45} {'A':>12} {'D1':>12} {'D2':>12}")
    print(f"{'─'*90}")
    for q in results["A"]["per_query"]:
        if q["query_id"] in original_ids:
            row = f"{q['query_id']:<6} {q['query'][:44]:<45}"
            for cfg in ["A", "D1", "D2"]:
                qr = next(x for x in results[cfg]["per_query"] if x["query_id"] == q["query_id"])
                dr = str(qr["dense_primary_rank"]) if qr["dense_primary_rank"] else "None"
                rr = str(qr["reranker_primary_rank"]) if qr["reranker_primary_rank"] else "None"
                row += f" {dr+'→'+rr:>12}"
            print(row)

    # Stratified analysis
    print(f"\n{'='*70}")
    print("STRATIFIED ANALYSIS — Reranker strict MRR")
    print(f"{'='*70}")
    stratified_report(results, "query_type")
    stratified_report(results, "expected_failure_mode")
    stratified_report(results, "document_id")

    # Save full results
    out_path = Path("data/eval_38_results.json")
    with open(out_path, "w") as f:
        json.dump({
            k: {
                "config": v["config"],
                "metrics": v["metrics"],
                "per_query": v["per_query"],
            }
            for k, v in results.items()
        }, f, indent=2)
    print(f"\nFull results → {out_path}")
