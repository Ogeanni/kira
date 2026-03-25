"""
scripts/ingest_docs.py

Loads chunks and embeddings from Layer 1 into the vector store.
Connects ingestion → knowledge layers for the first time.

This script is idempotent — safe to run multiple times.
ChromaDB and Pinecone both use upsert, so re-running updates
existing vectors rather than creating duplicates.

Run:
    python scripts/ingest_docs.py
    python scripts/ingest_docs.py --backend pinecone  (override .env)
"""

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import get_settings
from knowledge import get_vector_store

settings = get_settings()


def load_chunks_and_embeddings(strategy: str) -> tuple[list[dict], list[dict]]:
    """
    Loads chunk and embedding JSON files for a given strategy.
    Raises clearly if files are missing — run chunker.py and
    embedder.py first.
    """
    chunk_path = settings.chunks_dir / f"{strategy}.json"
    emb_path = settings.embeddings_dir / f"{strategy}.json"

    if not chunk_path.exists():
        raise FileNotFoundError(
            f"Chunks not found: {chunk_path}\n"
            f"Run: python ingestion/chunker.py"
        )
    if not emb_path.exists():
        raise FileNotFoundError(
            f"Embeddings not found: {emb_path}\n"
            f"Run: python ingestion/embedder.py"
        )

    with open(chunk_path) as f:
        chunks = json.load(f)
    with open(emb_path) as f:
        embeddings = json.load(f)

    return chunks, embeddings


def verify_alignment(chunks: list[dict], embeddings: list[dict]) -> None:
    """
    Verifies every chunk has a matching embedding.
    Misalignment would mean some chunks get stored without vectors
    — they'd never be retrieved.
    """
    chunk_ids = {c["chunk_id"] for c in chunks}
    emb_ids = {e["chunk_id"] for e in embeddings}

    missing = chunk_ids - emb_ids
    if missing:
        raise ValueError(
            f"{len(missing)} chunks have no embedding.\n"
            f"Re-run: python ingestion/embedder.py"
        )


def ingest(strategy: str, backend_override: str | None = None) -> None:
    """
    Full ingestion pipeline for one chunking strategy.
    Loads → verifies → upserts → confirms count.
    """
    # Allow CLI override of backend without changing .env
    if backend_override:
        import os
        os.environ["VECTOR_STORE_BACKEND"] = backend_override
        # Clear lru_cache so settings re-reads the env var
        from config.settings import get_settings as _gs
        _gs.cache_clear()

    store = get_vector_store()
    backend_name = type(store).__name__

    print(f"\n── Ingesting: {strategy} → {backend_name} ──")

    # Load
    chunks, embeddings = load_chunks_and_embeddings(strategy)
    print(f"  Chunks loaded    : {len(chunks)}")
    print(f"  Embeddings loaded: {len(embeddings)}")

    # Verify
    verify_alignment(chunks, embeddings)
    print(f"  Alignment check  : OK")

    # Show namespace breakdown before upsert
    from collections import Counter
    ns_counts = Counter(c.get("client", "global") for c in chunks)
    for ns, count in sorted(ns_counts.items()):
        print(f"  Namespace '{ns}': {count} chunks")

    # Upsert
    count_before = store.count()
    upserted = store.upsert(chunks, embeddings)
    count_after = store.count()

    print(f"\n  Vectors before   : {count_before}")
    print(f"  Vectors upserted : {upserted}")
    print(f"  Vectors after    : {count_after}")


def run_spot_check(store, strategy: str) -> None:
    """
    Runs a quick retrieval check after ingestion.
    Verifies the store actually returns relevant results —
    not just that upsert didn't error.
    """
    from openai import OpenAI
    client = OpenAI(api_key=settings.openai_api_key)

    test_query = "What ACOS threshold triggers bid reduction?"
    response = client.embeddings.create(
        model=settings.openai_embedding_model,
        input=test_query,
    )
    query_embedding = response.data[0].embedding

    print(f"\n── Spot check: '{test_query}' ──")

    # Query global namespace
    results = store.query(
        query_embedding=query_embedding,
        top_k=3,
        namespace="global",
    )

    for i, r in enumerate(results, 1):
        print(f"\n  Result {i} (score: {r.score})")
        print(f"  Source : {r.metadata.get('source_file', 'unknown')}")
        print(f"  Text   : {r.text[:80]}...")

    # Query client-specific namespace
    print(f"\n── Spot check: natura namespace ──")
    natura_results = store.query(
        query_embedding=query_embedding,
        top_k=2,
        namespace="natura",
    )

    if natura_results:
        for i, r in enumerate(natura_results, 1):
            print(f"\n  Result {i} (score: {r.score})")
            print(f"  Text   : {r.text[:80]}...")
    else:
        print("  No results — natura namespace may be empty")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest documents into vector store")
    parser.add_argument(
        "--backend",
        choices=["chromadb", "pinecone"],
        help="Override vector store backend (default: from .env)",
    )
    parser.add_argument(
        "--strategy",
        default=settings.chunking_strategy,
        choices=["fixed_size", "recursive", "sentence_window"],
        help=f"Chunking strategy (default: {settings.chunking_strategy} from .env)",
    )
    parser.add_argument(
        "--skip-spot-check",
        action="store_true",
        help="Skip retrieval spot check after ingestion",
    )
    args = parser.parse_args()

    print(f"Strategy : {args.strategy}")
    print(f"Backend  : {args.backend or settings.vector_store_backend}")

    ingest(strategy=args.strategy, backend_override=args.backend)

    if not args.skip_spot_check:
        store = get_vector_store()
        run_spot_check(store, args.strategy)

    print("\nIngestion complete.")