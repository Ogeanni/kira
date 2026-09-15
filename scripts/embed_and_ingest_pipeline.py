"""
scripts/embed_and_ingest_pipeline.py

Embeds pipeline chunks with D2 representation and ingests to Pinecone.

D2 representation (from experiment):
    embedding_text = document_title + "\n" + section_title + "\n" + chunk_text
    reranker_text  = same as embedding_text

    chunk_text (source truth) is stored in metadata unchanged.
    window_context is stored in metadata for LLM delivery.

Namespacing:
    client == "global"  → namespace "global"
    client == "natura"  → namespace "natura"
    etc.
"""

import json
import time
from pathlib import Path
import sys
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from openai import OpenAI
from pinecone import Pinecone
from config.settings import get_settings

settings = get_settings()
client = OpenAI(api_key=settings.openai_api_key)
pc = Pinecone(api_key=settings.pinecone_api_key)
index = pc.Index(settings.pinecone_index_name)


def build_d2_text(chunk: dict) -> str:
    """
    Builds D2 embedding representation.
    Omits empty components gracefully.
    """
    parts = [
        p for p in [
            chunk.get("document_title", ""),
            chunk.get("section_title", ""),
            chunk.get("text", ""),
        ]
        if p and p.strip()
    ]
    return "\n".join(parts)


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Embeds a batch of texts using OpenAI."""
    response = client.embeddings.create(
        model=settings.openai_embedding_model,
        input=texts,
    )
    return [r.embedding for r in response.data]


def ingest_to_pinecone(chunks: list[dict]) -> dict:
    """
    Embeds chunks with D2 representation and upserts to Pinecone.
    Returns summary of what was ingested.
    """
    # Group by namespace
    by_namespace = {}
    for chunk in chunks:
        ns = chunk.get("client", "global")
        if ns not in by_namespace:
            by_namespace[ns] = []
        by_namespace[ns].append(chunk)

    print(f"Namespaces to ingest: {sorted(by_namespace.keys())}")
    for ns, ns_chunks in sorted(by_namespace.items()):
        print(f"  {ns}: {len(ns_chunks)} chunks")

    total_upserted = 0
    batch_size = 50

    for namespace, ns_chunks in sorted(by_namespace.items()):
        print(f"\nIngesting namespace: {namespace} ({len(ns_chunks)} chunks)")

        for i in range(0, len(ns_chunks), batch_size):
            batch = ns_chunks[i:i + batch_size]

            # Build D2 embedding texts
            embedding_texts = [build_d2_text(c) for c in batch]

            # Embed
            embeddings = embed_batch(embedding_texts)

            # Build Pinecone vectors
            vectors = []
            for chunk, embedding in zip(batch, embeddings):
                vectors.append({
                    "id": chunk["chunk_id"],
                    "values": embedding,
                    "metadata": {
                        "chunk_text":     chunk["text"],
                        "window_context": chunk.get("window_context", chunk["text"]),
                        "source_file":    chunk["source_file"],
                        "doc_type":       chunk["doc_type"],
                        "client":         chunk.get("client", "global"),
                        "section_title":  chunk.get("section_title", ""),
                        "document_title": chunk.get("document_title", ""),
                        "chunk_strategy": chunk.get("chunk_strategy", ""),
                        "chunk_index":    chunk.get("chunk_index", 0),
                        "embedding_repr": "D2",
                    }
                })

            index.upsert(vectors=vectors, namespace=namespace)
            total_upserted += len(vectors)
            print(f"  Batch {i//batch_size + 1}: upserted {len(vectors)} vectors")

            # Avoid rate limits
            time.sleep(0.5)

    return {"total_upserted": total_upserted, "namespaces": list(by_namespace.keys())}


if __name__ == "__main__":
    chunks_path = Path("data/chunks/pipeline_final.json")

    print(f"Loading chunks from {chunks_path}")
    with open(chunks_path) as f:
        chunks = json.load(f)

    print(f"Total chunks: {len(chunks)}")

    # Verify D2 text for ground truth chunk
    gt = next(
        (c for c in chunks if "buy_box" in c["source_file"].lower()
         and "Common causes" in c.get("text", "")),
        None
    )
    if gt:
        print(f"\nD2 text for ground truth chunk:")
        print(build_d2_text(gt))
        print()

    # Clear existing index
    print("Clearing existing Pinecone index...")
    stats = index.describe_index_stats()
    for ns in stats.namespaces:
        index.delete(delete_all=True, namespace=ns)
        print(f"  Cleared namespace: {ns}")
    time.sleep(2)

    # Ingest
    result = ingest_to_pinecone(chunks)

    # Verify
    time.sleep(3)
    stats = index.describe_index_stats()
    print(f"\nVerification:")
    print(f"  Total vectors: {stats.total_vector_count}")
    for ns, data in stats.namespaces.items():
        print(f"  {ns}: {data.vector_count} vectors")

    print(f"\nDone. {result['total_upserted']} vectors ingested with D2 representation.")
