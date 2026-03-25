"""
ingestion/embedder.py

Embeds chunks using OpenAI text-embedding-3-small.
Production concerns handled here:
  1. Caching    — never re-embed the same text twice
  2. Batching   — up to 100 inputs per API request
  3. Cost track — know exactly what you spend per run
  4. Retry      — backs off on rate limit (429) errors

Run directly to embed all three chunking strategies:
    python ingestion/embedder.py
"""

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
import sys

from openai import OpenAI
from dotenv import load_dotenv

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import get_settings

load_dotenv()
settings = get_settings()


@dataclass
class EmbeddingResult:
    chunk_id: str
    embedding: list[float]
    model: str
    cache_hit: bool
    tokens_used: int
    cost_usd: float


class Embedder:
    """
    Wraps OpenAI embeddings with caching, batching, and cost tracking.

    Why cache?
    ----------
    text-embedding-3-small costs $0.00002 per 1K tokens.
    On a small dataset this is fractions of a cent — but the habit
    matters. In production, re-embedding the same SOPs on every
    pipeline run would waste money and add latency. The cache is a
    SHA-256 hash of (model + text) → embedding vector stored on disk.
    Same text + same model = instant return, zero API call.

    Why batch?
    ----------
    OpenAI allows up to 2048 inputs per request. Sending chunks one
    at a time means one HTTP round-trip per chunk — slow and wasteful.
    Batching 100 chunks per request reduces round-trips by 100x.
    """

    def __init__(self):
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_embedding_model
        self.cache_path = settings.embeddings_dir / ".cache.json"
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache = self._load_cache()

        # Session tracking — reset each time Embedder is instantiated
        self.session_tokens = 0
        self.session_cost = 0.0
        self.session_cache_hits = 0
        self.session_api_calls = 0

    def _load_cache(self) -> dict:
        if self.cache_path.exists():
            with open(self.cache_path) as f:
                return json.load(f)
        return {}

    def _save_cache(self):
        with open(self.cache_path, "w") as f:
            json.dump(self.cache, f)

    def _cache_key(self, text: str) -> str:
        """
        Deterministic key: SHA-256 of model name + text content.
        Changing the model invalidates all cache entries automatically.
        """
        return hashlib.sha256(f"{self.model}:{text}".encode()).hexdigest()

    def embed_chunks(self, chunks: list[dict], batch_size: int = 100) -> list[EmbeddingResult]:
        """
        Embeds a list of chunk dicts (as loaded from JSON).
        Returns EmbeddingResult for each chunk in the same order.

        Two-pass approach:
          Pass 1 — check cache for every chunk
          Pass 2 — batch embed only the cache misses
        This preserves output ordering without complex bookkeeping.
        """
        results: list[EmbeddingResult | None] = [None] * len(chunks)
        to_embed: list[tuple[int, dict]] = []   # (original_index, chunk)

        # Pass 1: cache lookup
        for i, chunk in enumerate(chunks):
            key = self._cache_key(chunk["text"])
            if key in self.cache:
                results[i] = EmbeddingResult(
                    chunk_id=chunk["chunk_id"],
                    embedding=self.cache[key],
                    model=self.model,
                    cache_hit=True,
                    tokens_used=0,
                    cost_usd=0.0,
                )
                self.session_cache_hits += 1
            else:
                to_embed.append((i, chunk))

        if not to_embed:
            print(f"  All {len(chunks)} chunks served from cache.")
            return results

        print(f"  Cache hits: {self.session_cache_hits} | "
              f"To embed: {len(to_embed)}")

        # Pass 2: batch embed cache misses
        for batch_start in range(0, len(to_embed), batch_size):
            batch = to_embed[batch_start: batch_start + batch_size]
            texts = [chunk["text"] for _, chunk in batch]

            # Retry up to 3 times on rate limit
            for attempt in range(3):
                try:
                    response = self.client.embeddings.create(
                        model=self.model,
                        input=texts,
                    )
                    break
                except Exception as e:
                    if "429" in str(e) and attempt < 2:
                        wait = 2 ** attempt     # 1s, 2s, 4s
                        print(f"  Rate limited. Waiting {wait}s...")
                        time.sleep(wait)
                    else:
                        raise

            tokens = response.usage.total_tokens
            cost = (tokens / 1000) * settings.embedding_cost_per_1k_tokens
            self.session_tokens += tokens
            self.session_cost += cost
            self.session_api_calls += 1

            batch_num = batch_start // batch_size + 1
            total_batches = (len(to_embed) + batch_size - 1) // batch_size
            print(f"  Batch {batch_num}/{total_batches}: "
                  f"{tokens} tokens | ${cost:.6f}")

            # Write results back to their original positions
            for (orig_idx, chunk), emb_data in zip(batch, response.data):
                key = self._cache_key(chunk["text"])
                self.cache[key] = emb_data.embedding   # store in cache

                results[orig_idx] = EmbeddingResult(
                    chunk_id=chunk["chunk_id"],
                    embedding=emb_data.embedding,
                    model=self.model,
                    cache_hit=False,
                    tokens_used=tokens // len(batch),   # approximate per chunk
                    cost_usd=cost / len(batch),
                )

        self._save_cache()
        return results

    def print_session_summary(self):
        print(f"\n── Embedding session summary ──")
        print(f"  API calls made    : {self.session_api_calls}")
        print(f"  Tokens used       : {self.session_tokens:,}")
        print(f"  Total cost        : ${self.session_cost:.6f}")
        print(f"  Cache hits        : {self.session_cache_hits}")
        print(f"  Cache size (disk) : {len(self.cache)} entries")


def embed_strategy(strategy: str) -> list[EmbeddingResult]:
    """
    Loads chunks for a strategy, embeds them, saves results.
    Returns list of EmbeddingResult.
    """
    chunk_path = settings.chunks_dir / f"{strategy}.json"
    if not chunk_path.exists():
        raise FileNotFoundError(
            f"No chunks found for '{strategy}'. "
            f"Run: python ingestion/chunker.py first."
        )

    with open(chunk_path) as f:
        chunks = json.load(f)

    embedder = Embedder()
    print(f"\n── Strategy: {strategy} ({len(chunks)} chunks) ──")
    results = embedder.embed_chunks(chunks)

    # Save embeddings alongside chunks
    out_path = settings.embeddings_dir / f"{strategy}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump([
            {
                "chunk_id": r.chunk_id,
                "embedding": r.embedding,
                "model": r.model,
                "cache_hit": r.cache_hit,
                "tokens_used": r.tokens_used,
                "cost_usd": r.cost_usd,
            }
            for r in results
        ], f)

    embedder.print_session_summary()
    print(f"  Saved → {out_path}")
    return results


if __name__ == "__main__":
    print("\nEmbedding all chunking strategies...")
    print(f"Model: {settings.openai_embedding_model}")
    print(f"Cache: {settings.embeddings_dir / '.cache.json'}")

    for strategy in ["fixed_size", "recursive", "sentence_window"]:
        embed_strategy(strategy)

    print("\n── Cache behaviour demo ──")
    print("Running fixed_size again — should be 100% cache hits, zero cost...")
    embed_strategy("fixed_size")
    print("\nDone. All embeddings saved to data/embeddings/")