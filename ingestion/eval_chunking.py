"""
ingestion/eval_chunking.py

Benchmarks all three chunking strategies using Ragas.
Answers: which strategy produces the best retrieval quality?

How it works:
  1. We define 5 questions with known ground truth answers
     drawn directly from our synthetic documents.
  2. For each strategy, we retrieve the top-3 chunks per question
     using cosine similarity (no vector DB needed yet).
  3. We pass question + retrieved chunks to GPT-4o to generate an answer.
  4. Ragas scores the answer on three metrics:
       faithfulness      — is the answer grounded in the chunks?
       answer_relevancy  — does it actually address the question?
       context_precision — are the retrieved chunks relevant?
  5. The winning strategy gets set in your .env.

Run:
    python ingestion/eval_chunking.py
"""
import json
import os
from pathlib import Path
import sys
from dotenv import load_dotenv

load_dotenv()

import numpy as np
from datasets import Dataset
from openai import OpenAI
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import get_settings

settings = get_settings()

client = OpenAI(api_key=settings.openai_api_key)
# ── Eval set ──────────────────────────────────────────────────────────

EVAL_SET = [
    {
        "question": "What format should Amazon listing titles follow?",
        "ground_truth": (
            "Titles should lead with the primary keyword and follow the format: "
            "Brand, Primary Keyword, Key Feature, Size or Variant, Secondary Keyword. "
            "Maximum 200 characters. Never use ALL CAPS or promotional language."
        ),
    },
    {
        "question": "What ACOS threshold triggers bid reduction on keywords?",
        "ground_truth": (
            "If ACOS exceeds the target by more than 10% for 3 consecutive days, "
            "bids on keywords with ACOS above 60% should be reduced by 15%."
        ),
    },
    {
        "question": "What terms are absolutely prohibited in Amazon listing copy?",
        "ground_truth": (
            "Prohibited terms include superlatives like best and #1, "
            "medical claims like cure and treat, guarantee language, "
            "competitor brand names, and pricing language like sale or discount."
        ),
    },
    {
        "question": "What should the executive summary of a weekly report contain?",
        "ground_truth": (
            "The executive summary should be 3 to 5 sentences, lead with the "
            "most important metric movement, state direction and magnitude, "
            "and include one sentence of causal explanation."
        ),
    },
    {
        "question": "What are the restricted terms specific to the Natura Skincare client?",
        "ground_truth": (
            "Natura prohibits anti-aging, chemical-free, and any competitor brand names. "
            "The word natural may only be used with EWG certification referenced. "
            "They use supports skin renewal instead of anti-aging."
        ),
    },
]


# ── Retrieval helpers ─────────────────────────────────────────────────

def get_embedding(text: str) -> list[float]:
    response = client.embeddings.create(
        model=settings.openai_embedding_model,
        input=text,
    )
    return response.data[0].embedding


def cosine_similarity(a: list[float], b: list[float]) -> float:
    a_arr = np.array(a)
    b_arr = np.array(b)
    return float(
        np.dot(a_arr, b_arr) / (np.linalg.norm(a_arr) * np.linalg.norm(b_arr))
    )


def retrieve_top_k(
    question: str,
    chunks: list[dict],
    embeddings: list[dict],
    k: int = 3,
) -> list[str]:
    question_embedding = get_embedding(question)
    emb_map = {e["chunk_id"]: e["embedding"] for e in embeddings}

    scored = []
    for chunk in chunks:
        emb = emb_map.get(chunk["chunk_id"])
        if emb is None:
            continue
        score = cosine_similarity(question_embedding, emb)
        context = chunk.get("window_context") or chunk["text"]
        scored.append((score, context))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [context for _, context in scored[:k]]


def generate_answer(question: str, contexts: list[str]) -> str:
    context_block = "\n\n---\n\n".join(contexts)
    response = client.chat.completions.create(
        model=settings.openai_chat_model,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer the question using only the provided context. "
                    "Be concise and factual. "
                    "If the context does not contain enough information, "
                    "say so explicitly — do not invent details."
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{context_block}\n\nQuestion: {question}",
            },
        ],
    )
    return response.choices[0].message.content


# ── Evaluation ────────────────────────────────────────────────────────

def evaluate_strategy(strategy: str) -> dict:
    chunk_path = settings.chunks_dir / f"{strategy}.json"
    emb_path = settings.embeddings_dir / f"{strategy}.json"

    if not chunk_path.exists() or not emb_path.exists():
        print(f"  Skipping {strategy} — run chunker.py and embedder.py first.")
        return {}

    with open(chunk_path) as f:
        chunks = json.load(f)
    with open(emb_path) as f:
        embeddings = json.load(f)

    rows = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [],
    }

    for item in EVAL_SET:
        contexts = retrieve_top_k(item["question"], chunks, embeddings, k=3)
        answer = generate_answer(item["question"], contexts)

        rows["question"].append(item["question"])
        rows["answer"].append(answer)
        rows["contexts"].append(contexts)
        rows["ground_truth"].append(item["ground_truth"])

        print(f"    Q: {item['question'][:55]}...")
        print(f"    A: {answer[:80]}...")
        print()

    dataset = Dataset.from_dict(rows)
    scores = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_precision],
)
    return scores


def run_eval():
    print("\nRunning Ragas evaluation across chunking strategies...")
    print(f"Questions: {len(EVAL_SET)} | Retrieval k=3 | Model: {settings.openai_chat_model}\n")

    all_scores = {}
    for strategy in ["fixed_size", "recursive", "sentence_window"]:
        print(f"── {strategy} ──")
        scores = evaluate_strategy(strategy)
        all_scores[strategy] = scores
        if scores:
            print(f"  faithfulness      : {scores['faithfulness']:.3f}")
            print(f"  answer_relevancy  : {scores['answer_relevancy']:.3f}")
            print(f"  context_precision : {scores['context_precision']:.3f}")
        print()

    # Rank by average of faithfulness + context_precision
    ranked = sorted(
        [(s, m) for s, m in all_scores.items() if m],
        key=lambda x: (
            x[1].get("faithfulness", 0) + x[1].get("context_precision", 0)
        ) / 2,
        reverse=True,
    )

    print("── Results ──")
    print(f"{'Strategy':<22} {'Faithful':>10} {'Relevancy':>10} {'Precision':>10}")
    print("─" * 56)
    for strategy, scores in ranked:
        print(
            f"{strategy:<22} "
            f"{scores.get('faithfulness', 0):>10.3f} "
            f"{scores.get('answer_relevancy', 0):>10.3f} "
            f"{scores.get('context_precision', 0):>10.3f}"
        )

    if ranked:
        winner = ranked[0][0]
        winner_scores = ranked[0][1]
        print(f"\nWinner: {winner}")
        print(f"  faithfulness {winner_scores.get('faithfulness', 0):.3f} "
              f"(threshold: {settings.ragas_faithfulness_threshold})")

        if winner_scores.get("faithfulness", 0) >= settings.ragas_faithfulness_threshold:
            print(f"  Meets production threshold.")
        else:
            print(f"  Below threshold. Consider improving documents or eval questions.")

        print(f"\nNext step: set CHUNKING_STRATEGY={winner} in your .env")

    # Save results
    out_path = settings.data_dir / "chunking_eval_results.json"
    with open(out_path, "w") as f:
        json.dump(
            {s: dict(m) for s, m in all_scores.items() if m},
            f, indent=2,
        )
    print(f"Results saved → {out_path}")


if __name__ == "__main__":
    run_eval()