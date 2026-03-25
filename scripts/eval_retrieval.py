"""
scripts/eval_retrieval.py

Runs Ragas evaluation against the live vector store.

Unlike ingestion/eval_chunking.py which evaluated chunking strategies
against JSON files, this evaluates the production retrieval pipeline
end to end — the same path an agent takes.

Use this as a quality gate:
  - Run after re-ingesting documents
  - Run after changing chunking strategy
  - Run in CI to catch retrieval regressions

Exit codes:
  0 — all metrics above threshold
  1 — one or more metrics below threshold

Run:
    python scripts/eval_retrieval.py
    python scripts/eval_retrieval.py --strategy sentence_window
"""

import argparse
import json
import sys
from pathlib import Path

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
from knowledge.retriever import Retriever, RetrievalQuery
from observability.tracer import log_retrieval_quality

settings = get_settings()
client = OpenAI(api_key=settings.openai_api_key)
retriever = Retriever()

# Same eval set as ingestion/eval_chunking.py
# Keeping them identical means scores are directly comparable
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


def generate_answer(question: str, contexts: list[str]) -> str:
    """Generates a grounded answer from retrieved contexts."""
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
                    "say so explicitly."
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{context_block}\n\nQuestion: {question}",
            },
        ],
    )
    return response.choices[0].message.content


def run_eval() -> dict:
    """
    Runs Ragas evaluation against the live vector store.
    Returns scores dict. Exits with code 1 if below threshold.
    """
    print(f"\nRunning live retrieval eval...")
    print(f"Backend  : {type(retriever.store).__name__}")
    print(f"Strategy : {settings.chunking_strategy}")
    print(f"Model    : {settings.openai_chat_model}")
    print(f"Questions: {len(EVAL_SET)}\n")

    rows = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [],
    }

    for item in EVAL_SET:
        # Use the same retrieval path as the Research Agent
        response = retriever.retrieve(RetrievalQuery(
            question=item["question"],
            namespace="global",
            top_k=settings.retrieval_top_k,
        ))

        contexts = response.contexts
        answer = generate_answer(item["question"], contexts)

        rows["question"].append(item["question"])
        rows["answer"].append(answer)
        rows["contexts"].append(contexts)
        rows["ground_truth"].append(item["ground_truth"])

        print(f"  Q: {item['question'][:55]}...")
        print(f"  A: {answer[:70]}...")
        print()

    dataset = Dataset.from_dict(rows)
    scores = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
    )

    faith = scores["faithfulness"]
    precision = scores["context_precision"]
    relevancy = scores["answer_relevancy"]

    print(f"── Results ──")
    print(f"  faithfulness      : {faith:.3f}  "
          f"(threshold: {settings.ragas_faithfulness_threshold})")
    print(f"  answer_relevancy  : {relevancy:.3f}")
    print(f"  context_precision : {precision:.3f}  "
          f"(threshold: {settings.ragas_context_precision_threshold})")

    # Log to LangSmith if enabled
    log_retrieval_quality(
        strategy=settings.chunking_strategy,
        faithfulness=faith,
        context_precision=precision,
    )

    # Save results
    out_path = settings.data_dir / "live_eval_results.json"
    with open(out_path, "w") as f:
        json.dump({
            "strategy": settings.chunking_strategy,
            "backend": type(retriever.store).__name__,
            "faithfulness": faith,
            "answer_relevancy": relevancy,
            "context_precision": precision,
            "thresholds": {
                "faithfulness": settings.ragas_faithfulness_threshold,
                "context_precision": settings.ragas_context_precision_threshold,
            },
        }, f, indent=2)
    print(f"\nResults saved → {out_path}")

    # Quality gate — exit 1 if below threshold
    passed = (
        faith >= settings.ragas_faithfulness_threshold
        and precision >= settings.ragas_context_precision_threshold
    )

    if passed:
        print(f"\nQuality gate: PASSED")
    else:
        print(f"\nQuality gate: FAILED")
        print(f"  One or more metrics below threshold.")
        print(f"  Check your chunking strategy and re-ingest if needed.")
        sys.exit(1)

    return dict(scores)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strategy",
        default=settings.chunking_strategy,
        help="Chunking strategy to evaluate (default: from .env)",
    )
    args = parser.parse_args()
    run_eval()