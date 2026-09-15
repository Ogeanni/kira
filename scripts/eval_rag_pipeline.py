"""
scripts/eval_rag_pipeline.py

Full RAG pipeline evaluation for KIRA.
Evaluates all three stages independently so failure can be
attributed to retrieval, context, or generation — not just
flagged as "the answer was wrong."

Three evaluation approaches combined:
  1. Ragas    — faithfulness, context precision, answer relevancy
  2. LLM-as-judge — quality scoring without ground truth labels
  3. Stage isolation — retrieval metrics separate from generation metrics

Why LLM-as-judge:
  Manual ground truth construction doesn't scale. At 5 questions it's
  feasible. At 500 it's not. LLM-as-judge automates quality assessment
  using a capable model (GPT-4o) to evaluate the internal consistency
  of what the system produced — no pre-written correct answer needed.
  The judge evaluates three things:
    - Is the answer faithful to the retrieved context?
    - Does the answer actually address the question?
    - Is the answer complete — does it cover what was asked?

When to run this:
  - Before deploying any change to chunking, embedding, or retrieval
  - After re-ingesting documents
  - As a pre-deployment quality gate in CI (exits with code 1 on failure)

Run:
    python scripts/eval_rag_pipeline.py
    python scripts/eval_rag_pipeline.py --client natura
    python scripts/eval_rag_pipeline.py --verbose
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision

from config.settings import get_settings
from knowledge.retriever import Retriever, RetrievalQuery

settings = get_settings()
client = OpenAI(api_key=settings.openai_api_key)
retriever = Retriever()


# ── Eval set ──────────────────────────────────────────────────────────
# Ground truth answers written independently from the source documents.
# Each question targets a specific layer of the knowledge base.
# The ground_truth is what the system SHOULD say — used by Ragas
# for answer correctness comparison.
# The expected_source tells us which document type should be retrieved —
# used for retrieval stage evaluation.

EVAL_SET = [
    {
        "question": "What format should Amazon listing titles follow?",
        "ground_truth": (
            "Titles should lead with the primary keyword and follow the format: "
            "Brand, Primary Keyword, Key Feature, Size or Variant, Secondary Keyword. "
            "Maximum 200 characters. Never use ALL CAPS or promotional language."
        ),
        "expected_source_type": "sop",
        "difficulty": "easy",
    },
    {
        "question": "What ACOS threshold triggers bid reduction on keywords?",
        "ground_truth": (
            "If ACOS exceeds the target by more than 10% for 3 consecutive days, "
            "bids on keywords with ACOS above 60% should be reduced by 15%."
        ),
        "expected_source_type": "sop",
        "difficulty": "medium",
    },
    {
        "question": "What terms are absolutely prohibited in Amazon listing copy?",
        "ground_truth": (
            "Prohibited terms include superlatives like best and number one, "
            "medical claims like cure and treat, guarantee language, "
            "competitor brand names, and pricing language like sale or discount."
        ),
        "expected_source_type": "compliance",
        "difficulty": "easy",
    },
    {
        "question": "What should the executive summary of a weekly report contain?",
        "ground_truth": (
            "The executive summary should be 3 to 5 sentences, lead with the "
            "most important metric movement, state direction and magnitude, "
            "and include one sentence of causal explanation."
        ),
        "expected_source_type": "framework",
        "difficulty": "medium",
    },
    {
        "question": "What steps should be taken when a product loses Buy Box?",
        "ground_truth": (
            "When a product loses Buy Box, first check if a competitor is "
            "undercutting price and match within 2 percent. Then verify inventory "
            "levels are above 30 days supply. Check seller feedback score and "
            "resolve any negative feedback. Review shipping performance metrics."
        ),
        "expected_source_type": "sop",
        "difficulty": "hard",
    },
    {
        "question": "What is the recommended image standard for main product images?",
        "ground_truth": (
            "Main product images must have a pure white background, show the "
            "product occupying at least 85 percent of the frame, be at least "
            "1000 pixels on the longest side, and show no additional objects, "
            "text, or graphics."
        ),
        "expected_source_type": "compliance",
        "difficulty": "medium",
    },
    {
        "question": "How should anomalies be investigated before escalating to a client?",
        "ground_truth": (
            "Before escalating, verify the anomaly against at least two data "
            "sources, check if the anomaly is isolated to one ASIN or affects "
            "the whole account, review recent listing changes or campaigns, "
            "and check for platform-wide issues on the Amazon seller forums."
        ),
        "expected_source_type": "framework",
        "difficulty": "hard",
    },
]


# ── Stage 1: Retrieval evaluation ─────────────────────────────────────

def evaluate_retrieval(
    question: str,
    expected_source_type: str,
    retrieved_chunks: list[dict],
    verbose: bool = False,
) -> dict:
    """
    Evaluates whether retrieval returned chunks from the expected source type.

    In a production system with labelled chunk IDs you'd measure Recall@k
    and MRR precisely. Without chunk labels, we measure source type match —
    did retrieval return at least one chunk from the expected document type?

    This is a proxy metric. It catches the most common retrieval failure:
    the system returned compliance chunks for an SOP question, or framework
    chunks for a compliance question.
    """
    if not retrieved_chunks:
        return {
            "retrieved_count": 0,
            "source_type_match": False,
            "source_types_found": [],
            "expected_source_type": expected_source_type,
            "top_score": 0.0,
            "score_spread": 0.0,
            "verdict": "FAIL — no chunks retrieved",
        }

    source_types_found = list(set(c.get("doc_type", "unknown") for c in retrieved_chunks))
    source_type_match = expected_source_type in source_types_found

    scores = [c.get("score", 0.0) for c in retrieved_chunks]
    top_score = max(scores) if scores else 0.0
    score_spread = (max(scores) - min(scores)) if len(scores) > 1 else 0.0

    verdict = "PASS" if source_type_match else f"FAIL — expected {expected_source_type}, got {source_types_found}"

    if verbose:
        print(f"    Retrieved {len(retrieved_chunks)} chunks")
        print(f"    Source types: {source_types_found}")
        print(f"    Top score: {top_score:.4f} | Spread: {score_spread:.4f}")
        print(f"    Retrieval verdict: {verdict}")

    return {
        "retrieved_count": len(retrieved_chunks),
        "source_type_match": source_type_match,
        "source_types_found": source_types_found,
        "expected_source_type": expected_source_type,
        "top_score": round(top_score, 4),
        "score_spread": round(score_spread, 4),
        "verdict": verdict,
    }


# ── Stage 2: Context evaluation ───────────────────────────────────────

def evaluate_context(
    question: str,
    contexts: list[str],
    verbose: bool = False,
) -> dict:
    """
    Evaluates the quality of the context window passed to the LLM.

    Checks three things:
    1. Context coverage  — does the combined context contain enough information?
    2. Context noise     — is the context focused or padded with irrelevant text?
    3. Context length    — is the context window appropriately sized?

    Uses LLM-as-judge for coverage and noise — these require semantic
    understanding that rule-based metrics can't provide.
    """
    if not contexts:
        return {
            "context_count": 0,
            "total_chars": 0,
            "coverage_score": 0,
            "noise_score": 0,
            "verdict": "FAIL — no context",
        }

    combined_context = "\n\n---\n\n".join(contexts)
    total_chars = len(combined_context)

    judge_prompt = f"""You are evaluating the context window provided to a RAG system.

Question asked: {question}

Context retrieved:
{combined_context[:3000]}

Score the context on two dimensions:

1. Coverage (1-5): Does the context contain enough information to answer the question?
   1 = context is completely irrelevant
   3 = context is partially relevant, some information present
   5 = context contains all information needed to answer fully

2. Noise (1-5): How focused is the context? Is it about the right topic?
   1 = context is mostly about unrelated topics
   3 = context mixes relevant and irrelevant information
   5 = context is tightly focused on the question topic

Respond ONLY with valid JSON:
{{"coverage_score": N, "noise_score": N, "coverage_reasoning": "...", "noise_reasoning": "..."}}"""

    try:
        response = client.chat.completions.create(
            model=settings.openai_chat_model,
            temperature=0,
            messages=[{"role": "user", "content": judge_prompt}],
        )
        result = json.loads(response.choices[0].message.content.strip())
        coverage_score = result.get("coverage_score", 0)
        noise_score = result.get("noise_score", 0)
    except Exception as e:
        coverage_score = 0
        noise_score = 0
        if verbose:
            print(f"    Context judge error: {e}")

    verdict = "PASS" if coverage_score >= 3 and noise_score >= 3 else "FAIL"

    if verbose:
        print(f"    Context chars: {total_chars}")
        print(f"    Coverage score: {coverage_score}/5 | Noise score: {noise_score}/5")
        print(f"    Context verdict: {verdict}")

    return {
        "context_count": len(contexts),
        "total_chars": total_chars,
        "coverage_score": coverage_score,
        "noise_score": noise_score,
        "verdict": verdict,
    }


# ── Stage 3: Generation evaluation ───────────────────────────────────

def generate_answer(question: str, contexts: list[str]) -> str:
    """Generates an answer from retrieved contexts — same as production."""
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
                    "say so explicitly — do not guess."
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{context_block}\n\nQuestion: {question}",
            },
        ],
    )
    return response.choices[0].message.content


def evaluate_generation_llm_judge(
    question: str,
    answer: str,
    contexts: list[str],
    ground_truth: str,
    verbose: bool = False,
) -> dict:
    """
    LLM-as-judge evaluation of the generated answer.

    Scores three dimensions:
    1. Faithfulness  — are all claims in the answer supported by context?
    2. Relevance     — does the answer address the question asked?
    3. Completeness  — does the answer cover what was needed?

    Why LLM-as-judge here:
    These dimensions require semantic reasoning. Rule-based metrics
    can check if words appear in context (faithfulness proxy) but
    can't evaluate whether the answer actually addresses the intent
    of the question or whether it's complete given what the context
    could support.

    The judge also gets the ground truth — it uses this to assess
    completeness but not to penalise different correct phrasings.
    """
    context_block = "\n\n---\n\n".join(contexts[:3])

    judge_prompt = f"""You are evaluating a RAG system's generated answer.

Question: {question}

Retrieved context:
{context_block[:2000]}

Generated answer:
{answer}

Reference answer (what a correct answer should contain):
{ground_truth}

Score the generated answer on three dimensions:

1. Faithfulness (1-5): Are all claims in the generated answer supported by the retrieved context?
   1 = answer contains claims not in context (hallucination)
   3 = answer mostly grounded, minor unsupported claims
   5 = every claim in the answer is explicitly supported by context

2. Relevance (1-5): Does the answer address the question that was asked?
   1 = answer is about something else entirely
   3 = answer partially addresses the question
   5 = answer directly and completely addresses the question

3. Completeness (1-5): Does the answer cover the key points from the reference answer?
   1 = answer is missing most key points from the reference
   3 = answer covers some key points but misses important ones
   5 = answer covers all key points from the reference answer

Respond ONLY with valid JSON:
{{"faithfulness": N, "relevance": N, "completeness": N, "faithfulness_reasoning": "...", "relevance_reasoning": "...", "completeness_reasoning": "...", "failure_stage": "retrieval|context|generation|none"}}

For failure_stage: if the answer is poor because the context didn't contain the right information, say "retrieval". If the context had the information but was noisy or poorly structured, say "context". If the context was good but the answer ignored or misused it, say "generation". If the answer is good, say "none"."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",  # use GPT-4o for judging — better reasoning
            temperature=0,
            messages=[{"role": "user", "content": judge_prompt}],
        )
        result = json.loads(response.choices[0].message.content.strip())
    except Exception as e:
        if verbose:
            print(f"    Generation judge error: {e}")
        result = {
            "faithfulness": 0,
            "relevance": 0,
            "completeness": 0,
            "failure_stage": "unknown",
        }

    avg_score = (
        result.get("faithfulness", 0) +
        result.get("relevance", 0) +
        result.get("completeness", 0)
    ) / 3

    verdict = "PASS" if avg_score >= 3.5 else "FAIL"

    if verbose:
        print(f"    Faithfulness: {result.get('faithfulness')}/5")
        print(f"    Relevance   : {result.get('relevance')}/5")
        print(f"    Completeness: {result.get('completeness')}/5")
        print(f"    Avg score   : {avg_score:.2f}/5")
        print(f"    Failure stage: {result.get('failure_stage', 'unknown')}")
        print(f"    Generation verdict: {verdict}")

    return {
        "faithfulness": result.get("faithfulness", 0),
        "relevance": result.get("relevance", 0),
        "completeness": result.get("completeness", 0),
        "avg_score": round(avg_score, 2),
        "failure_stage": result.get("failure_stage", "unknown"),
        "faithfulness_reasoning": result.get("faithfulness_reasoning", ""),
        "relevance_reasoning": result.get("relevance_reasoning", ""),
        "completeness_reasoning": result.get("completeness_reasoning", ""),
        "verdict": verdict,
    }


# ── Ragas evaluation ──────────────────────────────────────────────────

def run_ragas_eval(ragas_rows: dict) -> dict:
    """Runs Ragas on accumulated rows — faithfulness, relevancy, precision."""
    dataset = Dataset.from_dict(ragas_rows)
    scores = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
    )
    return {
        "faithfulness": round(float(scores["faithfulness"]), 3),
        "answer_relevancy": round(float(scores["answer_relevancy"]), 3),
        "context_precision": round(float(scores["context_precision"]), 3),
    }


# ── Failure attribution ───────────────────────────────────────────────

def attribute_failure(
    retrieval_result: dict,
    context_result: dict,
    generation_result: dict,
) -> str:
    """
    Attributes a failed answer to its root cause stage.

    Decision logic:
    1. If retrieval returned no chunks or wrong source types → retrieval failure
    2. If retrieval was ok but context coverage was low → context/ranking failure
    3. If retrieval and context were ok but generation scored low → generation failure
    4. If everything scored ok → no failure
    """
    if not retrieval_result["source_type_match"] or retrieval_result["retrieved_count"] == 0:
        return "retrieval"

    if context_result["coverage_score"] < 3:
        return "context"

    if generation_result["avg_score"] < 3.5:
        lm_stage = generation_result.get("failure_stage", "generation")
        return lm_stage if lm_stage != "none" else "generation"

    return "none"


# ── Main evaluation loop ──────────────────────────────────────────────

def run_full_eval(
    client_id: str = "global",
    verbose: bool = False,
) -> dict:
    """
    Runs the full three-stage evaluation pipeline.
    Returns a structured report with per-question and aggregate results.
    """
    print(f"\n{'═' * 55}")
    print(f"KIRA RAG Pipeline Evaluation")
    print(f"{'═' * 55}")
    print(f"Backend  : {type(retriever.store).__name__}")
    print(f"Strategy : {settings.chunking_strategy}")
    print(f"Client   : {client_id}")
    print(f"Questions: {len(EVAL_SET)}")
    print(f"Judge    : gpt-4o (generation) + {settings.openai_chat_model} (context)")
    print(f"{'─' * 55}\n")

    results = []
    ragas_rows = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [],
    }

    failure_attribution = {
        "retrieval": 0,
        "context": 0,
        "generation": 0,
        "none": 0,
    }

    for i, item in enumerate(EVAL_SET, 1):
        question = item["question"]
        print(f"Q{i}/{len(EVAL_SET)} [{item['difficulty'].upper()}] {question[:60]}...")

        # ── Retrieval ────────────────────────────────────────────────
        response = retriever.retrieve(RetrievalQuery(
            question=question,
            namespace="global",
            top_k=settings.retrieval_top_k,
        ))

        retrieved_chunks = [
            {
                "context": c,
                "doc_type": r.doc_type if hasattr(r, "doc_type") else "unknown",
                "score": r.score if hasattr(r, "score") else 0.0,
                "source_file": r.source_file if hasattr(r, "source_file") else "",
            }
            for c, r in zip(response.contexts, response.results)
        ] if hasattr(response, "results") else [
            {"context": c, "doc_type": "unknown", "score": 0.0, "source_file": ""}
            for c in response.contexts
        ]

        retrieval_result = evaluate_retrieval(
            question, item["expected_source_type"], retrieved_chunks, verbose
        )

        # ── Context ──────────────────────────────────────────────────
        contexts = response.contexts
        context_result = evaluate_context(question, contexts, verbose)

        # ── Generation ───────────────────────────────────────────────
        answer = generate_answer(question, contexts)

        if verbose:
            print(f"\n  Generated answer:\n  {answer[:150]}...\n")

        generation_result = evaluate_generation_llm_judge(
            question, answer, contexts, item["ground_truth"], verbose
        )

        # ── Failure attribution ───────────────────────────────────────
        failure_stage = attribute_failure(
            retrieval_result, context_result, generation_result
        )
        failure_attribution[failure_stage] += 1

        # ── Accumulate for Ragas ─────────────────────────────────────
        ragas_rows["question"].append(question)
        ragas_rows["answer"].append(answer)
        ragas_rows["contexts"].append(contexts)
        ragas_rows["ground_truth"].append(item["ground_truth"])

        result = {
            "question": question,
            "difficulty": item["difficulty"],
            "expected_source_type": item["expected_source_type"],
            "answer": answer,
            "retrieval": retrieval_result,
            "context": context_result,
            "generation": generation_result,
            "failure_stage": failure_stage,
        }
        results.append(result)

        status = "✓" if failure_stage == "none" else f"✗ [{failure_stage}]"
        print(f"  {status} | retrieval={retrieval_result['verdict'].split()[0]} | "
              f"context={context_result['coverage_score']}/5 | "
              f"gen={generation_result['avg_score']:.1f}/5\n")

    # ── Ragas ────────────────────────────────────────────────────────
    print("Running Ragas evaluation...")
    ragas_scores = run_ragas_eval(ragas_rows)

    # ── Aggregate scores ─────────────────────────────────────────────
    avg_coverage = sum(r["context"]["coverage_score"] for r in results) / len(results)
    avg_gen_score = sum(r["generation"]["avg_score"] for r in results) / len(results)
    retrieval_pass_rate = sum(
        1 for r in results if r["retrieval"]["source_type_match"]
    ) / len(results)

    # ── Quality gate ─────────────────────────────────────────────────
    gate_passed = (
        ragas_scores["faithfulness"] >= settings.ragas_faithfulness_threshold
        and ragas_scores["context_precision"] >= settings.ragas_context_precision_threshold
        and avg_gen_score >= 3.5
        and retrieval_pass_rate >= 0.8
    )

    report = {
        "timestamp": datetime.utcnow().isoformat(),
        "backend": type(retriever.store).__name__,
        "chunking_strategy": settings.chunking_strategy,
        "client_id": client_id,
        "questions_evaluated": len(EVAL_SET),
        "aggregate": {
            "retrieval_pass_rate": round(retrieval_pass_rate, 3),
            "avg_context_coverage": round(avg_coverage, 2),
            "avg_generation_score": round(avg_gen_score, 2),
            "ragas_faithfulness": ragas_scores["faithfulness"],
            "ragas_context_precision": ragas_scores["context_precision"],
            "ragas_answer_relevancy": ragas_scores["answer_relevancy"],
        },
        "failure_attribution": failure_attribution,
        "quality_gate_passed": gate_passed,
        "thresholds": {
            "ragas_faithfulness": settings.ragas_faithfulness_threshold,
            "ragas_context_precision": settings.ragas_context_precision_threshold,
            "avg_generation_score": 3.5,
            "retrieval_pass_rate": 0.8,
        },
        "per_question": results,
    }

    return report


def print_report(report: dict) -> None:
    """Prints a structured summary of the evaluation report."""
    agg = report["aggregate"]
    fa = report["failure_attribution"]

    print(f"\n{'═' * 55}")
    print(f"EVALUATION RESULTS")
    print(f"{'═' * 55}")

    print(f"\n── Stage 1: Retrieval ──────────────────────────────")
    print(f"  Pass rate    : {agg['retrieval_pass_rate']:.0%}")
    print(f"  (Did the right source type come back?)")

    print(f"\n── Stage 2: Context ────────────────────────────────")
    print(f"  Avg coverage : {agg['avg_context_coverage']:.1f}/5")
    print(f"  Precision    : {agg['ragas_context_precision']:.3f}")

    print(f"\n── Stage 3: Generation ─────────────────────────────")
    print(f"  Avg score    : {agg['avg_generation_score']:.2f}/5")
    print(f"  Faithfulness : {agg['ragas_faithfulness']:.3f}  (threshold: {report['thresholds']['ragas_faithfulness']})")
    print(f"  Relevancy    : {agg['ragas_answer_relevancy']:.3f}")

    print(f"\n── Failure attribution ─────────────────────────────")
    total = sum(fa.values())
    for stage, count in fa.items():
        pct = count / total * 100 if total > 0 else 0
        bar = "█" * count
        print(f"  {stage:<12}: {bar} {count} ({pct:.0f}%)")

    print(f"\n── Quality gate ────────────────────────────────────")
    gate = "✓ PASSED" if report["quality_gate_passed"] else "✗ FAILED"
    print(f"  {gate}")

    if not report["quality_gate_passed"]:
        print(f"\n  Failed thresholds:")
        agg = report["aggregate"]
        t = report["thresholds"]
        if agg["ragas_faithfulness"] < t["ragas_faithfulness"]:
            print(f"    Faithfulness {agg['ragas_faithfulness']:.3f} < {t['ragas_faithfulness']}")
        if agg["ragas_context_precision"] < t["ragas_context_precision"]:
            print(f"    Context precision {agg['ragas_context_precision']:.3f} < {t['ragas_context_precision']}")
        if agg["avg_generation_score"] < t["avg_generation_score"]:
            print(f"    Generation score {agg['avg_generation_score']:.2f} < {t['avg_generation_score']}")
        if agg["retrieval_pass_rate"] < t["retrieval_pass_rate"]:
            print(f"    Retrieval pass rate {agg['retrieval_pass_rate']:.0%} < {t['retrieval_pass_rate']:.0%}")

    print(f"\n{'═' * 55}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--client",
        default="global",
        help="Client namespace to evaluate against (default: global)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed per-question output",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Save full report to JSON file",
    )
    args = parser.parse_args()

    report = run_full_eval(client_id=args.client, verbose=args.verbose)
    print_report(report)

    # Save report
    output_path = args.output or f"data/eval_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Full report saved → {output_path}")

    # Exit code for CI
    sys.exit(0 if report["quality_gate_passed"] else 1)