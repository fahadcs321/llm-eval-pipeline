"""
run_deepeval.py — Programmatic DeepEval runner.

The pytest suite (evals/tests/test_rag_quality.py) is what blocks a PR. This
runner is its batch sibling: it scores the same metrics with DeepEval's
``evaluate()`` API, aggregates them into a single JSON, and derives a
hallucination *rate* — the numbers the gate, Langfuse, and the dashboard consume.

All scoring uses the configured judge (Groq by default), so no OpenAI key.

Usage:
    python evals/runners/run_deepeval.py --output results/deepeval_nightly.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evals.judge import get_deepeval_judge
from system_under_test.rag_pipeline import query as rag_query

# A faithfulness score below this counts a sample as a hallucination.
HALLUCINATION_THRESHOLD = 0.5


def _build_test_cases(goldens: list[dict[str, Any]]) -> list[Any]:
    from deepeval.test_case import LLMTestCase

    cases = []
    for i, item in enumerate(goldens, start=1):
        print(f"  [{i}/{len(goldens)}] {item['question'][:60]}")
        result = rag_query(item["question"])
        context = list(result.get("contexts") or result.get("sources") or [])
        cases.append(
            LLMTestCase(
                input=item["question"],
                actual_output=result["answer"],
                expected_output=item["ground_truth"],
                retrieval_context=context or None,
            )
        )
    return cases


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def run_deepeval(golden_path: str, output_path: str) -> dict[str, float]:
    with open(golden_path, encoding="utf-8") as f:
        goldens = json.load(f)

    from deepeval import evaluate
    from deepeval.metrics import (
        AnswerRelevancyMetric,
        FaithfulnessMetric,
    )

    from evals.metrics.custom_metrics import build_on_topic_metric

    judge = get_deepeval_judge()
    metrics = [
        AnswerRelevancyMetric(threshold=0.75, model=judge),
        FaithfulnessMetric(threshold=0.80, model=judge),
        build_on_topic_metric(judge),
    ]

    print(f"Running DeepEval over {len(goldens)} golden pairs...")
    test_cases = _build_test_cases(goldens)

    print("Scoring with DeepEval (judge: Groq)...")
    eval_result = evaluate(test_cases=test_cases, metrics=metrics)

    # Aggregate per-metric means and a hallucination rate from faithfulness.
    by_name: dict[str, list[float]] = {}
    faithfulness_scores: list[float] = []
    for tr in eval_result.test_results:
        for md in tr.metrics_data or []:
            by_name.setdefault(md.name, []).append(float(md.score or 0.0))
            if "faithful" in md.name.lower():
                faithfulness_scores.append(float(md.score or 0.0))

    def pick(*needles: str) -> float:
        for name, scores in by_name.items():
            if any(n in name.lower() for n in needles):
                return _mean(scores)
        return 0.0

    results = {
        "answer_relevancy": pick("relevancy"),
        "faithfulness": pick("faithful"),
        "geval_on_topic": pick("on-topic", "on topic", "topic"),
        "hallucination_rate": (
            _mean([1.0 if s < HALLUCINATION_THRESHOLD else 0.0 for s in faithfulness_scores])
        ),
        "n_questions": float(len(goldens)),
    }

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\n── DeepEval Results ─────────────────────────────────────────")
    for key, value in results.items():
        print(f"  {key:>20}: {value:.3f}")
    print("──────────────────────────────────────────────────────────────")
    print(f"Written to {output_path}\n")
    return results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DeepEval batch evaluation.")
    parser.add_argument("--golden", default="evals/datasets/golden_rag.json")
    parser.add_argument("--output", default="results/deepeval_nightly.json")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_deepeval(args.golden, args.output)
