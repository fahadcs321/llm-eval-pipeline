"""
run_ragas.py — RAGAS evaluation runner (RAG-specific metrics).

Runs the system under test over the golden dataset, capturing the *actual
retrieved contexts* (not just source filenames), then scores faithfulness,
answer relevancy, context recall, and context precision with RAGAS.

Driven by the configured judge provider (Groq by default) and local embeddings,
so it needs no OpenAI key. Writes an aggregate JSON the gate and dashboard read.

Usage:
    python evals/runners/run_ragas.py --output results/ragas_nightly.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

# Make the repo root importable when run as a script (python evals/runners/run_ragas.py),
# not only as a module (python -m evals.runners.run_ragas). The runners use absolute
# imports (evals.*, system_under_test.*), which need the repo root on sys.path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from evals.judge import _install_ragas_compat, get_ragas_models  # noqa: E402
from system_under_test.rag_pipeline import query as rag_query  # noqa: E402


def _get_int(name: str, default: int) -> int:
    """Read an int from the environment, falling back to ``default``."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _aggregate(scores: Any, metric: str) -> float:
    """Pull a single mean float for ``metric`` from a RAGAS result object."""
    try:
        df = scores.to_pandas()
    except Exception:  # pragma: no cover - depends on ragas version
        try:
            return float(scores[metric])
        except Exception:
            return 0.0
    if metric not in df.columns:
        return 0.0
    series = df[metric].dropna()
    return float(series.mean()) if len(series) else 0.0


def run_ragas_eval(golden_path: str, output_path: str, limit: int | None = None) -> dict[str, float]:
    with open(golden_path, encoding="utf-8") as f:
        goldens = json.load(f)

    if limit is not None and limit > 0:
        goldens = goldens[:limit]

    questions, answers, contexts, ground_truths = [], [], [], []

    print(f"Running RAGAS over {len(goldens)} golden pairs...")
    for i, item in enumerate(goldens, start=1):
        print(f"  [{i}/{len(goldens)}] {item['question'][:60]}")
        result = rag_query(item["question"])
        questions.append(item["question"])
        answers.append(result["answer"])
        # Prefer real retrieved chunk texts; fall back to source names.
        contexts.append(list(result.get("contexts") or result.get("sources") or []))
        ground_truths.append(item["ground_truth"])

    _install_ragas_compat()
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )
    from ragas.run_config import RunConfig

    # RAGAS answer_relevancy generates `strictness` paraphrased questions in a
    # single call via the LLM `n` parameter. Groq rejects n>1 ("'n': number must
    # be at most 1"), so pin strictness to 1 for Groq-compatible scoring. Raise
    # RAGAS_STRICTNESS on a provider that supports n>1 (e.g. OpenAI).
    answer_relevancy.strictness = _get_int("RAGAS_STRICTNESS", 1)

    dataset = Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        }
    )

    llm, embeddings = get_ragas_models()
    # Serialize judge calls by default so free-tier rate limits don't trip RAGAS's
    # 16-way concurrency into timeouts (faithfulness/recall/precision are the first
    # to fail). Raise RAGAS_MAX_WORKERS on a dev-tier key for a faster full sweep.
    run_config = RunConfig(
        max_workers=_get_int("RAGAS_MAX_WORKERS", 1),
        timeout=_get_int("RAGAS_TIMEOUT", 300),
    )
    print(
        f"Scoring with RAGAS (judge: Groq, embeddings: local, "
        f"max_workers={run_config.max_workers}, timeout={run_config.timeout}s)..."
    )
    scores = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
        llm=llm,
        embeddings=embeddings,
        run_config=run_config,
    )

    results = {
        "faithfulness": _aggregate(scores, "faithfulness"),
        "answer_relevancy": _aggregate(scores, "answer_relevancy"),
        "context_recall": _aggregate(scores, "context_recall"),
        "context_precision": _aggregate(scores, "context_precision"),
        "n_questions": float(len(goldens)),
    }

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\n── RAGAS Results ────────────────────────────────────────────")
    for key, value in results.items():
        print(f"  {key:>20}: {value:.3f}")
    print("──────────────────────────────────────────────────────────────")
    print(f"Written to {output_path}\n")
    return results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation.")
    parser.add_argument("--golden", default="evals/datasets/golden_rag.json")
    parser.add_argument("--output", default="results/ragas_nightly.json")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N golden pairs (useful for a quick, quota-safe demo).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_ragas_eval(args.golden, args.output, limit=args.limit)
