"""
langfuse_logger.py — Write eval scores to Langfuse after every run.
This feeds the 7-day rolling baseline used by drift_detector.py
and gives you a searchable, traceable history of every eval run.

Self-hosted Langfuse: docker compose up (2 commands, MIT licensed)
Cloud Langfuse: langfuse.com (free tier available)
"""

import json
import os
from datetime import datetime

from langfuse import Langfuse


def get_client() -> Langfuse:
    return Langfuse(
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        host=os.getenv("LANGFUSE_HOST", "http://localhost:3000"),
    )


def log_eval_run(
    results: dict,
    run_type: str = "nightly",  # "pr_gate" | "nightly" | "canary"
    model: str = "gpt-4o-mini",
    git_sha: str = "",
):
    """
    Log a complete eval run result to Langfuse as a scored trace.
    Each metric becomes a named score on the trace.
    """
    client = get_client()

    # Create a trace representing this eval run
    trace = client.trace(
        name=f"eval_run_{run_type}",
        tags=[run_type, "eval", model],
        metadata={
            "run_type": run_type,
            "model": model,
            "git_sha": git_sha or os.getenv("GITHUB_SHA", "local"),
            "timestamp": datetime.now().isoformat(),
            "n_questions": results.get("n_questions", 0),
        },
    )

    # Log each metric as a named score
    # This enables time-series queries in Langfuse: "faithfulness over last 30 days"
    metric_fields = [
        "faithfulness",
        "answer_relevancy",
        "context_recall",
        "context_precision",
        "hallucination_rate",
        "bias_score",
        "toxicity_score",
        "task_completion",
        "geval_on_topic",
        "latency_p95_ms",
        "cost_per_query_usd",
        "total_cost_usd",
    ]

    for metric in metric_fields:
        if metric in results:
            client.score(
                trace_id=trace.id,
                name=metric,
                value=float(results[metric]),
            )

    client.flush()
    print(f"Logged eval run to Langfuse (trace: {trace.id})")
    return trace.id


def log_eval_run_from_files(
    ragas_path: str = "results/ragas_nightly.json",
    cost_path: str = "results/cost_nightly.json",
    run_type: str = "nightly",
):
    """Convenience: load result files and log both to Langfuse together."""
    results = {}

    try:
        with open(ragas_path) as f:
            results.update(json.load(f))
    except FileNotFoundError:
        print(f"⚠️  {ragas_path} not found, skipping RAGAS metrics")

    try:
        with open(cost_path) as f:
            results.update(json.load(f))
    except FileNotFoundError:
        print(f"⚠️  {cost_path} not found, skipping cost metrics")

    if results:
        return log_eval_run(results, run_type=run_type)
    else:
        print("No results to log")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ragas", default="results/ragas_nightly.json")
    parser.add_argument("--cost", default="results/cost_nightly.json")
    parser.add_argument("--run-type", default="nightly")
    args = parser.parse_args()
    log_eval_run_from_files(args.ragas, args.cost, args.run_type)
