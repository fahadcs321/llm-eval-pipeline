"""
promote_failures.py — Pull failing production traces from Langfuse
and promote them into the golden dataset.

This keeps the eval dataset calibrated to ACTUAL production failures,
not just synthetic ones. Run weekly via cron.

This is the 2026 standard: the golden dataset is a living document,
not a frozen snapshot from launch day.

Usage:
  python scripts/promote_failures.py --days 7 --output evals/datasets/golden_edge.json
"""

import argparse
import json
import os
from datetime import datetime, timedelta
from pathlib import Path

from langfuse import Langfuse


def fetch_failing_traces(days: int = 7, min_score: float = 0.5) -> list[dict]:
    """
    Pull traces from Langfuse where hallucination or faithfulness scores
    were below threshold, indicating a real production failure.
    """
    client = Langfuse(
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        host=os.getenv("LANGFUSE_HOST", "http://localhost:3000"),
    )

    cutoff = datetime.now() - timedelta(days=days)

    print(f"Fetching failing traces from last {days} days...")

    # Fetch traces with low scores
    traces = client.fetch_traces(
        from_timestamp=cutoff,
        tags=["rag", "production"],
        limit=500,
    ).data

    failing = []
    for trace in traces:
        # Get scores attached to this trace
        scores = {s.name: s.value for s in (trace.scores or [])}

        faithfulness = scores.get("faithfulness", 1.0)
        hallucination = scores.get("hallucination_rate", 0.0)

        # A trace is a "production failure" if it scored badly
        is_failure = faithfulness < 0.65 or hallucination > 0.15

        if is_failure and trace.input and trace.output:
            failing.append(
                {
                    "question": trace.input.get("question", ""),
                    "actual_answer": trace.output.get("answer", ""),
                    "ground_truth": "",  # needs human review to fill
                    "context_source": "production_failure",
                    "failure_scores": {
                        "faithfulness": faithfulness,
                        "hallucination": hallucination,
                    },
                    "trace_id": trace.id,
                    "date": trace.timestamp.isoformat() if trace.timestamp else "",
                }
            )

    print(f"Found {len(failing)} production failures to review")
    return failing


def merge_into_golden(new_items: list[dict], output_path: str):
    """
    Merge new failure cases into the existing edge-case golden set.
    Deduplicates by question text.
    """
    existing = []
    if Path(output_path).exists():
        with open(output_path) as f:
            existing = json.load(f)

    existing_questions = {e["question"].strip().lower() for e in existing}

    added = 0
    for item in new_items:
        q = item["question"].strip().lower()
        if q and q not in existing_questions:
            existing.append(item)
            existing_questions.add(q)
            added += 1

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)

    print(f"Added {added} new failure cases -> {output_path}")
    print(f"Total edge cases in dataset: {len(existing)}")
    print("\n⚠️  Review the new entries and fill in 'ground_truth' before")
    print("   they're used in the eval suite. Unfilled entries are skipped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=7, help="Look back N days for failing traces")
    parser.add_argument(
        "--output",
        default="evals/datasets/golden_edge.json",
        help="Output path for edge-case golden set",
    )
    args = parser.parse_args()

    Path(args.output).parent.mkdir(exist_ok=True, parents=True)
    failures = fetch_failing_traces(days=args.days)
    merge_into_golden(failures, args.output)
