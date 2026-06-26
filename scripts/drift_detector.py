"""
drift_detector.py — Detects metric drift vs rolling 7-day baseline.
Called by the nightly GitHub Actions job.
Sends Slack alert and exits 1 if any metric degrades > 5%.

This is the 2026 standard: the baseline is a rolling 7-day observation,
not a frozen number. Models drift, prompts drift, datasets drift.
"""

import glob
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

from evals.metrics.thresholds import DRIFT_THRESHOLD_PCT


def load_recent_results(results_dir: str = "results", days: int = 7) -> list[dict]:
    """Load all nightly result files from the past N days."""
    cutoff = datetime.now() - timedelta(days=days)
    results = []

    for path in sorted(glob.glob(f"{results_dir}/ragas_*.json")):
        try:
            mtime = datetime.fromtimestamp(Path(path).stat().st_mtime)
            if mtime >= cutoff:
                with open(path) as f:
                    data = json.load(f)
                    data["_date"] = mtime.isoformat()
                    results.append(data)
        except Exception:
            continue

    return results


def compute_baseline(results: list[dict]) -> dict:
    """Average each metric across the N recent results."""
    if not results:
        return {}

    metrics = [k for k in results[0].keys() if not k.startswith("_")]
    baseline = {}

    for metric in metrics:
        values = [r[metric] for r in results if isinstance(r.get(metric), float)]
        if values:
            baseline[metric] = sum(values) / len(values)

    return baseline


def check_drift(current_path: str) -> bool:
    """
    Compare current nightly results against 7-day baseline.
    Returns True if all metrics are within acceptable drift.
    """
    with open(current_path) as f:
        current = json.load(f)

    recent = load_recent_results()

    if len(recent) < 3:
        print("⚠️  Not enough historical data for drift detection (need >= 3 days)")
        return True  # pass, not enough history yet

    baseline = compute_baseline(recent)

    print(f"\n── Drift Detection (vs {len(recent)}-day baseline) ──────────────────")
    print(f"{'Metric':<30} {'Baseline':>10} {'Current':>10} {'Change':>10} {'Status':>8}")
    print("-" * 72)

    any_drift = False

    for metric, baseline_val in baseline.items():
        current_val = current.get(metric)
        if current_val is None or not isinstance(current_val, float):
            continue

        change = current_val - baseline_val
        change_pct = abs(change) / max(baseline_val, 0.001)

        # For lower-is-better metrics, an increase is bad
        # For higher-is-better, a decrease is bad
        is_bad_direction = (
            change > 0
            and metric
            in [
                "hallucination_rate",
                "bias_score",
                "toxicity_score",
                "latency_p95_ms",
                "cost_per_query_usd",
            ]
        ) or (
            change < 0
            and metric
            in [
                "faithfulness",
                "answer_relevancy",
                "context_recall",
                "context_precision",
                "task_completion",
                "geval_on_topic",
            ]
        )

        drifted = is_bad_direction and change_pct > DRIFT_THRESHOLD_PCT

        status = "🔴 DRIFT" if drifted else "✅ OK"
        print(
            f"{metric:<30} {baseline_val:>10.3f} {current_val:>10.3f} {change:>+10.3f} {status:>8}"
        )

        if drifted:
            any_drift = True

    print("─" * 72 + "\n")

    return not any_drift


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--current", default="results/ragas_nightly.json")
    args = parser.parse_args()

    passed = check_drift(args.current)

    if not passed:
        from evals.reporting.slack_alert import send_alert

        send_alert(
            "*LLM Quality Drift Detected* — nightly eval shows metric regression "
            "> 5% vs 7-day baseline. Check GitHub Actions for details."
        )
        print("Drift detected. Slack alert attempted.")
        sys.exit(1)
    else:
        print("No drift detected. All metrics within baseline.")
        sys.exit(0)
