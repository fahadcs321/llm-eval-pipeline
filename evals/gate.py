"""
gate.py — The CI quality gate.

Reads an eval results JSON, checks every metric present against thresholds.py,
prints a report, and exits non-zero if any gate fails. This is the script that
turns "the eval scored X" into "the merge is blocked" — GitHub Actions reads the
exit code.

Deliberately dependency-free (stdlib + thresholds only) so it is fast and fully
unit-testable offline, with no API keys.

Usage:
    python -m evals.gate --results results/eval_results.json
"""

from __future__ import annotations

import argparse
import json
import sys

from evals.metrics.thresholds import THRESHOLDS, check_metric


def evaluate_gates(results: dict[str, float], layer: str | None = None) -> tuple[bool, list[str]]:
    """Check every threshold-governed metric present in ``results``.

    Args:
        results: metric name -> value (extra keys are ignored).
        layer: if given ("pr" | "nightly"), only enforce gates for that CI layer.
               A metric's ``ci_layer`` of "both" always applies.

    Returns (all_passed, report_lines).
    """
    lines: list[str] = []
    lines.append("── CI Quality Gate ──────────────────────────────────────────")
    all_passed = True
    checked = 0

    for metric, spec in THRESHOLDS.items():
        if metric not in results:
            continue
        if layer and spec.get("ci_layer", "both") not in ("both", layer):
            continue

        passed, message = check_metric(metric, float(results[metric]))
        lines.append("  " + message)
        checked += 1
        all_passed = all_passed and passed

    if checked == 0:
        lines.append("  (no threshold-governed metrics found in results)")
    lines.append("─────────────────────────────────────────────────────────────")
    verdict = "PASS — merge allowed" if all_passed else "FAIL — merge blocked"
    lines.append(f"  Result: {verdict}")
    return all_passed, lines


def run_gate(results_path: str, layer: str | None = None) -> bool:
    with open(results_path, encoding="utf-8") as f:
        results = json.load(f)

    passed, report = evaluate_gates(results, layer=layer)
    print("\n".join(report))
    return passed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM eval CI quality gate.")
    parser.add_argument("--results", default="results/eval_results.json")
    parser.add_argument(
        "--layer",
        choices=["pr", "nightly"],
        default=None,
        help="Only enforce gates for this CI layer (default: all).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    ok = run_gate(args.results, layer=args.layer)
    sys.exit(0 if ok else 1)
