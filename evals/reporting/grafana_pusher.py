"""
grafana_pusher.py — Push eval metrics to Grafana via the HTTP API.
Creates time-series data points that feed the rolling dashboard.

After running nightly eval:
  python evals/reporting/grafana_pusher.py

The dashboard (dashboard/grafana_dashboard.json) reads from this datasource.
"""

import json
import os
import time
from datetime import datetime

import requests

GRAFANA_URL = os.getenv("GRAFANA_URL", "http://localhost:3001")
GRAFANA_API_KEY = os.getenv("GRAFANA_API_KEY", "")


def push_metrics(results: dict, run_type: str = "nightly"):
    """
    Push metrics to Grafana using the Annotations or Push API.
    Uses Grafana's simple JSON push format.
    """
    if not GRAFANA_API_KEY:
        print("⚠️  GRAFANA_API_KEY not set. Skipping Grafana push.")
        return

    timestamp_ms = int(time.time() * 1000)
    headers = {
        "Authorization": f"Bearer {GRAFANA_API_KEY}",
        "Content-Type": "application/json",
    }

    # Push each metric as an annotation (simple, works on any Grafana)
    annotation_text = " | ".join(
        f"{k}: {v:.3f}" if isinstance(v, float) else f"{k}: {v}"
        for k, v in results.items()
        if not k.startswith("_")
    )

    annotation_payload = {
        "text": f"[{run_type}] {annotation_text}",
        "tags": ["eval", run_type],
        "time": timestamp_ms,
    }

    try:
        resp = requests.post(
            f"{GRAFANA_URL}/api/annotations",
            headers=headers,
            json=annotation_payload,
            timeout=10,
        )
        resp.raise_for_status()
        print("✅ Pushed metrics annotation to Grafana")
    except requests.RequestException as e:
        print(f"⚠️  Grafana push failed: {e}")

    # Also print a clean summary for the CI log
    print("\n── Grafana Push Summary ─────────────────────────────────────")
    print(f"  Run type:  {run_type}")
    print(f"  Timestamp: {datetime.now().isoformat()}")
    for k, v in results.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.3f}")
    print("─────────────────────────────────────────────────────────────\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ragas", default="results/ragas_nightly.json")
    parser.add_argument("--cost", default="results/cost_nightly.json")
    parser.add_argument("--run-type", default="nightly")
    args = parser.parse_args()

    results = {}
    for path in [args.ragas, args.cost]:
        try:
            with open(path) as f:
                results.update(json.load(f))
        except FileNotFoundError:
            pass

    push_metrics(results, run_type=args.run_type)
