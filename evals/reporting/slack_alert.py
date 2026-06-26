"""
slack_alert.py — Post quality alerts to Slack via an incoming webhook.

Used by the drift detector and the nightly job to notify the team when a metric
regresses. No-ops cleanly when SLACK_WEBHOOK_URL is unset, so local and CI runs
without Slack configured don't fail.
"""

from __future__ import annotations

import os
from typing import Any


def send_alert(text: str, blocks: list[dict[str, Any]] | None = None) -> bool:
    """Post ``text`` to Slack. Returns True if sent, False if skipped/failed."""
    webhook = os.getenv("SLACK_WEBHOOK_URL")
    if not webhook:
        print("SLACK_WEBHOOK_URL not set — skipping Slack alert.")
        return False

    payload: dict[str, Any] = {"text": text}
    if blocks:
        payload["blocks"] = blocks

    try:
        import requests

        resp = requests.post(webhook, json=payload, timeout=10)
        resp.raise_for_status()
        print("Slack alert sent.")
        return True
    except Exception as exc:  # noqa: BLE001 - alerting must never crash the job
        print(f"Slack alert failed (non-blocking): {exc}")
        return False


def drift_alert(drifted_metrics: dict[str, tuple[float, float]]) -> bool:
    """Format and send a drift alert.

    Args:
        drifted_metrics: metric -> (baseline_value, current_value)
    """
    lines = ["*LLM Quality Drift Detected* — nightly eval regressed vs 7-day baseline:"]
    for metric, (baseline, current) in drifted_metrics.items():
        lines.append(f"• `{metric}`: {baseline:.3f} → {current:.3f}")
    lines.append("Check GitHub Actions for the full report.")
    return send_alert("\n".join(lines))


if __name__ == "__main__":
    send_alert("Test alert from the LLM eval pipeline.")
