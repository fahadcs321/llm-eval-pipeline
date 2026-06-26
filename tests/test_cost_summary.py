"""Tests for the cost tracker's aggregation math (no LLM calls)."""

from __future__ import annotations

from evals.runners.run_cost import CostTracker


def test_summary_on_empty_tracker():
    s = CostTracker().summary()
    assert s["n_queries"] == 0
    assert s["total_cost_usd"] == 0.0
    assert s["latency_p50_ms"] == 0
    assert s["latency_p95_ms"] == 0


def test_summary_aggregates_manual_data():
    t = CostTracker()
    # Simulate three tracked queries without calling any LLM.
    t.total_cost = 0.003
    t.total_tokens = 1500
    t.n_queries = 3
    t.latencies = [100.0, 200.0, 300.0]

    s = t.summary()
    assert s["n_queries"] == 3
    assert s["avg_cost_per_query_usd"] == round(0.003 / 3, 6)
    assert s["avg_tokens_per_query"] == 500
    assert s["latency_p95_ms"] >= s["latency_p50_ms"]
