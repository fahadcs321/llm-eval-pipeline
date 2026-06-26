"""Tests for the CI quality gate (evals/gate.py)."""

from __future__ import annotations

import json

from evals.gate import evaluate_gates, run_gate

PASSING = {
    "faithfulness": 0.87,
    "answer_relevancy": 0.83,
    "context_recall": 0.78,
    "context_precision": 0.71,
    "hallucination_rate": 0.03,
}

FAILING = {**PASSING, "faithfulness": 0.55}


def test_passing_results_pass():
    passed, report = evaluate_gates(PASSING)
    assert passed
    assert any("PASS — merge allowed" in line for line in report)


def test_one_bad_metric_blocks_merge():
    passed, report = evaluate_gates(FAILING)
    assert not passed
    assert any("FAIL — merge blocked" in line for line in report)


def test_unknown_keys_are_ignored():
    passed, _ = evaluate_gates({**PASSING, "totally_unknown": 0.0})
    assert passed


def test_layer_filter_skips_nightly_only_metrics():
    # context_recall is nightly-only; a bad value should NOT block a PR-layer gate.
    results = {**PASSING, "context_recall": 0.10}
    pr_passed, _ = evaluate_gates(results, layer="pr")
    nightly_passed, _ = evaluate_gates(results, layer="nightly")
    assert pr_passed
    assert not nightly_passed


def test_run_gate_reads_file(tmp_path):
    p = tmp_path / "results.json"
    p.write_text(json.dumps(PASSING), encoding="utf-8")
    assert run_gate(str(p)) is True

    p.write_text(json.dumps(FAILING), encoding="utf-8")
    assert run_gate(str(p)) is False
