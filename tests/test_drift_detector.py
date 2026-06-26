"""Tests for the drift detector's baseline math and decision logic."""

from __future__ import annotations

import json

import pytest

from scripts import drift_detector


def test_compute_baseline_averages_metrics():
    recent = [
        {"faithfulness": 0.80, "answer_relevancy": 0.70, "_date": "x"},
        {"faithfulness": 0.90, "answer_relevancy": 0.80, "_date": "y"},
    ]
    baseline = drift_detector.compute_baseline(recent)
    assert baseline["faithfulness"] == pytest.approx(0.85)
    assert baseline["answer_relevancy"] == pytest.approx(0.75)


def test_compute_baseline_empty_is_empty():
    assert drift_detector.compute_baseline([]) == {}


def _write_current(tmp_path, data):
    p = tmp_path / "current.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return str(p)


def test_no_drift_when_metrics_hold(tmp_path, monkeypatch):
    baseline_runs = [{"faithfulness": 0.85} for _ in range(4)]
    monkeypatch.setattr(drift_detector, "load_recent_results", lambda *a, **k: baseline_runs)
    current = _write_current(tmp_path, {"faithfulness": 0.86})
    assert drift_detector.check_drift(current) is True


def test_drift_flagged_when_quality_drops(tmp_path, monkeypatch):
    baseline_runs = [{"faithfulness": 0.85} for _ in range(4)]
    monkeypatch.setattr(drift_detector, "load_recent_results", lambda *a, **k: baseline_runs)
    # A >5% drop in a higher-is-better metric must flag drift.
    current = _write_current(tmp_path, {"faithfulness": 0.60})
    assert drift_detector.check_drift(current) is False


def test_insufficient_history_passes(tmp_path, monkeypatch):
    monkeypatch.setattr(
        drift_detector, "load_recent_results", lambda *a, **k: [{"faithfulness": 0.8}]
    )
    current = _write_current(tmp_path, {"faithfulness": 0.10})
    # Not enough history (< 3) → cannot judge drift → pass.
    assert drift_detector.check_drift(current) is True
