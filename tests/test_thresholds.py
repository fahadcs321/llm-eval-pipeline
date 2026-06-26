"""Tests for the threshold definitions and per-metric checks."""

from __future__ import annotations

from evals.metrics.thresholds import THRESHOLDS, check_metric


def test_every_threshold_is_well_formed():
    for name, spec in THRESHOLDS.items():
        assert spec["direction"] in ("higher_is_better", "lower_is_better"), name
        # Exactly one bound matching the direction.
        if spec["direction"] == "higher_is_better":
            assert "min" in spec, name
        else:
            assert "max" in spec, name
        assert spec.get("ci_layer") in ("both", "pr", "nightly"), name


def test_higher_is_better_passes_above_floor():
    passed, msg = check_metric("faithfulness", 0.92)
    assert passed
    assert "PASS" in msg


def test_higher_is_better_fails_below_floor():
    passed, _ = check_metric("faithfulness", 0.50)
    assert not passed


def test_lower_is_better_passes_below_ceiling():
    passed, _ = check_metric("hallucination_rate", 0.01)
    assert passed


def test_lower_is_better_fails_above_ceiling():
    passed, _ = check_metric("hallucination_rate", 0.20)
    assert not passed


def test_boundary_is_inclusive():
    # value exactly equal to the threshold must pass
    floor = THRESHOLDS["answer_relevancy"]["min"]
    assert check_metric("answer_relevancy", floor)[0]
    ceiling = THRESHOLDS["toxicity_score"]["max"]
    assert check_metric("toxicity_score", ceiling)[0]


def test_unknown_metric_is_not_gated():
    passed, msg = check_metric("made_up_metric", 999.0)
    assert passed
    assert "No threshold" in msg
