"""
conftest.py — Shared test configuration.

These tests are deliberately offline: they exercise the pure logic of the eval
pipeline (thresholds, the CI gate, drift math, cost aggregation, dataset
integrity) with no API keys, no network, and no heavy ML dependencies. That is
what lets the CI 'unit' job stay green on every push without spending quota.

The real LLM-judge suites live under evals/tests/ and run only when keys exist.
"""

from __future__ import annotations

import os
import sys

# Ensure the repo root is importable as the package root.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
