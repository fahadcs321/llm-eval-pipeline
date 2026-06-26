"""
test_rag_quality.py — DeepEval pytest suite for the RAG system under test.

This is the suite that BLOCKS a PR. One parametrized test per golden Q&A pair;
``assert_test`` fails if any metric falls below its threshold, and the non-zero
exit code is what GitHub Actions reads to block the merge.

Scored by the configured judge (Groq by default), so no OpenAI key. The whole
module skips cleanly when no judge key is present, so it never runs — or pays —
during the offline unit job.

Run locally:   deepeval test run evals/tests/test_rag_quality.py
Run in CI:     deepeval test run evals/tests/test_rag_quality.py
"""

from __future__ import annotations

import json
import time

import pytest
from deepeval import assert_test
from deepeval.metrics import (
    AnswerRelevancyMetric,
    BiasMetric,
    FaithfulnessMetric,
    ToxicityMetric,
)
from deepeval.test_case import LLMTestCase

from evals.config import settings
from evals.judge import get_deepeval_judge
from evals.metrics.custom_metrics import build_on_topic_metric
from system_under_test.rag_pipeline import query as rag_query

pytestmark = pytest.mark.skipif(
    not settings.judge_api_key,
    reason="No judge API key set — skipping live LLM-judge eval.",
)

# How many goldens to score in the fast PR gate. Override with EVAL_SAMPLE_SIZE.
import os  # noqa: E402

_SAMPLE = int(os.getenv("EVAL_SAMPLE_SIZE", "20"))


def _build_cases() -> tuple[list[LLMTestCase], list[float]]:
    with open("evals/datasets/golden_rag.json", encoding="utf-8") as f:
        goldens = json.load(f)[:_SAMPLE]

    cases: list[LLMTestCase] = []
    latencies: list[float] = []
    for g in goldens:
        t0 = time.time()
        result = rag_query(g["question"])
        latencies.append((time.time() - t0) * 1000)
        cases.append(
            LLMTestCase(
                input=g["question"],
                actual_output=result["answer"],
                expected_output=g["ground_truth"],
                retrieval_context=list(result.get("contexts") or result.get("sources") or [])
                or None,
            )
        )
    return cases, latencies


# Built once per session only when the module is not skipped.
if settings.judge_api_key:
    _JUDGE = get_deepeval_judge()
    _CASES, _LATENCIES = _build_cases()
    RAG_METRICS = [
        AnswerRelevancyMetric(threshold=0.75, model=_JUDGE),
        FaithfulnessMetric(threshold=0.80, model=_JUDGE),
        BiasMetric(threshold=0.10, model=_JUDGE),
        ToxicityMetric(threshold=0.02, model=_JUDGE),
        build_on_topic_metric(_JUDGE),
    ]
else:  # pragma: no cover - module is skipped in this case
    _CASES, _LATENCIES, RAG_METRICS = [], [], []


@pytest.mark.parametrize("test_case", _CASES)
def test_rag_quality(test_case: LLMTestCase):
    """Evaluate one golden Q&A pair against every metric threshold."""
    assert_test(test_case, RAG_METRICS)


def test_latency_p95():
    """P95 latency must stay under 3000ms."""
    if not _LATENCIES:
        pytest.skip("No latency data recorded.")
    ordered = sorted(_LATENCIES)
    p95 = ordered[min(int(0.95 * len(ordered)), len(ordered) - 1)]
    assert p95 < 3000, f"P95 latency {p95:.0f}ms exceeds 3000ms threshold"
