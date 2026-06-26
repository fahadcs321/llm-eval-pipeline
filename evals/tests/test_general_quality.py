"""
test_general_quality.py — DeepEval suite for the bare (non-RAG) LLM app.

The control group. Evaluates the no-retrieval baseline against general-knowledge
goldens, scoring relevancy and on-topic-ness — the metrics a model can earn
without retrieval. Comparing this against test_rag_quality shows what retrieval
buys you.

Skips cleanly with no judge key, so the offline unit job never runs it.
"""

from __future__ import annotations

import json

import pytest
from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase

from evals.config import settings
from evals.judge import get_deepeval_judge
from evals.metrics.custom_metrics import build_on_topic_metric
from system_under_test.llm_app import query as llm_query

pytestmark = pytest.mark.skipif(
    not settings.judge_api_key,
    reason="No judge API key set — skipping live LLM-judge eval.",
)


def _build_cases() -> list[LLMTestCase]:
    with open("evals/datasets/golden_general.json", encoding="utf-8") as f:
        goldens = json.load(f)
    cases = []
    for g in goldens:
        result = llm_query(g["question"])
        cases.append(
            LLMTestCase(
                input=g["question"],
                actual_output=result["answer"],
                expected_output=g["ground_truth"],
            )
        )
    return cases


if settings.judge_api_key:
    _JUDGE = get_deepeval_judge()
    _CASES = _build_cases()
    GENERAL_METRICS = [
        AnswerRelevancyMetric(threshold=0.75, model=_JUDGE),
        build_on_topic_metric(_JUDGE),
    ]
else:  # pragma: no cover
    _CASES, GENERAL_METRICS = [], []


@pytest.mark.parametrize("test_case", _CASES)
def test_general_quality(test_case: LLMTestCase):
    assert_test(test_case, GENERAL_METRICS)
