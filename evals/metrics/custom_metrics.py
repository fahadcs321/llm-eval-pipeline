"""
custom_metrics.py — G-Eval custom metrics with plain-English criteria.

G-Eval uses chain-of-thought LLM-as-judge scoring and shows ~81% correlation
with human ratings in 2026 benchmarks — the practical default for subjective
checks a deterministic metric can't capture (on-topic, conciseness, citation).

These are exposed as *factories* that take a judge ``model`` so the same metric
can be scored by Groq, OpenAI, or any DeepEval-compatible judge. Pass the judge
from ``evals.judge.get_deepeval_judge()``; omit it to use DeepEval's default.
"""

from __future__ import annotations

from typing import Any

from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams


def build_on_topic_metric(model: Any = None) -> GEval:
    """Answer stays within the system's intended scope (no off-topic drift)."""
    return GEval(
        name="On-Topic",
        criteria=(
            "Determine whether the 'actual output' stays within the intended scope "
            "of the system. It should answer the 'input' question without going "
            "off-topic, discussing unrelated subjects, or volunteering unsolicited "
            "opinions."
        ),
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
        ],
        threshold=0.80,
        model=model,
    )


def build_conciseness_metric(model: Any = None) -> GEval:
    """Answer is appropriately concise — no padding or repetition."""
    return GEval(
        name="Conciseness",
        criteria=(
            "Determine whether the 'actual output' is appropriately concise. It "
            "should not repeat itself, pad with filler sentences, or include "
            "information irrelevant to the 'input' question."
        ),
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
        ],
        threshold=0.75,
        model=model,
    )


def build_citation_accuracy_metric(model: Any = None) -> GEval:
    """Every factual claim is attributable to the retrieved context."""
    return GEval(
        name="Citation Accuracy",
        criteria=(
            "Determine whether every factual claim in the 'actual output' is "
            "attributable to the documents provided in 'retrieval_context'. The "
            "output should not cite sources absent from the retrieval context or "
            "attribute claims to the wrong source."
        ),
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.RETRIEVAL_CONTEXT,
        ],
        threshold=0.80,
        model=model,
    )
