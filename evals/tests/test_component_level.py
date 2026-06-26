"""
test_component_level.py — Per-node evaluation using DeepEval tracing.

Instead of only evaluating the final output, this evaluates EACH component
(retriever, generator) separately, so you can pinpoint exactly which node
caused a quality failure.

This is the 2026 standard — component-level evals via @observe decorators.
"""

import json

from deepeval.dataset import EvaluationDataset, Golden
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase
from deepeval.tracing import observe, update_current_span

with open("evals/datasets/golden_rag.json") as f:
    goldens_raw = json.load(f)


# ── Wrap each component with @observe ─────────────────────────────────────────
# This tells DeepEval to score each component individually.
# You'll see retriever score and generator score separately in Confident AI.


@observe(metrics=[FaithfulnessMetric(threshold=0.80)])
def retriever_component(query: str, retriever) -> list[str]:
    """Evaluate only the retriever: did it pull relevant chunks?"""
    results = retriever.search(query)
    context = [r.page_content for r in results]

    update_current_span(
        test_case=LLMTestCase(
            input=query,
            actual_output="\n".join(context),
        )
    )
    return context


@observe(metrics=[AnswerRelevancyMetric(threshold=0.75)])
def generator_component(query: str, context: list[str], llm) -> str:
    """Evaluate only the generator: is the answer relevant to the question?"""
    answer = llm.invoke(f"Context: {context}\nQuestion: {query}")

    update_current_span(
        test_case=LLMTestCase(
            input=query,
            actual_output=answer,
        )
    )
    return answer


# ── Run component-level evaluation ───────────────────────────────────────────
def test_component_level_eval():
    """
    Runs retriever and generator separately so failures can be attributed
    to the right component. Much more useful for debugging than end-to-end only.
    """
    # Import your actual system components here
    # from system_under_test.rag_pipeline import retriever, llm

    dataset = EvaluationDataset(goldens=[Golden(input=g["question"]) for g in goldens_raw[:20]])

    for _golden in dataset.evals_iterator():
        # These @observe-decorated calls automatically create DeepEval spans
        # context = retriever_component(_golden.input, retriever)
        # answer = generator_component(_golden.input, context, llm)
        pass

    # Component scores appear in Confident AI dashboard,
    # showing retriever faithfulness and generator relevancy separately
