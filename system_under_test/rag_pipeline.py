"""
rag_pipeline.py — The system under test (SUT).

By default this evaluates Project 1, the Self-Healing RAG pipeline, by importing
its ``answer_query`` entry point. That is the whole point of the pairing:

    Project 1  = "I built a production AI system"
    Project 2  = "I know how to measure and maintain it"   (this repo)

To evaluate a *different* system (a model swap, a new prompt, a competing RAG),
point ``SELF_HEALING_RAG_PATH`` at another checkout or edit ``_load_project_one``
below. Everything downstream — DeepEval, RAGAS, the cost tracker — consumes the
same ``{answer, sources, contexts}`` contract.

If Project 1 is not importable (e.g. running this repo standalone in CI without
the sibling checkout), the SUT degrades to a plain Groq completion so the
pipeline still demonstrates end-to-end without crashing.
"""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from typing import Any

# Where Project 1 lives relative to this repo. Override with SELF_HEALING_RAG_PATH.
_DEFAULT_PROJECT_ONE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "self_healing_rag",
)


@lru_cache(maxsize=1)
def _load_project_one() -> Any | None:
    """Import Project 1's ``answer_query``, or return None if unavailable."""
    project_path = os.getenv("SELF_HEALING_RAG_PATH", _DEFAULT_PROJECT_ONE)
    if project_path and project_path not in sys.path and os.path.isdir(project_path):
        sys.path.insert(0, project_path)

    try:
        from src.graph.graph import answer_query  # type: ignore

        return answer_query
    except Exception:
        return None


@lru_cache(maxsize=1)
def _baseline_client() -> Any:
    """A no-retrieval Groq baseline, used only if Project 1 is unavailable."""
    from langchain_groq import ChatGroq

    model = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
    return ChatGroq(model=model, temperature=0)


def _call_baseline(question: str) -> dict[str, Any]:
    client = _baseline_client()
    answer = client.invoke(
        f"You are a helpful assistant. Answer concisely and accurately.\n\nQuestion: {question}"
    ).content
    return {"answer": answer, "sources": [], "contexts": []}


def query(question: str) -> dict[str, Any]:
    """Run a question through the system under test.

    Returns a dict with:
        answer:   str        — the system's response
        sources:  list[str]  — cited source document names (RAG only)
        contexts: list[str]  — the actual retrieved chunk texts (for RAGAS)
    """
    answer_query = _load_project_one()
    if answer_query is not None:
        result = answer_query(question)
        return {
            "answer": result.get("answer", ""),
            "sources": result.get("sources", []),
            "contexts": result.get("contexts", []),
            "grounded": result.get("grounded", False),
            "retries": result.get("retries", 0),
        }
    return _call_baseline(question)
