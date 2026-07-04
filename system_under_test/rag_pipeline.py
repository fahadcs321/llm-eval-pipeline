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


def _ensure_project_one_path() -> None:
    """Put Project 1's checkout on sys.path so its packages import."""
    project_path = os.getenv("SELF_HEALING_RAG_PATH", _DEFAULT_PROJECT_ONE)
    if project_path and project_path not in sys.path and os.path.isdir(project_path):
        sys.path.insert(0, project_path)


@lru_cache(maxsize=1)
def _load_project_one() -> Any | None:
    """Import Project 1's ``answer_query``, or return None if unavailable."""
    _ensure_project_one_path()
    try:
        from src.graph.graph import answer_query  # type: ignore

        return answer_query
    except Exception:
        return None


@lru_cache(maxsize=1)
def _load_news_sut() -> Any | None:
    """Import the multilingual news RAG entry point (SUT_MODE=news).

    Lives on Project 1's ``retriever-scale-demo`` branch under
    ``scripts/retriever_demo`` and answers over the hybrid, multilingual news
    index. Returns None if that branch/index isn't present, so the eval degrades
    cleanly to the default SUT.
    """
    _ensure_project_one_path()
    # Import as top-level ``retriever_demo`` (Project 1's scripts/ dir on the path),
    # NOT ``scripts.retriever_demo`` — this repo has its own ``scripts`` package that
    # would otherwise shadow it.
    project_path = os.getenv("SELF_HEALING_RAG_PATH", _DEFAULT_PROJECT_ONE)
    scripts_dir = os.path.join(project_path, "scripts")
    if os.path.isdir(scripts_dir) and scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    try:
        from retriever_demo.rag_over_news import answer_over_news  # type: ignore

        return answer_over_news
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
    # SUT_MODE=news evaluates the multilingual news RAG (hybrid retrieval over the
    # 10k-article index) instead of the default Self-Healing RAG.
    if os.getenv("SUT_MODE", "").lower() == "news":
        answer_over_news = _load_news_sut()
        if answer_over_news is not None:
            result = answer_over_news(question)
            return {
                "answer": result.get("answer", ""),
                "sources": result.get("sources", []),
                "contexts": result.get("contexts", []),
                "grounded": result.get("grounded", False),
                "retries": result.get("retries", 0),
            }

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
