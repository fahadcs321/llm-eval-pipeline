"""
llm_app.py — A plain (non-RAG) LLM app, used as a general-quality baseline.

The RAG pipeline (rag_pipeline.py) is the primary system under test. This module
is the control: a bare LLM with no retrieval. Evaluating both against the same
goldens shows exactly what retrieval buys you — the faithfulness and context
metrics that a naked model cannot earn.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

_SYSTEM_PROMPT = "You are a helpful assistant. Answer concisely and accurately."


@lru_cache(maxsize=1)
def _client() -> Any:
    from langchain_groq import ChatGroq

    model = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
    return ChatGroq(model=model, temperature=0)


def query(question: str) -> dict[str, Any]:
    """Answer with a bare LLM — no retrieval, no sources."""
    answer = _client().invoke(f"{_SYSTEM_PROMPT}\n\nQuestion: {question}").content
    return {"answer": answer, "sources": [], "contexts": []}
