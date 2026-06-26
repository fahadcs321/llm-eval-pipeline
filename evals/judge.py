"""
judge.py — The LLM-as-judge, wired to the configured provider (Groq by default).

DeepEval and RAGAS both default to OpenAI for their judge model and embeddings.
This module points them at the project's configured chat model instead (Groq),
plus a local sentence-transformers embedding model, so the entire eval pipeline
runs with no OpenAI key.

Three entry points:
  - get_chat_model()  -> a LangChain chat model for the judge provider
  - get_deepeval_judge() -> a DeepEvalBaseLLM wrapping that chat model
  - get_ragas_models() -> (llm, embeddings) wrappers RAGAS can drive
"""

from __future__ import annotations

import sys
import types
from functools import lru_cache
from typing import Any

from evals.config import settings


@lru_cache(maxsize=1)
def get_chat_model() -> Any:
    """Return a LangChain chat model for the configured judge provider."""
    provider = settings.judge_provider
    model = settings.judge_model_name

    if provider == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(model=model, temperature=0)
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model, temperature=0)
    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(model=model, temperature=0)
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=model, temperature=0)

    raise ValueError(f"Unknown judge provider: {provider!r}")


def _install_ragas_compat() -> None:
    """Shim Vertex AI modules ragas imports but langchain v1 removed.

    ragas 0.x unconditionally imports langchain_community's Vertex AI classes,
    which the langchain v1 community package no longer ships, so a plain
    ``import ragas`` raises ModuleNotFoundError. We never use Vertex AI, so inject
    lightweight placeholders only when the real modules are genuinely absent.
    """
    for modname, attrs in (
        ("langchain_community.chat_models.vertexai", ["ChatVertexAI"]),
        ("langchain_community.llms.vertexai", ["VertexAI"]),
    ):
        if modname in sys.modules:
            continue
        try:
            __import__(modname)
        except Exception:
            module = types.ModuleType(modname)
            for attr in attrs:
                setattr(module, attr, type(attr, (), {}))
            sys.modules[modname] = module


@lru_cache(maxsize=1)
def get_local_embeddings() -> Any:
    """A local sentence-transformers embedding model (no API cost)."""
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:  # older stacks expose it via langchain_community
        from langchain_community.embeddings import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name=settings.embedding_model)


def get_ragas_models() -> tuple[Any, Any]:
    """Return (llm, embeddings) wrapped for RAGAS, driven by the judge provider."""
    _install_ragas_compat()
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper

    llm = LangchainLLMWrapper(get_chat_model())
    embeddings = LangchainEmbeddingsWrapper(get_local_embeddings())
    return llm, embeddings


def get_deepeval_judge() -> Any:
    """Return a DeepEvalBaseLLM that scores with the configured chat model.

    Built lazily inside the function so importing this module never requires
    DeepEval (keeps the offline test suite light).
    """
    from deepeval.models.base_model import DeepEvalBaseLLM

    chat_model = get_chat_model()
    model_label = settings.judge_model_name

    class _LangChainJudge(DeepEvalBaseLLM):
        """Adapts a LangChain chat model to DeepEval's judge interface."""

        def __init__(self, model: Any, name: str) -> None:
            self._model = model
            self._name = name

        def load_model(self) -> Any:
            return self._model

        def generate(self, prompt: str, schema: Any = None) -> Any:
            model = self.load_model()
            if schema is not None:
                # DeepEval v3 passes a pydantic schema for structured metrics;
                # Groq/OpenAI chat models honour it via structured output.
                structured = model.with_structured_output(schema)
                return structured.invoke(prompt)
            return model.invoke(prompt).content

        async def a_generate(self, prompt: str, schema: Any = None) -> Any:
            model = self.load_model()
            if schema is not None:
                structured = model.with_structured_output(schema)
                return await structured.ainvoke(prompt)
            response = await model.ainvoke(prompt)
            return response.content

        def get_model_name(self) -> str:
            return f"{self._name} (via LangChain)"

    return _LangChainJudge(chat_model, model_label)
