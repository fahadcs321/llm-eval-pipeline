"""Tests for the configuration layer (provider -> key mapping, judge model)."""

from __future__ import annotations

from evals.config import DEFAULT_JUDGE_MODELS, Settings


def test_defaults_to_groq():
    s = Settings(judge_provider="groq", judge_model=None)
    assert s.judge_provider == "groq"
    assert s.judge_model_name == DEFAULT_JUDGE_MODELS["groq"]
    assert s.judge_key_attr == "groq_api_key"


def test_explicit_model_overrides_default():
    s = Settings(judge_provider="groq", judge_model="llama-3.1-8b-instant")
    assert s.judge_model_name == "llama-3.1-8b-instant"


def test_provider_key_mapping():
    assert Settings(judge_provider="openai").judge_key_attr == "openai_api_key"
    assert Settings(judge_provider="anthropic").judge_key_attr == "anthropic_api_key"
    assert Settings(judge_provider="google").judge_key_attr == "google_api_key"


def test_require_raises_on_missing_key():
    s = Settings(judge_provider="groq", groq_api_key=None)
    try:
        s.require("groq_api_key")
    except RuntimeError as exc:
        assert "GROQ_API_KEY" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("require() should have raised")


def test_require_passes_when_present():
    s = Settings(judge_provider="groq", groq_api_key="gsk_test")
    s.require("groq_api_key")  # must not raise
