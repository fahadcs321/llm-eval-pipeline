"""
config.py — Central configuration for the eval pipeline.

One place to decide which LLM judges the outputs, which model the system under
test runs on, and where observability data is shipped. Everything reads from the
environment so the same code runs locally, in CI, and in the nightly cron.

The judge defaults to Groq (free, no card required) rather than OpenAI, so the
whole pipeline runs without an OpenAI key — matching the system under test
(Project 1, Self-Healing RAG), which also runs on Groq.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

# Sensible default judge model per provider. The judge does LLM-as-judge scoring
# for DeepEval (G-Eval, faithfulness, relevancy) and RAGAS.
DEFAULT_JUDGE_MODELS = {
    "openai": "gpt-4o-mini",
    "groq": "llama-3.3-70b-versatile",
    "google": "gemini-1.5-flash",
    "anthropic": "claude-3-5-haiku-latest",
}


@dataclass(frozen=True)
class Settings:
    # ── Judge LLM (scores the outputs) ─────────────────────────────────────────
    judge_provider: str = field(
        default_factory=lambda: os.getenv(
            "JUDGE_PROVIDER", os.getenv("LLM_PROVIDER", "groq")
        ).lower()
    )
    judge_model: str | None = field(default_factory=lambda: os.getenv("JUDGE_MODEL"))

    # ── Provider API keys ──────────────────────────────────────────────────────
    openai_api_key: str | None = field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    groq_api_key: str | None = field(default_factory=lambda: os.getenv("GROQ_API_KEY"))
    google_api_key: str | None = field(default_factory=lambda: os.getenv("GOOGLE_API_KEY"))
    anthropic_api_key: str | None = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY"))

    # ── Embeddings (RAGAS uses these; local model = no API cost) ────────────────
    embedding_model: str = field(
        default_factory=lambda: os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
    )

    # ── Cost tracking ──────────────────────────────────────────────────────────
    cost_model: str = field(
        default_factory=lambda: os.getenv("COST_MODEL", "groq/llama-3.3-70b-versatile")
    )

    # ── Observability (all optional — skipped cleanly if unset) ────────────────
    langfuse_host: str = field(
        default_factory=lambda: os.getenv("LANGFUSE_HOST", "http://localhost:3000")
    )
    langfuse_public_key: str | None = field(
        default_factory=lambda: os.getenv("LANGFUSE_PUBLIC_KEY")
    )
    langfuse_secret_key: str | None = field(
        default_factory=lambda: os.getenv("LANGFUSE_SECRET_KEY")
    )
    grafana_url: str = field(
        default_factory=lambda: os.getenv("GRAFANA_URL", "http://localhost:3001")
    )
    grafana_api_key: str | None = field(default_factory=lambda: os.getenv("GRAFANA_API_KEY"))
    slack_webhook_url: str | None = field(default_factory=lambda: os.getenv("SLACK_WEBHOOK_URL"))

    @property
    def judge_model_name(self) -> str:
        return self.judge_model or DEFAULT_JUDGE_MODELS.get(
            self.judge_provider, "llama-3.3-70b-versatile"
        )

    @property
    def judge_key_attr(self) -> str:
        return {
            "openai": "openai_api_key",
            "groq": "groq_api_key",
            "google": "google_api_key",
            "anthropic": "anthropic_api_key",
        }.get(self.judge_provider, "groq_api_key")

    @property
    def judge_api_key(self) -> str | None:
        return getattr(self, self.judge_key_attr)

    def require(self, *attrs: str) -> None:
        """Raise a clear error if any required setting is missing."""
        missing = [a for a in attrs if not getattr(self, a, None)]
        if missing:
            raise RuntimeError(
                "Missing required configuration: "
                + ", ".join(a.upper() for a in missing)
                + ". Set them in your environment or .env file."
            )


settings = Settings()
