# LLM Eval CI/CD Pipeline

> Automated quality gates for LLM systems — runs on every push, blocks merges on
> regression, tracks drift over time. Unit tests, but for LLM quality.

[![CI](https://github.com/fahadcs321/llm-eval-pipeline/actions/workflows/pr_eval.yml/badge.svg)](https://github.com/fahadcs321/llm-eval-pipeline/actions/workflows/pr_eval.yml)
[![Nightly Eval](https://github.com/fahadcs321/llm-eval-pipeline/actions/workflows/nightly_eval.yml/badge.svg)](https://github.com/fahadcs321/llm-eval-pipeline/actions/workflows/nightly_eval.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## The problem

Most AI teams still rely on vibes and manual spot-checks to know whether a change
made things better or worse. Swap a model, tweak a prompt, update the knowledge
base — and nobody knows for a week whether it helped or quietly broke something.

## The solution

An automated evaluation pipeline that runs like a test suite:

```
git push → GitHub Actions
         → run the system under test against the golden Q&A set
         → DeepEval grades relevancy, faithfulness, bias, toxicity, on-topic
         → RAGAS grades faithfulness, recall, precision
         → cost + latency tracked via LiteLLM
         → quality gate: PASS (merge allowed) or FAIL (merge blocked)
         → scores logged to Langfuse → Grafana shows the rolling trend
```

This repo evaluates [**Project 1 — Self-Healing RAG**](https://github.com/fahadcs321/self-healing-rag)
as its system under test:

- Project 1 = *"I built a production AI system."*
- Project 2 (this repo) = *"I know how to measure and maintain it."*

Point `system_under_test/rag_pipeline.py` at any other system to evaluate it
instead — the whole pipeline runs against the same `{answer, sources, contexts}`
contract.

---

## What runs, and when

| Layer | Trigger | Cost | What it does |
|-------|---------|------|--------------|
| **Unit** | every push + PR | free | Ruff lint + offline test suite. No API keys. Always green. |
| **Eval gate** | PR, when `RUN_EVAL=true` | cheap | DeepEval on a sampled set, gated on thresholds — blocks the merge. |
| **Nightly** | 02:00 cron | full | Full RAGAS + DeepEval sweep, cost, drift detection, dashboards. |

The offline unit job is what keeps the badge green on every commit without
spending a cent. The LLM-judge layers are gated behind a repo variable, so they
run only when you opt in with a key — the 2026 standard of *fast cheap checks on
every PR, full judge sweep nightly*.

---

## Stack

| Layer | Tool | Why |
|-------|------|-----|
| Eval framework | DeepEval v3 | pytest-native; `assert_test` blocks CI on failure |
| RAG eval | RAGAS | faithfulness / recall / precision — RAG-specific |
| Judge LLM | **Groq · Llama 3.3 70B** | free, no card; swappable via `JUDGE_PROVIDER` |
| Embeddings | sentence-transformers (local) | RAGAS scoring with no OpenAI key |
| Cost tracking | LiteLLM | cost-per-query on every run |
| Observability | Langfuse | full traces, rolling baseline |
| Dashboard | Grafana | 30-day quality trend, drift alerts |
| CI/CD | GitHub Actions | offline gate + gated judge + nightly cron |

No OpenAI key required anywhere — the judge defaults to Groq, embeddings are
local. Set `JUDGE_PROVIDER=openai|google|anthropic` to switch.

---

## Metrics & thresholds

All thresholds live in one place — [`evals/metrics/thresholds.py`](evals/metrics/thresholds.py).
The CI gate, drift detector, and dashboard all import from it.

| Metric | Threshold | Layer |
|--------|-----------|-------|
| Hallucination rate | < 5% | both |
| Answer relevancy | ≥ 0.75 | both |
| Faithfulness | ≥ 0.80 | both |
| Toxicity | < 0.02 | both |
| Bias | < 0.10 | nightly |
| Context recall | ≥ 0.70 | nightly |
| Context precision | ≥ 0.65 | nightly |
| Latency p95 | < 3000 ms | both |
| Cost per query | < $0.002 | nightly |

---

## Quick start

```bash
git clone https://github.com/fahadcs321/llm-eval-pipeline
cd llm-eval-pipeline

# Offline checks — no keys needed
make install-dev
make lint
make test

# Live eval — needs a Groq key (free at console.groq.com)
make install
cp .env.example .env          # set GROQ_API_KEY
make eval-rag                 # RAGAS sweep over the golden set
make gate                     # apply thresholds → exit non-zero on regression
```

The system under test is Project 1, resolved automatically from `../self_healing_rag`.
Override the location with `SELF_HEALING_RAG_PATH`. If Project 1 isn't present,
the SUT degrades to a bare Groq baseline so the pipeline still runs.

---

## How the gate blocks a merge

`evals/gate.py` reads a results JSON, checks every metric against `thresholds.py`,
and exits non-zero on any regression. GitHub Actions reads that exit code:

```bash
$ python -m evals.gate --results results/ragas_nightly.json --layer pr
── CI Quality Gate ──────────────────────────────────────────
  ✅ PASS  faithfulness: 0.870 (threshold: >= 0.8)
  ❌ FAIL  answer_relevancy: 0.610 (threshold: >= 0.75)
─────────────────────────────────────────────────────────────
  Result: FAIL — merge blocked
$ echo $?
1
```

---

## Project structure

```
llm-eval-pipeline/
├── evals/
│   ├── config.py                # provider/key/model config (Groq by default)
│   ├── judge.py                 # the LLM-as-judge: Groq for DeepEval + RAGAS
│   ├── gate.py                  # CI quality gate — exits non-zero on regression
│   ├── datasets/                # golden_rag / golden_edge / golden_general
│   ├── metrics/                 # thresholds (single source of truth) + G-Eval
│   ├── tests/                   # DeepEval pytest suites (the real gate)
│   ├── runners/                 # run_ragas · run_deepeval · run_cost
│   └── reporting/               # langfuse · grafana · slack
├── tests/                       # offline unit suite — no keys, keeps CI green
├── system_under_test/           # rag_pipeline (Project 1) + llm_app baseline
├── scripts/                     # drift_detector · synthesize_goldens · promote_failures
├── dashboard/grafana_dashboard.json
└── .github/workflows/           # pr_eval (unit + gated eval) · nightly_eval
```

---

## The living golden dataset

The golden set is not frozen. Three layers keep it calibrated to reality:

1. **Hand-crafted** — corpus-aligned Q&A pairs in `golden_rag.json`.
2. **Synthesized** — `scripts/synthesize_goldens.py` expands coverage from docs.
3. **Promoted failures** — `scripts/promote_failures.py` pulls failing production
   traces from Langfuse into `golden_edge.json` weekly.

---

## Notes from a real run

Running the full RAGAS sweep on the **Groq free tier** will hit its 100k
tokens/day limit — RAGAS makes several judge calls per sample. That throttling
is exactly what the cost tracker is built to surface. For an unthrottled sweep,
use a Groq Dev-tier key or run nightly when the daily quota resets.

---

## Built by

**Muhammad Fahad** · BSc Computer Science
[GitHub](https://github.com/fahadcs321) · [LinkedIn](https://www.linkedin.com/in/muhammad-fahad-89a1b0358/)
