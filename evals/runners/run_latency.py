"""
run_latency.py — Measure the RAG pipeline's per-stage and end-to-end latency.

The gate has a ``latency_p95_ms`` threshold (< 3000ms) that nothing measured.
This fills it: it times the self-healing RAG's four stages —

    retrieve (Qdrant)  →  rerank (Cohere)  →  generate (LLM)  →  critique (LLM)

— over a sample of golden questions and reports p50 / p95 / mean per stage plus
end to end. The two LLM calls dominate; retrieval is a rounding error, which is
exactly why the design retrieves cheaply and spends the budget on generation.

Writes ``latency_p95_ms`` (+ a breakdown) so the gate and dashboard can consume it.

Usage:
    python evals/runners/run_latency.py --limit 5 --output results/latency.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from system_under_test.rag_pipeline import _ensure_project_one_path  # noqa: E402

_STAGES = ("retrieve", "rerank", "generate", "critique", "total")


def _pct(xs: list[float], p: float) -> float:
    s = sorted(xs)
    return s[min(int(p * len(s)), len(s) - 1)] if s else 0.0


def _ms(dt: float) -> float:
    return dt * 1000.0


def run(golden_path: str, output_path: str, limit: int | None) -> dict:
    with open(golden_path, encoding="utf-8") as f:
        goldens = json.load(f)
    if limit:
        goldens = goldens[:limit]

    _ensure_project_one_path()
    # Drive Project 1's graph nodes directly so each stage is timed in isolation.
    from src.graph.nodes import critique, generate, rerank, retrieve  # type: ignore

    timings: dict[str, list[float]] = {k: [] for k in _STAGES}

    # Warm up (model/connection load) so cold-start isn't charged to stage 1.
    try:
        retrieve({"query": "warm up"})
    except Exception:
        pass

    print(f"Timing the self-healing RAG over {len(goldens)} questions...")
    for i, item in enumerate(goldens, start=1):
        q = item["question"]
        print(f"  [{i}/{len(goldens)}] {q[:56]}")
        state: dict = {"query": q}
        t_all = time.perf_counter()
        for stage, fn in (("retrieve", retrieve), ("rerank", rerank),
                          ("generate", generate), ("critique", critique)):
            t = time.perf_counter()
            state.update(fn(state))
            timings[stage].append(_ms(time.perf_counter() - t))
        timings["total"].append(_ms(time.perf_counter() - t_all))

    breakdown = {
        s: {"mean": round(sum(v) / len(v), 1), "p50": round(_pct(v, 0.50), 1),
            "p95": round(_pct(v, 0.95), 1)}
        for s, v in timings.items() if v
    }
    results = {
        "n_queries": float(len(goldens)),
        "latency_p95_ms": breakdown["total"]["p95"],
        "latency_p50_ms": breakdown["total"]["p50"],
        "latency_mean_ms": breakdown["total"]["mean"],
        "stage_breakdown_ms": breakdown,
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\n── Self-Healing RAG latency (per stage, ms) ─────────────────")
    print(f"  {'stage':<10}{'mean':>10}{'p50':>10}{'p95':>10}")
    for s in _STAGES:
        if s in breakdown:
            b = breakdown[s]
            print(f"  {s:<10}{b['mean']:>10.1f}{b['p50']:>10.1f}{b['p95']:>10.1f}")
    print("─────────────────────────────────────────────────────────────")
    budget = "PASS" if results["latency_p95_ms"] < 3000 else "OVER"
    print(f"  p95 end-to-end: {results['latency_p95_ms']:.0f} ms  (gate < 3000 ms → {budget})")
    print(f"Written to {output_path}\n")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Benchmark self-healing RAG latency.")
    ap.add_argument("--golden", default="evals/datasets/golden_rag.json")
    ap.add_argument("--output", default="results/latency.json")
    ap.add_argument("--limit", type=int, default=5)
    args = ap.parse_args()
    run(args.golden, args.output, args.limit)
