"""
run_cost.py — Track LLM cost per query across every eval run.
Uses LiteLLM's built-in cost tracking.

Why this matters: eval runs that cost $9 per PR get disabled by engineers
within a month. This tracker keeps the eval suite cost-aware.

Usage:
  python evals/runners/run_cost.py --output results/cost_nightly.json
"""

import argparse
import json
import os
import time
from pathlib import Path

# Default to the Groq model the system under test runs on (no OpenAI key needed).
DEFAULT_COST_MODEL = os.getenv("COST_MODEL", "groq/llama-3.3-70b-versatile")


class CostTracker:
    def __init__(self):
        self.total_cost = 0.0
        self.total_tokens = 0
        self.n_queries = 0
        self.latencies = []

    def track(self, query: str, model: str = DEFAULT_COST_MODEL) -> dict:
        # Imported lazily so the module (and its pure summary math) is testable
        # offline without litellm installed.
        import litellm
        from litellm import completion

        t0 = time.time()
        response = completion(
            model=model,
            messages=[{"role": "user", "content": query}],
        )

        latency_ms = (time.time() - t0) * 1000
        try:
            cost = litellm.completion_cost(completion_response=response)
        except Exception:
            cost = 0.0
        tokens = response.usage.total_tokens

        self.total_cost += cost
        self.total_tokens += tokens
        self.n_queries += 1
        self.latencies.append(latency_ms)

        return {
            "answer": response.choices[0].message.content,
            "cost_usd": cost,
            "tokens": tokens,
            "latency_ms": latency_ms,
        }

    def summary(self) -> dict:
        latencies_sorted = sorted(self.latencies)
        n = len(latencies_sorted)
        return {
            "n_queries": self.n_queries,
            "total_cost_usd": round(self.total_cost, 6),
            "avg_cost_per_query_usd": round(self.total_cost / max(self.n_queries, 1), 6),
            "total_tokens": self.total_tokens,
            "avg_tokens_per_query": round(self.total_tokens / max(self.n_queries, 1)),
            "latency_p50_ms": round(latencies_sorted[int(0.50 * n)], 1) if n else 0,
            "latency_p95_ms": round(latencies_sorted[int(0.95 * n)], 1) if n else 0,
        }


def run_cost_eval(golden_path: str, output_path: str):
    with open(golden_path) as f:
        goldens = json.load(f)

    tracker = CostTracker()

    print(f"Cost tracking across {len(goldens)} queries...")
    for item in goldens:
        tracker.track(item["question"])

    summary = tracker.summary()

    Path(output_path).parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n── Cost Report ─────────────────────────────────────────────")
    print(f"  Total cost:             ${summary['total_cost_usd']:.4f}")
    print(f"  Avg cost per query:     ${summary['avg_cost_per_query_usd']:.6f}")
    print(f"  Avg tokens per query:   {summary['avg_tokens_per_query']}")
    print(f"  Latency p50:            {summary['latency_p50_ms']}ms")
    print(f"  Latency p95:            {summary['latency_p95_ms']}ms")
    print("────────────────────────────────────────────────────────────\n")

    # Gate: alert if avg cost exceeds $0.002 per query
    if summary["avg_cost_per_query_usd"] > 0.002:
        print(
            f"⚠️  COST WARNING: ${summary['avg_cost_per_query_usd']:.6f}/query"
            f" exceeds $0.002 threshold"
        )

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden", default="evals/datasets/golden_rag.json")
    parser.add_argument("--output", default="results/cost_nightly.json")
    args = parser.parse_args()
    run_cost_eval(args.golden, args.output)
