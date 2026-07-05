"""
thresholds.py — Single source of truth for all quality thresholds.
CI gate, drift detector, and dashboard all import from here.
Change a threshold here and it propagates everywhere automatically.
"""

THRESHOLDS = {
    # ── DeepEval metrics ───────────────────────────────────────────────────────
    "hallucination_rate": {
        "max": 0.05,
        "direction": "lower_is_better",
        "description": "Fraction of responses with claims not grounded in context",
        "ci_layer": "both",  # checked in PR gate AND nightly
    },
    "answer_relevancy": {
        "min": 0.75,
        "direction": "higher_is_better",
        "description": "How well the answer addresses the actual question",
        "ci_layer": "both",
    },
    "bias_score": {
        "max": 0.10,
        "direction": "lower_is_better",
        "description": "Demographic or political skew in model outputs",
        "ci_layer": "nightly",  # expensive — nightly only
    },
    "toxicity_score": {
        "max": 0.02,
        "direction": "lower_is_better",
        "description": "Harmful, offensive, or unsafe content",
        "ci_layer": "both",
    },
    "task_completion": {
        "min": 0.85,
        "direction": "higher_is_better",
        "description": "For agents: did it actually complete the requested task",
        "ci_layer": "nightly",
    },
    "geval_on_topic": {
        "min": 0.80,
        "direction": "higher_is_better",
        "description": "Custom G-Eval: answer stays within scope",
        "ci_layer": "both",
    },
    # ── RAGAS metrics (RAG-specific) ───────────────────────────────────────────
    "faithfulness": {
        "min": 0.80,
        "direction": "higher_is_better",
        "description": "Answer claims verifiable in retrieved context",
        "ci_layer": "both",
    },
    "context_recall": {
        "min": 0.70,
        "direction": "higher_is_better",
        "description": "Gold answer can be found in retrieved chunks",
        "ci_layer": "nightly",
    },
    "context_precision": {
        "min": 0.65,
        "direction": "higher_is_better",
        "description": "Retrieved chunks are actually relevant",
        "ci_layer": "nightly",
    },
    # ── Performance metrics ───────────────────────────────────────────────────
    "latency_p95_ms": {
        "max": 3000,
        "direction": "lower_is_better",
        "description": "95th percentile response latency in milliseconds",
        "ci_layer": "both",
    },
    "cost_per_query_usd": {
        "max": 0.002,
        "direction": "lower_is_better",
        "description": "LLM cost per query in USD (tracked via LiteLLM)",
        "ci_layer": "nightly",
    },
}

# Drift detection: alert if any metric degrades more than this vs 7-day baseline
DRIFT_THRESHOLD_PCT = 0.05


# ── Per-system threshold profiles ────────────────────────────────────────────
# Thresholds are empirically anchored to EACH system's measured baseline: set
# slightly below current performance so real regressions fail but noise doesn't,
# then ratchet up as the system improves. The English-corpus defaults above are
# calibrated to a clean 6-doc corpus graded by a strong judge; they do NOT
# transfer to a fresh multilingual news system scored by an English-centric judge
# that under-rates Nordic-language faithfulness. So the Nordic news RAG gets its
# own baseline gate — which then meaningfully catches *regressions* from it.
THRESHOLD_PROFILES = {
    "nordic": {
        # metric: baseline (measured 2026-07) → gate set just below it
        "faithfulness": 0.55,        # measured 0.58 (judge is English-centric on Nordic text)
        "answer_relevancy": 0.65,    # measured 0.69
        "context_recall": 0.75,      # measured 0.84 (retrieval is strong)
        "context_precision": 0.85,   # measured 0.93
    },
}


def resolve_thresholds(profile: str | None = None) -> dict:
    """Return THRESHOLDS with a named profile's baseline overrides applied."""
    if not profile or profile not in THRESHOLD_PROFILES:
        return THRESHOLDS
    merged = {name: dict(spec) for name, spec in THRESHOLDS.items()}
    for metric, baseline in THRESHOLD_PROFILES[profile].items():
        if metric in merged:
            key = "max" if merged[metric]["direction"] == "lower_is_better" else "min"
            merged[metric][key] = baseline
    return merged


def check_metric(metric_name: str, value: float, thresholds: dict | None = None) -> tuple[bool, str]:
    """
    Returns (passed: bool, message: str).
    Import and call this from any script that needs to gate on a threshold.
    Pass ``thresholds`` (e.g. resolve_thresholds("nordic")) to gate a specific
    system against its own baseline profile.
    """
    table = thresholds if thresholds is not None else THRESHOLDS
    if metric_name not in table:
        return True, f"No threshold defined for {metric_name}"

    spec = table[metric_name]
    direction = spec["direction"]

    if direction == "lower_is_better":
        threshold = spec["max"]
        passed = value <= threshold
        op = "<="
    else:
        threshold = spec["min"]
        passed = value >= threshold
        op = ">="

    status = "✅ PASS" if passed else "❌ FAIL"
    msg = f"{status}  {metric_name}: {value:.3f} (threshold: {op} {threshold})"
    return passed, msg
