"""
streamlit_app.py — "Build ↔ Measure" demo for the LLM Eval CI/CD pipeline.

Design: "Mission Control" — a CI-status view of LLM quality. The signature
element is the threshold bar on every metric card: the fill is the score, the
tick is the gate threshold, and the color is the verdict. The gate banner reads
like a build status, because that is exactly what it is.

  1. Quality-gate dashboard — the latest cached eval run (results/*.json) as
     metric cards with threshold bars, per system-under-test.
  2. Live "grade a question" — left: Project 1 answers with its self-heal trace;
     right: this pipeline judges the answer independently and gates it.

Shared "Reasoning Instrument" system: violet-ink night, porcelain text,
Bricolage Grotesque / Albert Sans / Spline Sans Mono.

Run with:
    streamlit run app/streamlit_app.py --server.port 8502     # beside P1 on :8501
"""

import json
import os
import re
import sys
from pathlib import Path

# Make evals.* / system_under_test.* importable when Streamlit runs this file directly.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

# Bridge Streamlit Cloud secrets into the environment BEFORE importing anything that
# reads os.getenv. .strip() guards pasted newlines.
try:
    for _key, _val in st.secrets.items():
        if isinstance(_val, str):
            os.environ[_key] = _val.strip()
except Exception:
    pass

# Put Project 1 on the path so we can import its full entry point (answer_query returns
# the critique + reason the bridge's SUT contract omits — we want the full trace here).
_P1_PATH = os.getenv("SELF_HEALING_RAG_PATH", str(ROOT.parent / "self_healing_rag"))
if os.path.isdir(_P1_PATH) and _P1_PATH not in sys.path:
    sys.path.insert(0, _P1_PATH)

from evals.gate import evaluate_gates  # noqa: E402
from evals.judge import get_chat_model  # noqa: E402
from evals.metrics.thresholds import THRESHOLDS, check_metric, resolve_thresholds  # noqa: E402

# ── Page setup ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LLM Eval · measures Project 1",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Design tokens ─────────────────────────────────────────────────────────────
BG = "#0D0A1C"
SURFACE = "#171232"
SURFACE_2 = "#131028"
BORDER = "#2A2450"
OK = "#3EE08F"
BAD = "#FF7A7A"
WARN = "#FFD166"
ICE = "#7DE1FF"
VIOLET = "#9F8BFF"
TEXT = "#F4F1E8"
MUTED = "#9A94B8"

EXAMPLES = [
    "What does the RAGAS faithfulness metric measure?",
    "What does LangGraph enable that a LangChain chain cannot?",
    "What is the difference between dense and sparse retrieval?",
    "Who won the 2022 FIFA World Cup?",  # off-corpus → honest refusal + a low judge score
]

# Metrics we surface in the dashboard, grouped by source (only shown if present).
METRIC_GROUPS = [
    ("RAGAS · retrieval quality",
     ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]),
    ("DeepEval · safety & scope",
     ["hallucination_rate", "toxicity_score", "bias_score", "geval_on_topic"]),
    ("Cost & latency", ["cost_per_query_usd", "latency_p95_ms"]),
]


def icon(name: str, size: int = 15) -> str:
    paths = {
        "gauge": '<path d="m12 14 4-4"/><path d="M3.34 19a10 10 0 1 1 17.32 0"/>',
        "scale": '<path d="m16 16 3-8 3 8c-.87.65-1.92 1-3 1s-2.13-.35-3-1Z"/>'
        '<path d="m2 16 3-8 3 8c-.87.65-1.92 1-3 1s-2.13-.35-3-1Z"/>'
        '<path d="M7 21h10"/><path d="M12 3v18"/><path d="M3 7h2c2 0 5-1 7-2 2 1 5 2 7 2h2"/>',
        "check": '<path d="M20 6 9 17l-5-5"/>',
        "x": '<path d="M18 6 6 18"/><path d="m6 6 12 12"/>',
        "alert": '<path d="M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 '
        '2 0 0 0-3.4 0Z"/><path d="M12 9v4"/><path d="M12 17h.01"/>',
        "search": '<circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/>',
        "doc": '<path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z"/>'
        '<path d="M14 2v5h5"/>',
        "loop": '<path d="M17 2.1 21 6l-3.9 3.9"/><path d="M3 12V9a3 3 0 0 1 3-3h15"/>'
        '<path d="M7 21.9 3 18l3.9-3.9"/><path d="M21 12v3a3 3 0 0 1-3 3H3"/>',
        "spark": '<path d="M12 3v4M12 17v4M3 12h4M17 12h4M5.6 5.6l2.8 2.8M15.6 15.6l2.8 '
        '2.8M5.6 18.4l2.8-2.8M15.6 8.4l2.8-2.8"/>',
        "shield": '<path d="M20 13c0 5-3.5 7.5-7.7 8.9a1 1 0 0 1-.6 0C7.5 20.5 4 18 4 13V6a1 1 '
        '0 0 1 1-1c2 0 4.5-1.2 6.2-2.7a1 1 0 0 1 1.5 0C14.5 3.8 17 5 19 5a1 1 0 0 1 1 1Z"/>',
    }
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
        f'stroke-linecap="round" stroke-linejoin="round">{paths[name]}</svg>'
    )


# ── Global styles ─────────────────────────────────────────────────────────────
st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,700;12..96,800&family=Albert+Sans:wght@400;500;600;700&family=Spline+Sans+Mono:wght@400;500;600&display=swap');
.stApp {{
  background:
    radial-gradient(1000px 520px at 12% -12%, rgba(159,139,255,0.13), transparent 60%),
    radial-gradient(800px 460px at 95% -8%, rgba(125,225,255,0.08), transparent 55%),
    {BG};
  font-family: 'Albert Sans', system-ui, sans-serif; color: {TEXT};
}}
#MainMenu, header[data-testid="stHeader"], footer {{ display: none; }}
.block-container {{ padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1180px; }}
code, .mono {{ font-family: 'Spline Sans Mono', monospace; }}

/* ── mission-control header (left-aligned) ───────────── */
.eyebrow {{ display:flex; align-items:center; justify-content:space-between; margin-bottom:.9rem; }}
.eyebrow .tag {{
  display:inline-flex; align-items:center; gap:.5rem; font-family:'Spline Sans Mono',monospace;
  font-size:.68rem; letter-spacing:.16em; text-transform:uppercase; color:{ICE};
}}
.eyebrow .sut {{
  font-family:'Spline Sans Mono',monospace; font-size:.66rem; letter-spacing:.06em;
  color:{MUTED}; border:1px solid {BORDER}; border-radius:999px; padding:.25rem .7rem;
}}
.eyebrow .sut b {{ color:{OK}; font-weight:600; }}
.masthead {{ display:flex; align-items:flex-end; justify-content:space-between; gap:2rem;
  border-bottom:1px solid {BORDER}; padding-bottom:1.3rem; margin-bottom:1.3rem; flex-wrap:wrap; }}
.masthead h1 {{
  font-family:'Bricolage Grotesque',sans-serif; font-weight:800; font-size:3rem;
  line-height:1.02; letter-spacing:-.015em; margin:0; max-width:560px;
  background:linear-gradient(180deg,#FFFFFF,#CFC8E8);
  -webkit-background-clip:text; -webkit-text-fill-color:transparent;
}}
.masthead .side {{ max-width:400px; }}
.masthead .side p {{ color:{MUTED}; font-size:.98rem; line-height:1.65; margin:0 0 .8rem; }}
.flow {{ display:flex; flex-wrap:wrap; gap:.35rem; font-family:'Spline Sans Mono',monospace; font-size:.72rem; }}
.flow span {{ color:{TEXT}; background:{SURFACE}; border:1px solid {BORDER};
  padding:.24rem .55rem; border-radius:7px; }}
.flow .arrow {{ color:{MUTED}; border:none; background:none; padding:.24rem .05rem; }}

.label {{ display:flex; align-items:center; gap:.5rem; color:{MUTED}; font-family:'Spline Sans Mono',monospace;
  font-size:.72rem; letter-spacing:.1em; text-transform:uppercase; margin:.2rem 0 .6rem; }}
.label svg {{ color:{ICE}; }}
.card {{ background:{SURFACE}; border:1px solid {BORDER}; border-radius:16px;
  padding:1.25rem 1.35rem; box-shadow:0 12px 32px rgba(0,0,0,0.3); }}
.answer-card {{ font-size:1.02rem; line-height:1.7; color:{TEXT}; }}
.pill {{ display:inline-flex; align-items:center; gap:.45rem; font-weight:600; font-size:.8rem;
  font-family:'Spline Sans Mono',monospace; padding:.32rem .7rem; border-radius:999px; margin-bottom:.9rem; }}

/* ── gate status block ───────────────────────────────── */
.gate {{ display:flex; align-items:center; gap:1rem; border-radius:14px; padding:1.1rem 1.3rem;
  border:1px solid {BORDER}; margin-bottom:1.1rem; }}
.gate .g-dot {{ width:14px; height:14px; border-radius:50%; flex:none; }}
.gate .g-title {{ font-family:'Bricolage Grotesque',sans-serif; font-weight:800; font-size:1.35rem;
  letter-spacing:-.01em; line-height:1.1; }}
.gate .g-sub {{ color:{MUTED}; font-family:'Spline Sans Mono',monospace; font-size:.72rem; margin-top:.3rem; }}

/* ── metric cards with threshold bars (the signature) ── */
.mgrid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(215px,1fr)); gap:.7rem; margin:.2rem 0 1.1rem; }}
.metric {{ background:{SURFACE_2}; border:1px solid {BORDER}; border-radius:12px; padding:.85rem 1rem .95rem; }}
.metric .m-top {{ display:flex; justify-content:space-between; align-items:center; }}
.metric .m-name {{ font-family:'Spline Sans Mono',monospace; font-size:.66rem; letter-spacing:.05em;
  text-transform:uppercase; color:{MUTED}; }}
.metric .m-val {{ font-family:'Spline Sans Mono',monospace; font-size:1.5rem; font-weight:600;
  color:{TEXT}; margin:.3rem 0 .45rem; }}
.metric .bar {{ position:relative; height:6px; border-radius:99px; background:rgba(154,148,190,.14); overflow:visible; }}
.metric .bar .fill {{ position:absolute; inset:0 auto 0 0; border-radius:99px; }}
.metric .bar .tick {{ position:absolute; top:-3px; bottom:-3px; width:2px; background:{ICE};
  border-radius:2px; box-shadow:0 0 6px rgba(125,225,255,.7); }}
.metric .m-thr {{ font-family:'Spline Sans Mono',monospace; font-size:.68rem; color:{MUTED}; margin-top:.5rem; }}

/* segmented SUT selector */
div[role="radiogroup"] {{ gap:.4rem; }}
div[role="radiogroup"] label {{
  background:{SURFACE_2}; border:1px solid {BORDER}; border-radius:999px !important;
  padding:.3rem .9rem !important; font-family:'Spline Sans Mono',monospace !important;
}}

/* two-column live grade */
.colhead {{ display:flex; align-items:center; gap:.55rem; font-family:'Spline Sans Mono',monospace;
  font-size:.72rem; letter-spacing:.08em; text-transform:uppercase; padding:.4rem .2rem .7rem; }}
.colhead .p1 {{ color:{OK}; }}
.colhead .p2 {{ color:{ICE}; }}
.stats {{ display:grid; grid-template-columns:repeat(3,1fr); gap:.7rem; margin:.9rem 0 1rem; }}
.stat {{ background:{SURFACE_2}; border:1px solid {BORDER}; border-radius:12px; padding:.8rem 1rem; text-align:center; }}
.stat .k {{ font-family:'Spline Sans Mono',monospace; font-size:.64rem; letter-spacing:.06em;
  text-transform:uppercase; color:{MUTED}; }}
.stat .v {{ font-family:'Spline Sans Mono',monospace; font-size:1.15rem; font-weight:600; color:{TEXT}; margin-top:.25rem; }}
.reason {{ color:{MUTED}; font-size:.9rem; line-height:1.6; border-left:2px solid {BORDER};
  padding-left:.85rem; margin:.2rem 0 1rem; }}
.reason b {{ color:{TEXT}; }}
.src {{ display:inline-flex; align-items:center; gap:.4rem; font-family:'Spline Sans Mono',monospace;
  font-size:.74rem; color:{TEXT}; background:rgba(139,125,255,0.06); border:1px solid {BORDER};
  padding:.3rem .6rem; border-radius:8px; margin:0 .4rem .4rem 0; }}
.src svg {{ color:{ICE}; }}
.ctx {{ background:rgba(13,10,28,0.5); border:1px solid {BORDER}; border-left:3px solid {VIOLET};
  border-radius:10px; padding:.7rem .9rem; margin-bottom:.55rem; color:#C9C4DE; font-size:.86rem; line-height:1.6; }}
.ctx .n {{ font-family:'Spline Sans Mono',monospace; color:{VIOLET}; font-weight:600; margin-right:.4rem; }}
.jrow {{ display:flex; align-items:center; justify-content:space-between; gap:.6rem;
  background:{SURFACE_2}; border:1px solid {BORDER}; border-radius:10px; padding:.6rem .85rem; margin-bottom:.5rem; }}
.jrow .j-left {{ display:flex; align-items:center; gap:.55rem; }}
.jrow .j-name {{ font-family:'Spline Sans Mono',monospace; font-size:.82rem; color:{TEXT}; }}
.jrow .j-thr {{ font-family:'Spline Sans Mono',monospace; font-size:.68rem; color:{MUTED}; }}
.jrow .j-val {{ font-family:'Spline Sans Mono',monospace; font-size:1rem; font-weight:600; }}

.stTextInput input {{ background:{SURFACE}!important; border:1px solid {BORDER}!important;
  border-radius:12px!important; color:{TEXT}!important; font-size:1rem!important; }}
.stTextInput input:focus {{ border-color:{ICE}!important; box-shadow:0 0 0 3px rgba(125,225,255,0.15)!important; }}
div.stButton > button {{ border-radius:10px; border:1px solid {BORDER}; background:{SURFACE}; color:{TEXT};
  font-family:'Albert Sans',sans-serif; font-size:.86rem; font-weight:500;
  transition:border-color .2s,color .2s,background .2s; }}
div.stButton > button:hover {{ border-color:{ICE}; color:{ICE}; background:rgba(125,225,255,0.05); }}
div.stButton > button[kind="primary"] {{ background:{OK}; border:1px solid {OK}; color:#052015; font-weight:700; }}
div.stButton > button[kind="primary"]:hover {{ background:#2BC77B; border-color:#2BC77B; color:#052015; }}
/* ── performance panel (measured latency) ────────────── */
.perf-head {{ display:flex; align-items:baseline; gap:.7rem; flex-wrap:wrap; margin-bottom:.9rem; }}
.perf-big {{ font-family:'Bricolage Grotesque',sans-serif; font-weight:800; font-size:1.7rem; color:{TEXT}; }}
.perf-sub {{ font-family:'Spline Sans Mono',monospace; font-size:.72rem; color:{MUTED}; }}
.perf-budget {{ margin-left:auto; font-family:'Spline Sans Mono',monospace; font-size:.72rem;
  border:1px solid {BORDER}; border-radius:999px; padding:.25rem .7rem; }}
.latrow {{ display:grid; grid-template-columns:96px 1fr 150px; align-items:center; gap:.7rem; margin-bottom:.5rem; }}
.latrow .ln {{ font-family:'Spline Sans Mono',monospace; font-size:.76rem; color:{TEXT}; }}
.latrow .track {{ height:11px; border-radius:99px; background:rgba(154,148,190,.12); overflow:hidden; }}
.latrow .track .fill {{ height:100%; border-radius:99px; }}
.latrow .lv {{ font-family:'Spline Sans Mono',monospace; font-size:.72rem; color:{MUTED}; text-align:right; }}
.latrow .lv b {{ color:{TEXT}; font-weight:600; }}
@media (max-width: 760px) {{ .masthead h1 {{ font-size:2.1rem; }} }}
@media (prefers-reduced-motion: reduce) {{ * {{ transition:none!important; animation:none!important; }} }}
</style>
""",
    unsafe_allow_html=True,
)


# ── Helpers ───────────────────────────────────────────────────────────────────
def _escape(text) -> str:
    return (str(text) if text is not None else "").replace("<", "&lt;").replace(">", "&gt;")


def _fmt_value(metric: str, value: float) -> str:
    if metric == "cost_per_query_usd":
        return f"${value:.4f}"
    if metric == "latency_p95_ms":
        return f"{value:.0f} ms"
    return f"{value:.3f}"


def _fmt_threshold(metric: str, table: dict | None = None) -> str:
    spec = (table or THRESHOLDS).get(metric, {})
    if spec.get("direction") == "lower_is_better":
        return f"≤ {spec.get('max')}"
    return f"≥ {spec.get('min')}"


def _metric_card(metric: str, value: float, table: dict | None = None) -> str:
    """A metric card with the signature threshold bar: fill = score, tick = gate."""
    table = table or THRESHOLDS
    spec = table.get(metric, {})
    ok, _ = check_metric(metric, value, thresholds=table)
    color = OK if ok else BAD
    if spec.get("direction") == "lower_is_better":
        thr = float(spec.get("max", 1.0))
        scale = max(2 * thr, value * 1.15, 1e-9)
    else:
        thr = float(spec.get("min", 1.0))
        scale = max(1.0, value * 1.05)
    fill = min(value / scale * 100, 100)
    tick = min(thr / scale * 100, 100)
    mark = (
        f'<span style="color:{OK}">{icon("check", 14)}</span>' if ok
        else f'<span style="color:{BAD}">{icon("x", 14)}</span>'
    )
    return (
        f'<div class="metric"><div class="m-top"><span class="m-name">{metric}</span>{mark}</div>'
        f'<div class="m-val" style="color:{color}">{_fmt_value(metric, value)}</div>'
        f'<div class="bar"><div class="fill" style="width:{fill:.1f}%; background:{color}"></div>'
        f'<div class="tick" style="left:{tick:.1f}%"></div></div>'
        f'<div class="m-thr">gate {_fmt_threshold(metric, table)}</div></div>'
    )


# ── Dashboard data: per system-under-test result files (never blended) ─────────
SUT_RESULT_FILES = {
    "Self-Healing RAG (original)": {
        "ragas_sample.json", "ragas_nightly.json", "ragas_pr.json",
        "deepeval_nightly.json", "deepeval_pr.json", "cost_nightly.json",
    },
    "Multilingual News RAG": {"ragas_news.json"},
}


def load_results(allowed: set[str]) -> tuple[dict, list[str]]:
    """Merge the given results/*.json files (newest wins). Returns (metrics, filenames)."""
    results_dir = ROOT / "results"
    merged: dict[str, float] = {}
    files: list[str] = []
    if not results_dir.is_dir():
        return merged, files
    paths = [p for p in results_dir.glob("*.json") if p.name in allowed]
    for path in sorted(paths, key=lambda p: p.stat().st_mtime):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        files.append(path.name)
        for key, val in data.items():
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                merged[key] = float(val)
    return merged, files


def render_dashboard(sut_label: str) -> None:
    results, files = load_results(SUT_RESULT_FILES.get(sut_label, set()))
    is_news = sut_label == "Multilingual News RAG"
    sut_note = (
        "10k multilingual news articles · hybrid dense+BM25+RRF · Nordic golden pairs"
        if is_news else "6-doc corpus · dense retrieval · English golden set"
    )
    st.markdown(
        f'<div class="label">{icon("gauge", 14)} {_escape(sut_label)} '
        f'<span style="text-transform:none;letter-spacing:0">· {sut_note}</span></div>',
        unsafe_allow_html=True,
    )

    if not results:
        cmd = (
            "SUT_MODE=news EMBEDDING_MODEL=intfloat/multilingual-e5-small python evals/runners/run_ragas.py "
            "--golden evals/datasets/golden_news.json --limit 4 --output results/ragas_news.json"
            if is_news else
            "python evals/runners/run_ragas.py --limit 3 --output results/ragas_sample.json"
        )
        st.markdown(
            f'<div class="card"><div class="reason">{icon("alert", 14)} No results for this system yet. '
            f"Run <code>{cmd}</code>.</div></div>",
            unsafe_allow_html=True,
        )
        return

    # Thresholds are anchored per system: the Nordic news RAG is gated against its
    # own baseline, not the English corpus' (an English-centric judge under-rates
    # Nordic faithfulness, so the English bar is the wrong ruler for it).
    profile = "nordic" if is_news else None
    table = resolve_thresholds(profile)
    passed, _ = evaluate_gates(results, layer="nightly", thresholds=table)
    g_color = OK if passed else BAD
    g_text = "GATE: PASS — merge allowed" if passed else "GATE: FAIL — merge blocked"
    src = ", ".join(files) if files else "—"
    thr_note = (
        "Nordic baseline profile · gate anchored to this system"
        if profile else "English-corpus thresholds"
    )

    cards = ""
    for group_title, metrics in METRIC_GROUPS:
        present = [m for m in metrics if m in results]
        if not present:
            continue
        cells = "".join(_metric_card(m, results[m], table) for m in present)
        cards += f'<div class="label" style="margin-top:.4rem">{group_title}</div><div class="mgrid">{cells}</div>'

    st.markdown(
        f"""
<div class="card">
  <div class="gate" style="border-color:{g_color}66; background:{g_color}0d">
    <span class="g-dot" style="background:{g_color}; box-shadow:0 0 14px {g_color}"></span>
    <div><div class="g-title" style="color:{g_color}">{g_text}</div>
      <div class="g-sub">{thr_note} · source <code>{_escape(src)}</code> · bar = score, tick = gate</div></div>
  </div>
  {cards}
</div>
""",
        unsafe_allow_html=True,
    )


# ── Performance panel: measured per-stage latency ─────────────────────────────
_STAGE_COLORS = {"retrieve": ICE, "rerank": VIOLET, "generate": OK, "critique": WARN}


def render_performance() -> None:
    path = ROOT / "results" / "latency.json"
    if not path.is_file():
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        stages = data["stage_breakdown_ms"]
    except Exception:
        return

    st.markdown(
        f'<div class="label" style="margin-top:1.4rem">{icon("gauge", 14)} Self-Healing RAG · measured latency '
        f'<span style="text-transform:none;letter-spacing:0">· {int(data.get("n_queries", 0))} queries, '
        "happy path</span></div>",
        unsafe_allow_html=True,
    )

    p50, p95 = data.get("latency_p50_ms", 0), data.get("latency_p95_ms", 0)
    under = p50 < 3000
    bcolor = OK if under else WARN
    order = ["retrieve", "rerank", "generate", "critique"]
    scale = max((stages[s]["p50"] for s in order if s in stages), default=1) or 1

    rows = ""
    for s in order:
        if s not in stages:
            continue
        b = stages[s]
        w = min(b["p50"] / scale * 100, 100)
        c = _STAGE_COLORS.get(s, MUTED)
        rows += (
            f'<div class="latrow"><span class="ln">{s}</span>'
            f'<div class="track"><div class="fill" style="width:{w:.1f}%; background:{c}"></div></div>'
            f'<span class="lv"><b>{b["p50"]:.0f}</b> ms · p95 {b["p95"]:.0f}</span></div>'
        )

    st.markdown(
        f"""
<div class="card">
  <div class="perf-head">
    <span class="perf-big">{p50:.0f} ms</span>
    <span class="perf-sub">median end-to-end · p95 {p95:.0f} ms · mean {data.get("latency_mean_ms", 0):.0f} ms</span>
    <span class="perf-budget" style="color:{bcolor}; border-color:{bcolor}66">
      {"✓ within" if under else "⚠ tail over"} 3000 ms budget</span>
  </div>
  {rows}
  <div class="reason" style="margin-top:.7rem">Retrieval is ~<b>32 ms</b> (10k hybrid index); the two LLM
  calls and the Cohere rerank dominate. The p95 tail is rerank network variance on the free tier —
  a local cross-encoder (BGE-reranker) removes it.</div>
</div>
""",
        unsafe_allow_html=True,
    )


# ── Live judge: one fast LLM call returning a JSON rubric ──────────────────────
_JUDGE_METRICS = ["faithfulness", "answer_relevancy", "geval_on_topic"]

_JUDGE_PROMPT = """You are a strict evaluation judge for a RAG system. Score the ANSWER \
against ONLY the retrieved CONTEXT and the QUESTION, on three axes from 0.0 to 1.0:
- faithfulness: are the answer's claims supported by the context? (unsupported claims lower it)
- answer_relevancy: does the answer actually address the question?
- geval_on_topic: does the answer stay within the scope of the context, without drifting?

If the answer is an honest refusal ("I don't know") because the context lacks the info, that is
faithful (high faithfulness) but low relevancy. Return ONLY compact JSON, no prose, with keys:
faithfulness, answer_relevancy, geval_on_topic (floats), verdict (one of grounded|weak|ungrounded),
reason (one short sentence).

QUESTION: {question}
CONTEXT:
{context}
ANSWER: {answer}
"""


def _parse_judge_json(raw: str) -> dict:
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    data = json.loads(match.group(0)) if match else {}
    out = {}
    for m in _JUDGE_METRICS:
        try:
            out[m] = max(0.0, min(1.0, float(data.get(m, 0.0))))
        except (TypeError, ValueError):
            out[m] = 0.0
    out["verdict"] = str(data.get("verdict", "n/a"))
    out["reason"] = str(data.get("reason", "—"))
    return out


def judge_answer(question: str, answer: str, contexts: list) -> dict:
    context_text = "\n".join(f"- {c[:600]}" for c in (contexts or [])[:4]) or "(no context retrieved)"
    prompt = _JUDGE_PROMPT.format(question=question, context=context_text, answer=answer)
    raw = get_chat_model().invoke(prompt).content
    return _parse_judge_json(raw)


def run_sut(question: str) -> dict:
    from src.graph.graph import answer_query  # Project 1's entry point

    return answer_query(question)


# ── Renderers for the two live columns ────────────────────────────────────────
def _sources_html(sources: list) -> str:
    if not sources:
        return ""
    chips = "".join(f'<span class="src">{icon("doc", 12)}{_escape(s)}</span>' for s in sources)
    return f'<div style="margin-bottom:.9rem">{chips}</div>'


def _contexts_html(contexts: list) -> str:
    if not contexts:
        return ""
    blocks = ""
    for i, ctx in enumerate(contexts, start=1):
        preview = _escape(ctx[:360] + ("…" if len(ctx) > 360 else ""))
        blocks += f'<div class="ctx"><span class="n">[{i}]</span>{preview}</div>'
    return f'<div class="label" style="margin-top:.4rem">{icon("search", 13)} Retrieved context</div>{blocks}'


def render_p1(result: dict) -> None:
    grounded = result.get("grounded", False)
    color = OK if grounded else WARN
    status_icon = icon("check") if grounded else icon("alert")
    status_text = "Grounded answer" if grounded else "Refused — could not ground"
    verdict = (result.get("critique") or "n/a").lower()

    st.markdown(
        f'<div class="colhead"><span class="p1">{icon("loop", 14)}</span>'
        f'<span class="p1">Project 1 · produces</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
<div class="card">
  <span class="pill" style="color:{color}; background:{color}1a; border:1px solid {color}55">{status_icon} {status_text}</span>
  <div class="answer-card">{_escape(result.get("answer", ""))}</div>
  <div class="stats">
    <div class="stat"><div class="k">Self-critique</div><div class="v" style="color:{color}">{verdict.upper()}</div></div>
    <div class="stat"><div class="k">Retries</div><div class="v">{result.get("retries", 0)}</div></div>
    <div class="stat"><div class="k">Sources</div><div class="v">{len(result.get("sources", []))}</div></div>
  </div>
  <div class="reason"><b>Critic reasoning:</b> {_escape(result.get("critique_reason") or "—")}</div>
  {_sources_html(result.get("sources", []))}
  {_contexts_html(result.get("contexts", []))}
</div>
""",
        unsafe_allow_html=True,
    )


def render_judge(scores: dict, p1_grounded: bool) -> None:
    gate_input = {m: scores[m] for m in _JUDGE_METRICS}
    passed, _ = evaluate_gates(gate_input, layer="pr")
    g_color = OK if passed else BAD
    g_text = "GATE: PASS" if passed else "GATE: FAIL"

    rows = ""
    for m in _JUDGE_METRICS:
        ok, _ = check_metric(m, scores[m])
        vcolor = OK if ok else BAD
        tick = icon("check", 14) if ok else icon("x", 14)
        rows += (
            f'<div class="jrow"><div class="j-left"><span style="color:{vcolor}">{tick}</span>'
            f'<span><div class="j-name">{m}</div><div class="j-thr">gate {_fmt_threshold(m)}</div></span></div>'
            f'<div class="j-val" style="color:{vcolor}">{scores[m]:.2f}</div></div>'
        )

    agree = (passed and p1_grounded) or (not passed and not p1_grounded)
    agree_txt = (
        "Independent judge <b>agrees</b> with Project 1's self-critique."
        if agree
        else "Independent judge <b>disagrees</b> with Project 1 — worth a look."
    )

    st.markdown(
        f'<div class="colhead"><span class="p2">{icon("scale", 14)}</span>'
        f'<span class="p2">Project 2 · judges (independent)</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
<div class="card">
  <div class="gate" style="border-color:{g_color}66; background:{g_color}0d; padding:.85rem 1.1rem">
    <span class="g-dot" style="background:{g_color}; box-shadow:0 0 12px {g_color}"></span>
    <div><div class="g-title" style="color:{g_color}; font-size:1.1rem">{g_text}</div>
      <div class="g-sub">verdict <code>{_escape(scores.get("verdict"))}</code> · judge
      <code>{_escape(os.getenv("JUDGE_MODEL", "groq default"))}</code></div></div>
  </div>
  {rows}
  <div class="reason" style="margin-top:.8rem"><b>Judge reasoning:</b> {_escape(scores.get("reason"))}</div>
  <div class="reason">{agree_txt}</div>
</div>
""",
        unsafe_allow_html=True,
    )


# ── Masthead ──────────────────────────────────────────────────────────────────
st.markdown(
    f"""
<div class="eyebrow">
  <span class="tag">{icon('scale', 13)} LLM-EVAL · CI/CD QUALITY HARNESS</span>
  <span class="sut">system under test <b>Self-Healing RAG</b></span>
</div>
<div class="masthead">
  <h1>Unit tests, but for LLM quality.</h1>
  <div class="side">
    <p>Every change is graded on faithfulness, relevancy and scope — then a
    <b style="color:{TEXT}">hard quality gate</b> blocks any regression, exactly like CI.</p>
    <div class="flow">
      <span>System under test</span><span class="arrow">→</span>
      <span>RAGAS / DeepEval</span><span class="arrow">→</span>
      <span>Thresholds</span><span class="arrow">→</span>
      <span>Gate</span>
    </div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# ── Dashboard ─────────────────────────────────────────────────────────────────
_sut = st.radio(
    "System under test", list(SUT_RESULT_FILES.keys()),
    horizontal=True, label_visibility="collapsed", key="sut_sel",
)
render_dashboard(_sut)

# ── Performance (measured latency) ────────────────────────────────────────────
render_performance()

# ── Live grade ────────────────────────────────────────────────────────────────
st.markdown(
    f'<div class="label" style="margin-top:1.4rem">{icon("spark", 14)} Grade a live question</div>',
    unsafe_allow_html=True,
)
st.session_state.setdefault("q", "")


def _set_q(q: str) -> None:
    st.session_state.q = q


chip_cols = st.columns(4)
for i, example in enumerate(EXAMPLES):
    chip_cols[i].button(example, key=f"ex_{i}", on_click=_set_q, args=(example,), use_container_width=True)

st.text_input(
    "Ask a question",
    key="q",
    placeholder="Ask about the indexed corpus — or something off-topic to watch it refuse…",
    label_visibility="collapsed",
)
run = st.button("Answer, then judge it", type="primary", use_container_width=True)

if run and st.session_state.q.strip():
    try:
        with st.spinner("Project 1 is retrieving, generating and self-critiquing…"):
            result = run_sut(st.session_state.q)
        with st.spinner("Project 2 is judging the answer independently…"):
            scores = judge_answer(st.session_state.q, result.get("answer", ""), result.get("contexts", []))
        left, right = st.columns(2, gap="large")
        with left:
            render_p1(result)
        with right:
            render_judge(scores, bool(result.get("grounded", False)))
    except Exception as exc:  # noqa: BLE001
        st.markdown(
            f"""
<div class="card" style="border-color:{BAD}66">
  <span class="pill" style="color:{BAD}; background:{BAD}1a; border:1px solid {BAD}55">{icon('alert')} Pipeline error</span>
  <div class="answer-card mono" style="font-size:.88rem; color:#FFB3B3">{_escape(str(exc))}</div>
  <div class="reason" style="margin-top:.8rem">Check that Qdrant is running and ingested, and that
  <code>GROQ_API_KEY</code> / <code>COHERE_API_KEY</code> are set (Streamlit secrets when deployed,
  or <code>.env</code> locally).</div>
</div>
""",
            unsafe_allow_html=True,
        )
elif run:
    st.markdown(
        f'<div class="reason">{icon("alert", 14)} Enter a question or pick an example above.</div>',
        unsafe_allow_html=True,
    )

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    f"""
<div style="text-align:center; margin-top:2.4rem; color:{MUTED}; font-family:'Spline Sans Mono',monospace; font-size:.72rem; line-height:1.9">
  Project 2 measures <a href="https://github.com/fahadcs321/self-healing-rag" style="color:{OK}; text-decoration:none">Self-Healing RAG</a>
  · DeepEval + RAGAS + LiteLLM · Groq judge, local embeddings<br>
  <a href="https://github.com/fahadcs321/llm-eval-pipeline" style="color:{ICE}; text-decoration:none">github.com/fahadcs321/llm-eval-pipeline</a>
</div>
""",
    unsafe_allow_html=True,
)
