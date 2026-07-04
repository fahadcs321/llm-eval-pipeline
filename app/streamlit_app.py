"""
streamlit_app.py — "Build ↔ Measure" demo for the LLM Eval CI/CD pipeline.

One page, three parts, matching Project 1's "code dark + run green" design system:

  1. Quality-gate dashboard — the latest cached eval run (results/*.json) rendered
     as metric cards vs thresholds, with the real CI gate verdict on top.
  2. Live "grade a question" — two columns:
       left  = Project 1 (Self-Healing RAG) answers + shows its own self-heal trace
       right = Project 2 (this pipeline) judges that answer independently and
               applies the same threshold gate.
     The story: P1 says "I'm grounded" (self-critique); P2 verifies it with an
     external LLM-as-judge and a hard pass/fail gate.

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
# reads os.getenv (the judge config, Project 1's config). Streamlit does not reliably
# expose secrets as env vars, so we do it explicitly. .strip() guards pasted newlines.
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
from evals.metrics.thresholds import THRESHOLDS, check_metric  # noqa: E402

# ── Page setup ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LLM Eval · measures Project 1",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Design tokens (identical to Project 1) ────────────────────────────────────
BG = "#0F172A"
SURFACE = "#1E293B"
SURFACE_2 = "#172033"
BORDER = "#334155"
ACCENT = "#22C55E"
DANGER = "#EF4444"
WARN = "#F59E0B"
TEXT = "#F8FAFC"
MUTED = "#94A3B8"

EXAMPLES = [
    "What does the RAGAS faithfulness metric measure?",
    "What does LangGraph enable that a LangChain chain cannot?",
    "What is the difference between dense and sparse retrieval?",
    "Who won the 2022 FIFA World Cup?",  # off-corpus → shows honest refusal + a low judge score
]

# Metrics we surface in the dashboard, grouped by source (only shown if present).
METRIC_GROUPS = [
    ("RAGAS · retrieval quality", ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]),
    ("DeepEval · safety & scope", ["hallucination_rate", "toxicity_score", "bias_score", "geval_on_topic"]),
    ("Cost & latency", ["cost_per_query_usd", "latency_p95_ms"]),
]


# ── SVG icon set (Lucide-style, inherit currentColor) ─────────────────────────
def icon(name: str, size: int = 18) -> str:
    paths = {
        "gauge": '<path d="m12 14 4-4"/><path d="M3.34 19a10 10 0 1 1 17.32 0"/>',
        "scale": '<path d="m16 16 3-8 3 8c-.87.65-1.92 1-3 1s-2.13-.35-3-1Z"/>'
        '<path d="m2 16 3-8 3 8c-.87.65-1.92 1-3 1s-2.13-.35-3-1Z"/>'
        '<path d="M7 21h10"/><path d="M12 3v18"/><path d="M3 7h2c2 0 5-1 7-2 2 1 5 2 7 2h2"/>',
        "check": '<path d="M20 6 9 17l-5-5"/>',
        "alert": '<path d="M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 '
        '2 0 0 0-3.4 0Z"/><path d="M12 9v4"/><path d="M12 17h.01"/>',
        "x": '<path d="M18 6 6 18"/><path d="m6 6 12 12"/>',
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


# ── Global styles (Project 1 tokens + eval-specific components) ────────────────
st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;500;600&family=Fira+Sans:wght@300;400;500;600;700&display=swap');
:root {{ --bg:{BG}; --surface:{SURFACE}; --border:{BORDER}; --accent:{ACCENT}; --text:{TEXT}; --muted:{MUTED}; }}
.stApp {{
  background:
    radial-gradient(900px 500px at 50% -10%, rgba(34,197,94,0.10), transparent 60%),
    radial-gradient(700px 400px at 100% 0%, rgba(56,189,248,0.06), transparent 55%),
    {BG};
  font-family: 'Fira Sans', system-ui, sans-serif;
}}
#MainMenu, header[data-testid="stHeader"], footer {{ display: none; }}
.block-container {{ padding-top: 2.5rem; padding-bottom: 3rem; max-width: 1180px; }}
code, .mono {{ font-family: 'Fira Code', monospace; }}

.hero {{ text-align: center; margin-bottom: 1.5rem; }}
.hero .badge {{
  display:inline-flex; align-items:center; gap:.5rem; font-family:'Fira Code',monospace;
  font-size:.72rem; letter-spacing:.06em; text-transform:uppercase; color:{ACCENT};
  background:rgba(34,197,94,0.10); border:1px solid rgba(34,197,94,0.30);
  padding:.35rem .8rem; border-radius:999px; margin-bottom:1rem;
}}
.hero h1 {{
  font-size:2.5rem; font-weight:700; line-height:1.1; margin:0 0 .6rem;
  background:linear-gradient(180deg,#FFFFFF,#B6C2D4);
  -webkit-background-clip:text; -webkit-text-fill-color:transparent;
}}
.hero p {{ color:{MUTED}; font-size:1.0rem; max-width:640px; margin:0 auto; line-height:1.6; }}
.flow {{ display:flex; flex-wrap:wrap; justify-content:center; gap:.4rem; margin:1.1rem 0 .25rem;
  font-family:'Fira Code',monospace; font-size:.76rem; }}
.flow span {{ color:{TEXT}; background:{SURFACE}; border:1px solid {BORDER}; padding:.28rem .6rem; border-radius:8px; }}
.flow .arrow {{ color:{MUTED}; border:none; background:none; padding:.28rem .1rem; }}

.label {{ display:flex; align-items:center; gap:.5rem; color:{MUTED}; font-family:'Fira Code',monospace;
  font-size:.74rem; letter-spacing:.08em; text-transform:uppercase; margin:.2rem 0 .6rem; }}
.label svg {{ color:{ACCENT}; }}

.card {{ background:{SURFACE}; border:1px solid {BORDER}; border-radius:16px; padding:1.3rem 1.4rem;
  box-shadow:0 10px 30px rgba(0,0,0,0.25); }}
.answer-card {{ font-size:1.02rem; line-height:1.7; color:{TEXT}; }}
.pill {{ display:inline-flex; align-items:center; gap:.45rem; font-weight:600; font-size:.82rem;
  padding:.35rem .75rem; border-radius:999px; margin-bottom:.9rem; }}

/* gate banner */
.gate {{ display:flex; align-items:center; gap:.9rem; border-radius:16px; padding:1.05rem 1.3rem;
  border:1px solid {BORDER}; margin-bottom:1rem; }}
.gate .g-icon {{ display:flex; }}
.gate .g-title {{ font-family:'Fira Code',monospace; font-weight:600; font-size:1.05rem; }}
.gate .g-sub {{ color:{MUTED}; font-size:.82rem; margin-top:.1rem; }}

/* metric grid */
.mgrid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); gap:.7rem; margin:.2rem 0 1rem; }}
.metric {{ background:{SURFACE_2}; border:1px solid {BORDER}; border-left:3px solid {BORDER};
  border-radius:12px; padding:.8rem .95rem; }}
.metric.pass {{ border-left-color:{ACCENT}; }}
.metric.fail {{ border-left-color:{DANGER}; }}
.metric .m-top {{ display:flex; align-items:center; justify-content:space-between; }}
.metric .m-name {{ font-family:'Fira Code',monospace; font-size:.7rem; letter-spacing:.04em;
  text-transform:uppercase; color:{MUTED}; }}
.metric .m-val {{ font-family:'Fira Code',monospace; font-size:1.35rem; font-weight:600; color:{TEXT}; margin:.25rem 0 .1rem; }}
.metric .m-thr {{ font-family:'Fira Code',monospace; font-size:.72rem; color:{MUTED}; }}

/* two-column headers */
.colhead {{ display:flex; align-items:center; gap:.55rem; font-family:'Fira Code',monospace;
  font-size:.72rem; letter-spacing:.06em; text-transform:uppercase; padding:.4rem .2rem .7rem; }}
.colhead .p1 {{ color:{ACCENT}; }}
.colhead .p2 {{ color:#38BDF8; }}

.stats {{ display:grid; grid-template-columns:repeat(3,1fr); gap:.75rem; margin:.25rem 0 1rem; }}
.stat {{ background:{SURFACE_2}; border:1px solid {BORDER}; border-radius:12px; padding:.85rem 1rem; text-align:center; }}
.stat .k {{ font-family:'Fira Code',monospace; font-size:.66rem; letter-spacing:.06em; text-transform:uppercase; color:{MUTED}; }}
.stat .v {{ font-family:'Fira Code',monospace; font-size:1.2rem; font-weight:600; color:{TEXT}; margin-top:.25rem; }}
.reason {{ color:{MUTED}; font-size:.9rem; line-height:1.6; border-left:2px solid {BORDER}; padding-left:.85rem; margin:.2rem 0 1rem; }}
.reason b {{ color:{TEXT}; }}
.src {{ display:inline-flex; align-items:center; gap:.4rem; font-family:'Fira Code',monospace; font-size:.76rem;
  color:{TEXT}; background:{SURFACE_2}; border:1px solid {BORDER}; padding:.3rem .6rem; border-radius:8px; margin:0 .4rem .4rem 0; }}
.src svg {{ color:{ACCENT}; }}
.ctx {{ background:{SURFACE_2}; border:1px solid {BORDER}; border-left:3px solid {ACCENT}; border-radius:10px;
  padding:.7rem .9rem; margin-bottom:.55rem; color:#CBD5E1; font-size:.86rem; line-height:1.6; }}
.ctx .n {{ font-family:'Fira Code',monospace; color:{ACCENT}; font-weight:600; margin-right:.4rem; }}

/* judge rows */
.jrow {{ display:flex; align-items:center; justify-content:space-between; gap:.6rem;
  background:{SURFACE_2}; border:1px solid {BORDER}; border-radius:10px; padding:.6rem .85rem; margin-bottom:.5rem; }}
.jrow .j-left {{ display:flex; align-items:center; gap:.55rem; }}
.jrow .j-name {{ font-family:'Fira Code',monospace; font-size:.82rem; color:{TEXT}; }}
.jrow .j-thr {{ font-family:'Fira Code',monospace; font-size:.7rem; color:{MUTED}; }}
.jrow .j-val {{ font-family:'Fira Code',monospace; font-size:1.0rem; font-weight:600; }}

.stTextInput input {{ background:{SURFACE}!important; border:1px solid {BORDER}!important; border-radius:12px!important;
  color:{TEXT}!important; font-size:1rem!important; }}
.stTextInput input:focus {{ border-color:{ACCENT}!important; box-shadow:0 0 0 3px rgba(34,197,94,0.18)!important; }}
div.stButton > button {{ border-radius:10px; border:1px solid {BORDER}; background:{SURFACE}; color:{TEXT};
  font-family:'Fira Code',monospace; font-size:.8rem; font-weight:500; transition:border-color .2s,color .2s,background .2s; }}
div.stButton > button:hover {{ border-color:{ACCENT}; color:{ACCENT}; background:{SURFACE_2}; }}
div.stButton > button[kind="primary"] {{ background:{ACCENT}; border:1px solid {ACCENT}; color:#06210F; font-weight:700; }}
div.stButton > button[kind="primary"]:hover {{ background:#1FB055; border-color:#1FB055; color:#06210F; }}
@media (prefers-reduced-motion: reduce) {{ * {{ transition:none!important; animation:none!important; }} }}
</style>
""",
    unsafe_allow_html=True,
)


# ── Small HTML helpers (shared with Project 1's renderer) ─────────────────────
def _escape(text: str) -> str:
    return (str(text) if text is not None else "").replace("<", "&lt;").replace(">", "&gt;")


def _sources_html(sources: list) -> str:
    if not sources:
        return ""
    chips = "".join(f'<span class="src">{icon("doc", 13)}{_escape(s)}</span>' for s in sources)
    return f'<div style="margin-bottom:.9rem">{chips}</div>'


def _contexts_html(contexts: list) -> str:
    if not contexts:
        return ""
    blocks = ""
    for i, ctx in enumerate(contexts, start=1):
        preview = _escape(ctx[:360] + ("…" if len(ctx) > 360 else ""))
        blocks += f'<div class="ctx"><span class="n">[{i}]</span>{preview}</div>'
    return f'<div class="label" style="margin-top:.4rem">{icon("search", 13)} Retrieved context</div>{blocks}'


def _fmt_value(metric: str, value: float) -> str:
    if metric == "cost_per_query_usd":
        return f"${value:.4f}"
    if metric == "latency_p95_ms":
        return f"{value:.0f} ms"
    return f"{value:.3f}"


def _fmt_threshold(metric: str) -> str:
    spec = THRESHOLDS.get(metric, {})
    if spec.get("direction") == "lower_is_better":
        return f"≤ {spec.get('max')}"
    return f"≥ {spec.get('min')}"


# ── Dashboard: load + render the latest cached eval run, per system-under-test ─
# Each SUT reads its own result files so the two are never blended.
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
        f'<div class="label">{icon("gauge", 14)} {_escape(sut_label)} — quality gate '
        f'<span style="color:{MUTED};text-transform:none;letter-spacing:0"> · {sut_note}</span></div>',
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

    passed, _ = evaluate_gates(results, layer="nightly")
    g_color, g_bg = (ACCENT, "rgba(34,197,94,0.10)") if passed else (DANGER, "rgba(239,68,68,0.10)")
    g_icon = icon("check", 22) if passed else icon("x", 22)
    g_text = "GATE: PASS — merge allowed" if passed else "GATE: FAIL — merge blocked"
    src = ", ".join(files) if files else "—"

    cards = ""
    for group_title, metrics in METRIC_GROUPS:
        present = [m for m in metrics if m in results]
        if not present:
            continue
        cells = ""
        for m in present:
            ok, _ = check_metric(m, results[m])
            cls = "pass" if ok else "fail"
            tick = (
                f'<span style="color:{ACCENT}">{icon("check", 15)}</span>'
                if ok
                else f'<span style="color:{DANGER}">{icon("x", 15)}</span>'
            )
            cells += (
                f'<div class="metric {cls}"><div class="m-top"><span class="m-name">{m}</span>{tick}</div>'
                f'<div class="m-val">{_fmt_value(m, results[m])}</div>'
                f'<div class="m-thr">threshold {_fmt_threshold(m)}</div></div>'
            )
        cards += f'<div class="label" style="margin-top:.3rem">{group_title}</div><div class="mgrid">{cells}</div>'

    st.markdown(
        f"""
<div class="card">
  <div class="gate" style="border-color:{g_color}; background:{g_bg}">
    <span class="g-icon" style="color:{g_color}">{g_icon}</span>
    <div><div class="g-title" style="color:{g_color}">{g_text}</div>
      <div class="g-sub">Thresholds from <code>evals/metrics/thresholds.py</code> · source: <code>{_escape(src)}</code></div></div>
  </div>
  {cards}
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


# ── Project 1 runner (full trace) ─────────────────────────────────────────────
def run_sut(question: str) -> dict:
    from src.graph.graph import answer_query  # Project 1's entry point

    return answer_query(question)


# ── Renderers for the two columns ─────────────────────────────────────────────
def render_p1(result: dict) -> None:
    grounded = result.get("grounded", False)
    color, bg = (ACCENT, "rgba(34,197,94,0.12)") if grounded else (WARN, "rgba(245,158,11,0.12)")
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
  <span class="pill" style="color:{color}; background:{bg}">{status_icon} {status_text}</span>
  <div class="answer-card">{_escape(result.get("answer", ""))}</div>
  <div class="stats" style="margin-top:1rem">
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
    g_color = ACCENT if passed else DANGER
    g_icon = icon("check", 20) if passed else icon("x", 20)
    g_text = "GATE: PASS" if passed else "GATE: FAIL"

    rows = ""
    for m in _JUDGE_METRICS:
        ok, _ = check_metric(m, scores[m])
        vcolor = ACCENT if ok else DANGER
        tick = icon("check", 15) if ok else icon("x", 15)
        rows += (
            f'<div class="jrow"><div class="j-left"><span style="color:{vcolor}">{tick}</span>'
            f'<span><div class="j-name">{m}</div><div class="j-thr">threshold {_fmt_threshold(m)}</div></span></div>'
            f'<div class="j-val" style="color:{vcolor}">{scores[m]:.2f}</div></div>'
        )

    # Agreement note: does the independent judge agree with P1's self-critique?
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
  <div class="gate" style="border-color:{g_color}; background:rgba(0,0,0,0.15)">
    <span class="g-icon" style="color:{g_color}">{g_icon}</span>
    <div><div class="g-title" style="color:{g_color}">{g_text}</div>
      <div class="g-sub">verdict: <code>{_escape(scores.get("verdict"))}</code> · judge: <code>{_escape(os.getenv("JUDGE_MODEL", "groq default"))}</code></div></div>
  </div>
  {rows}
  <div class="reason" style="margin-top:.8rem"><b>Judge reasoning:</b> {_escape(scores.get("reason"))}</div>
  <div class="reason">{agree_txt}</div>
</div>
""",
        unsafe_allow_html=True,
    )


# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown(
    f"""
<div class="hero">
  <div class="badge">{icon('scale', 14)} LLM-Eval · measures Project 1</div>
  <h1>Unit tests, but for LLM quality.</h1>
  <p>An automated eval pipeline that grades the <b>Self-Healing RAG</b> system on faithfulness,
  relevancy and scope — then a <b>hard quality gate</b> blocks any regression, just like CI.</p>
  <div class="flow">
    <span>System under test</span><span class="arrow">→</span>
    <span>RAGAS / DeepEval</span><span class="arrow">→</span>
    <span>Thresholds</span><span class="arrow">→</span>
    <span>Gate</span>
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
<div class="card" style="border-color:{DANGER}">
  <span class="pill" style="color:{DANGER}; background:rgba(239,68,68,0.12)">{icon('alert')} Pipeline error</span>
  <div class="answer-card mono" style="font-size:.88rem; color:#FCA5A5">{_escape(str(exc))}</div>
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
<div style="text-align:center; margin-top:2.5rem; color:{MUTED}; font-family:'Fira Code',monospace; font-size:.74rem; line-height:1.9">
  Project 2 measures <a href="https://github.com/fahadcs321/self-healing-rag" style="color:{ACCENT}; text-decoration:none">Self-Healing RAG</a>
  · DeepEval + RAGAS + LiteLLM · Groq judge, local embeddings<br>
  <a href="https://github.com/fahadcs321/llm-eval-pipeline" style="color:{ACCENT}; text-decoration:none">github.com/fahadcs321/llm-eval-pipeline</a>
</div>
""",
    unsafe_allow_html=True,
)
