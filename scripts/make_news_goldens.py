"""
make_news_goldens.py — Synthesize a multilingual golden set from the news corpus.

Golden sets for a Nordic/multilingual archive can't be English-only. This builds
one straight from the real news index: for articles that carry a reference
summary (Danish + MLSUM), the judge writes ONE natural question, in the article's
own language, that the summary answers. The summary becomes the ground truth.

That gives faithfulness/relevancy (reference-free) plus context recall/precision
(reference-based) on genuinely multilingual pairs — and demonstrates the
"synthesized golden dataset" idea on Nordic-language data.

Usage:
    python scripts/make_news_goldens.py --per-language 3
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evals.judge import get_chat_model  # noqa: E402

LANG_NAMES = {"da": "Danish", "de": "German", "fr": "French", "es": "Spanish"}

_CORPUS = Path(
    os.getenv("SELF_HEALING_RAG_PATH", str(Path(__file__).resolve().parents[2] / "self_healing_rag"))
) / "scripts" / "retriever_demo" / "artifacts" / "corpus.pkl"


def _question_from_summary(summary: str, language: str) -> str:
    lang = LANG_NAMES.get(language, language)
    prompt = (
        f"You are given a news summary in {lang}. Write exactly ONE clear, natural "
        f"question, in {lang}, that this summary answers. Return only the question, "
        f"no preamble.\n\nSUMMARY:\n{summary[:800]}"
    )
    return get_chat_model().invoke(prompt).content.strip().strip('"')


def build(per_language: int) -> None:
    with open(_CORPUS, "rb") as f:
        payloads = pickle.load(f)["payloads"]

    goldens: list[dict] = []
    for lang in ("da", "de", "fr", "es"):
        picked = 0
        for p in payloads:
            if picked >= per_language:
                break
            if p["language"] != lang:
                continue
            summary = (p.get("summary") or "").strip()
            if len(summary) < 120:  # need a real reference
                continue
            try:
                question = _question_from_summary(summary, lang)
            except Exception as exc:  # noqa: BLE001
                print(f"  [skip] {lang}: {exc}")
                continue
            if not question:
                continue
            goldens.append(
                {"question": question, "ground_truth": summary, "language": lang,
                 "source_title": p["title"][:80]}
            )
            print(f"  [{lang}] {question[:70]}")
            picked += 1

    out = Path("evals/datasets/golden_news.json")
    out.write_text(json.dumps(goldens, ensure_ascii=False, indent=2), encoding="utf-8")
    langs: dict[str, int] = {}
    for g in goldens:
        langs[g["language"]] = langs.get(g["language"], 0) + 1
    print(f"\nWrote {len(goldens)} multilingual golden pairs → {out}  ({langs})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-language", type=int, default=3)
    args = ap.parse_args()
    build(args.per_language)
