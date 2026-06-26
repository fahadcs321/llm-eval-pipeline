"""Integrity tests for the golden datasets — they gate real eval runs, so a
malformed entry must fail fast in CI rather than silently skewing scores."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

_DATASETS = Path("evals/datasets")
_RAG = _DATASETS / "golden_rag.json"
_EDGE = _DATASETS / "golden_edge.json"
_GENERAL = _DATASETS / "golden_general.json"


def _load(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@pytest.mark.parametrize("path", [_RAG, _EDGE, _GENERAL])
def test_dataset_is_nonempty_list(path):
    data = _load(path)
    assert isinstance(data, list)
    assert len(data) > 0


@pytest.mark.parametrize("path", [_RAG, _EDGE, _GENERAL])
def test_every_entry_has_required_fields(path):
    for i, item in enumerate(_load(path)):
        assert "question" in item, f"{path.name}[{i}] missing question"
        assert "ground_truth" in item, f"{path.name}[{i}] missing ground_truth"
        assert isinstance(item["question"], str)
        assert isinstance(item["ground_truth"], str)
        assert item["ground_truth"].strip(), f"{path.name}[{i}] empty ground_truth"


def test_rag_dataset_has_no_duplicate_questions():
    questions = [g["question"].strip().lower() for g in _load(_RAG)]
    assert len(questions) == len(set(questions)), "duplicate questions in golden_rag.json"


def test_rag_dataset_is_reasonably_sized():
    assert len(_load(_RAG)) >= 25
