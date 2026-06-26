"""
synthesize_goldens.py — Auto-generate golden Q&A pairs from your documents.
Uses DeepEval's Synthesizer to expand from 30 hand-crafted pairs to 150+,
covering edge cases, paraphrases, and multi-hop questions automatically.

Usage:
  python scripts/synthesize_goldens.py --docs data/raw/ --output evals/datasets/golden_rag.json --count 120
"""

import argparse
import json
from pathlib import Path

from deepeval.synthesizer import Synthesizer
from deepeval.synthesizer.config import (
    Evolution,
    EvolutionConfig,
)
from langchain_community.document_loaders import PyPDFLoader, TextLoader


def load_documents(docs_dir: str) -> list[str]:
    """Load all PDF and txt files from a directory."""
    docs = []
    path = Path(docs_dir)

    for f in path.rglob("*.pdf"):
        loader = PyPDFLoader(str(f))
        pages = loader.load()
        docs.extend([p.page_content for p in pages])

    for f in path.rglob("*.txt"):
        loader = TextLoader(str(f), encoding="utf-8")
        pages = loader.load()
        docs.extend([p.page_content for p in pages])

    print(f"Loaded {len(docs)} document chunks from {docs_dir}")
    return docs


def synthesize(docs_dir: str, output_path: str, count: int):
    docs = load_documents(docs_dir)

    synthesizer = Synthesizer()

    # Evolution config: generates diverse question types
    # - Simple: direct factual questions
    # - Multi-hop: requires connecting multiple chunks
    # - Paraphrase: same question, different wording
    # - Negation: "What does X NOT do?"
    # - Conditional: "If X, then what?"
    evolution_config = EvolutionConfig(
        evolutions={
            Evolution.SIMPLE: 0.3,
            Evolution.MULTI_HOP: 0.2,
            Evolution.PARAPHRASE: 0.2,
            Evolution.NEGATION: 0.15,
            Evolution.CONDITIONAL: 0.15,
        }
    )

    print(f"Synthesizing {count} golden Q&A pairs...")

    synthesizer.generate_goldens_from_docs(
        document_paths=[str(p) for p in Path(docs_dir).rglob("*") if p.suffix in [".pdf", ".txt"]],
        include_expected_output=True,
        max_goldens_per_document=max(5, count // max(len(docs) // 5, 1)),
        evolution_config=evolution_config,
    )

    goldens = synthesizer.synthetic_goldens

    # Convert to the JSON format used by our eval pipeline
    output = [
        {
            "question": g.input,
            "ground_truth": g.expected_output,
            "context_source": "synthesized",
            "evolution_type": g.additional_metadata.get("evolution_type", "simple")
            if g.additional_metadata
            else "simple",
        }
        for g in goldens[:count]
    ]

    Path(output_path).parent.mkdir(exist_ok=True, parents=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nSynthesized {len(output)} goldens -> {output_path}")
    print("Evolution breakdown:")
    from collections import Counter

    types = Counter(g["evolution_type"] for g in output)
    for evo_type, count_ in types.most_common():
        print(f"  {evo_type}: {count_}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--docs", default="data/raw/", help="Directory with source documents (PDF/txt)"
    )
    parser.add_argument(
        "--output", default="evals/datasets/golden_rag.json", help="Output JSON path"
    )
    parser.add_argument("--count", type=int, default=120, help="Number of golden pairs to generate")
    args = parser.parse_args()
    synthesize(args.docs, args.output, args.count)
