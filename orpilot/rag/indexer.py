"""Build and persist the corpus RAG index."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

from orpilot.paths import PROJECT_ROOT

CORPUS_DIR: Path = PROJECT_ROOT / "corpus" / "examples"
INDEX_PATH: Path = PROJECT_ROOT / "corpus" / "rag_index.json"

# ---------------------------------------------------------------------------
# Synonym expansion for BM25
# Each group is a set of interchangeable supply-chain terms. At index and
# query time, if any member is found in the text, all other members are
# appended so BM25 matches regardless of which term the user chooses.
# Keep groups conservative — false positives hurt more than missed synonyms.
# ---------------------------------------------------------------------------
_SYNONYM_GROUPS: list[frozenset[str]] = [
    # Facility types (singular + plural)
    frozenset({
        "warehouse", "warehouses",
        "distribution center", "distribution centers",
        "distribution centre", "distribution centres",
        "fulfillment center", "fulfillment centers",
        "fulfilment center", "fulfilment centres",
    }),
    frozenset({
        "plant", "plants",
        "factory", "factories",
        "production site", "production sites",
        "manufacturing site", "manufacturing sites",
        "manufacturing facility", "manufacturing facilities",
        "production facility", "production facilities",
    }),
    # Entities
    frozenset({"supplier", "suppliers", "vendor", "vendors"}),
    # Costs
    frozenset({"holding cost", "storage cost", "carrying cost"}),
    frozenset({"transportation cost", "shipping cost", "freight cost", "delivery cost"}),
    frozenset({"setup cost", "changeover cost"}),
    # Facility status actions
    frozenset({"open", "opening", "activate", "activation"}),
    frozenset({"close", "closing", "deactivate", "deactivation", "shut down"}),
    # Inventory states
    frozenset({"backlog", "backorder"}),
    frozenset({"safety stock", "buffer stock", "safety inventory"}),
]

# Pre-compile patterns (longest terms first within each group to avoid
# partial matches when a short term is a substring of a longer one)
_COMPILED_GROUPS: list[list[re.Pattern[str]]] = [
    [
        re.compile(r"\b" + re.escape(term) + r"\b", re.IGNORECASE)
        for term in sorted(group, key=len, reverse=True)
    ]
    for group in _SYNONYM_GROUPS
]


def expand_synonyms(text: str) -> str:
    """Append synonym expansions to text for improved BM25 recall.

    For each synonym group, if any member appears in the text, all other
    members that are absent are appended. The original text is preserved
    so token frequencies are unchanged; expansions only add new tokens.
    """
    additions: list[str] = []
    for group, patterns in zip(_SYNONYM_GROUPS, _COMPILED_GROUPS):
        matched = {
            term for term, pat in zip(
                sorted(group, key=len, reverse=True), patterns
            )
            if pat.search(text)
        }
        if matched:
            additions.extend(group - matched)
    if additions:
        return text + " " + " ".join(additions)
    return text


def _example_text(d: dict) -> str:
    """Build the text to embed/index for a corpus example."""
    p = d.get("problem", {})
    kw = d.get("ir_patterns", [])
    lines = [
        f"Title: {p.get('title', '')}",
        f"Keywords: {', '.join(kw)}",
        f"Description: {p.get('description', '')}",
        f"Objective: {p.get('objective', '')}",
        f"Objective description: {p.get('objective_description', '')}",
        f"Decision variables: {'; '.join(p.get('decision_variables', []))}",
        f"Parameters: {'; '.join(p.get('parameters', []))}",
    ]
    for c in p.get("constraints", []):
        if isinstance(c, dict) and c.get("description"):
            lines.append(f"Constraint: {c['description']}")
    return "\n".join(lines)


def build_index(
    corpus_dir: Path = CORPUS_DIR,
    embedder=None,  # Embedder | None
    force: bool = False,
) -> dict:
    """Build the index or load it from disk.

    If ``embedder`` is None, only BM25 is available (no embedding vectors).
    Pass ``force=True`` to rebuild even when the index file already exists.
    """
    if INDEX_PATH.exists() and not force:
        return json.loads(INDEX_PATH.read_text(encoding="utf-8"))

    examples_meta = []
    for json_file in sorted(corpus_dir.glob("*.json")):
        d = json.loads(json_file.read_text(encoding="utf-8"))
        if "ir" not in d:
            continue
        text = _example_text(d)
        examples_meta.append({
            "name": json_file.stem,
            "text": text,
            "problem_title": d.get("problem", {}).get("title", ""),
            "ir_patterns": d.get("ir_patterns", []),
            "problem": d.get("problem", {}),
            "csv_schemas": d.get("csv_schemas", {}),
            "ir": d["ir"],
        })

    # Compute embeddings if an embedder is provided
    embedding_model = None
    if embedder is not None:
        texts = [e["text"] for e in examples_meta]
        embeddings = embedder.embed_batch(texts)
        embedding_model = embedder.model
        for meta, emb in zip(examples_meta, embeddings):
            meta["embedding"] = emb
    else:
        for meta in examples_meta:
            meta["embedding"] = None

    index = {
        "embedding_model": embedding_model,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "examples": examples_meta,
    }
    INDEX_PATH.write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")
    print(f"[RAG] Index built: {len(examples_meta)} examples → {INDEX_PATH}")
    return index


def load_index() -> dict | None:
    """Load index from disk; return None if it doesn't exist."""
    if not INDEX_PATH.exists():
        return None
    return json.loads(INDEX_PATH.read_text(encoding="utf-8"))
