"""Hybrid retriever: embedding cosine similarity + BM25, fused via RRF."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from typing import Any

import numpy as np

from .bm25 import BM25
from .indexer import expand_synonyms


def _cosine_scores(query_emb: list[float], stored_embs: np.ndarray) -> np.ndarray:
    """Return cosine similarity of query_emb against every row of stored_embs."""
    q = np.array(query_emb, dtype=np.float32)
    q_norm = q / (np.linalg.norm(q) + 1e-10)
    norms = np.linalg.norm(stored_embs, axis=1, keepdims=True) + 1e-10
    return (stored_embs / norms) @ q_norm


def _rrf_fusion(rankings: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion: combine multiple ranked lists → sorted (idx, score)."""
    scores: dict[int, float] = defaultdict(float)
    for rank_list in rankings:
        for rank, idx in enumerate(rank_list):
            scores[idx] += 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


class Retriever:
    """Hybrid retriever over the pre-built corpus index."""

    def __init__(self, index: dict, embedder=None) -> None:
        """
        index:    loaded from rag_index.json
        embedder: Embedder | None — needed for semantic retrieval
        """
        self._examples: list[dict[str, Any]] = index["examples"]
        self._embedder = embedder

        # Build BM25 over indexed texts
        texts = [e["text"] for e in self._examples]
        self._bm25 = BM25(texts)

        # Build embedding matrix if vectors are present
        embs = [e.get("embedding") for e in self._examples]
        if all(v is not None for v in embs):
            self._emb_matrix = np.array(embs, dtype=np.float32)
        else:
            self._emb_matrix = None

    def retrieve(
        self,
        query_text: str,
        k: int = 2,
        embedder=None,  # override instance embedder
        semantic_only: bool = False,
    ) -> list[dict[str, Any]]:
        """Return up to k most relevant examples.

        semantic_only=True: rank by cosine similarity alone (no BM25).
        Default: hybrid RRF fusion of BM25 + semantic when embeddings are
        available, falling back to BM25 alone.
        """
        n = len(self._examples)
        emb = embedder or self._embedder

        if semantic_only:
            if emb is None or self._emb_matrix is None:
                raise ValueError("semantic_only requires an embedder and pre-built embeddings")
            query_emb = emb.embed(query_text)
            sem_scores = _cosine_scores(query_emb, self._emb_matrix)
            return [self._examples[i] for i in np.argsort(-sem_scores)[:k]]

        rankings: list[list[int]] = []

        # --- BM25 ranking (synonym-expanded query) ---
        bm25_scores = self._bm25.scores(expand_synonyms(query_text))
        bm25_order = sorted(range(n), key=lambda i: bm25_scores[i], reverse=True)
        rankings.append(bm25_order)

        # --- Semantic ranking (original query — embedding handles synonyms natively) ---
        if emb is not None and self._emb_matrix is not None:
            query_emb = emb.embed(query_text)
            sem_scores = _cosine_scores(query_emb, self._emb_matrix)
            sem_order = list(np.argsort(-sem_scores))
            rankings.append(sem_order)

        # --- RRF fusion ---
        fused = _rrf_fusion(rankings)
        return [self._examples[idx] for idx, _ in fused[:k]]

    def format_for_prompt(self, examples: list[dict[str, Any]]) -> str:
        """Format retrieved examples as a section to inject into the IR builder prompt."""
        if not examples:
            return ""
        parts = ["## Retrieved Reference Examples\n"]
        parts.append(
            "The following examples are structurally similar to the current problem.\n"
            "Study their IR to understand how similar problems are modeled.\n"
            "Borrow a pattern ONLY when the current problem explicitly requires it.\n"
            "Do NOT copy a pattern just because it appears in a retrieved example — "
            "check each one against the current problem description and csv_schemas first.\n\n"
            "Pattern borrowing rules:\n\n"
            "  temporal lag / ordered set / init constraint\n"
            "    → ONLY if the problem has time periods AND a state variable (inventory,\n"
            "      capacity, workforce) that carries over from one period to the next.\n"
            "    → If there is no time dimension, ignore all lag, ordered, and Periods[0] patterns.\n\n"
            "  domain_filter (variable) / sparse_filter (constraint)\n"
            "    → ONLY if row_count < full Cartesian product of the parameter's index columns.\n"
            "    → Verify with csv_schemas row_count and distinct_values — do not assume from\n"
            "      the problem description alone. If the table is dense, omit both.\n\n"
            "  exclude_diagonal\n"
            "    → ONLY if the variable is indexed over the same set twice AND self-loops are\n"
            "      structurally forbidden (e.g. arc variable in routing).\n"
            "    → Never set for variables over two different sets.\n\n"
            "  big-M linking (binary × continuous)\n"
            "    → ONLY if a binary yes/no decision gates whether a continuous flow is allowed.\n"
            "    → Do not introduce binary variables just because a retrieved example uses them.\n\n"
            "  step cost / binary tier selection\n"
            "    → ONLY if the problem explicitly has tiered or piecewise costs (e.g. quantity\n"
            "      discounts, production tiers). Do not add tiers to a smooth linear cost.\n\n"
            "  scalar variable (domain: [])\n"
            "    → ONLY if the objective needs an auxiliary unnamed scalar (e.g. a minimax\n"
            "      bound). Do not use for any indexed quantity.\n\n"
            "  deviation variables (surplus / shortfall split)\n"
            "    → ONLY if the problem explicitly allows backlogged demand, soft targets,\n"
            "      or over/under penalties (goal programming, backlogging).\n\n"
            "  set_size expression / upper_bound_set\n"
            "    → ONLY for integer position variables whose upper bound is the cardinality\n"
            "      of a set (e.g. MTZ subtour elimination). Never use for production quantities\n"
            "      or inventory levels.\n\n"
            "  duplicate set in domain (e.g. [Locations, Locations])\n"
            "    → ONLY if the parameter or variable is genuinely self-referential — same set\n"
            "      on both axes (e.g. a travel-cost or distance matrix).\n\n"
            "  scenario weighted sum\n"
            "    → ONLY if the problem has explicit probability-weighted scenarios.\n\n"
            "  BOM parameter (component yield matrix)\n"
            "    → ONLY if the problem has a bill-of-materials with component consumption ratios.\n"
        )
        for i, ex in enumerate(examples, 1):
            title = ex.get("problem_title", ex["name"])
            kw = ", ".join(ex.get("ir_patterns", []))
            problem = ex.get("problem", {})
            ir_str = json.dumps(ex["ir"], indent=2)
            parts.append(f"### Reference {i}: {title}")
            if kw:
                parts.append(f"Keywords: {kw}")
            if problem:
                parts.append(f"Problem:\n```json\n{json.dumps(problem, indent=2)}\n```")
            parts.append(f"IR:\n```json\n{ir_str}\n```")
        return "\n\n".join(parts)
