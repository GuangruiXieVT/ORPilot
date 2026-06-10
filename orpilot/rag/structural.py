"""IR schema fingerprinting for structural similarity retrieval.

Each model is represented as a set of string tokens encoding its structural
shape (variable types, dimensions, boolean feature flags, and canonical
modeling pattern tags). Jaccard similarity on these sets finds structurally
similar models regardless of domain vocabulary.
"""

from __future__ import annotations

from .patterns import detect_patterns


def _bucket(n: int, thresholds: list[int]) -> str:
    labels = ["small", "medium", "large"]
    for i, t in enumerate(thresholds):
        if n <= t:
            return labels[i]
    return labels[-1]


def _has_lag(expr: dict | None) -> bool:
    """Recursively check whether any node in an IR expression tree has a lag field."""
    if not isinstance(expr, dict):
        return False
    if expr.get("lag") is not None:
        return True
    for v in expr.values():
        if isinstance(v, dict) and _has_lag(v):
            return True
        if isinstance(v, list):
            for item in v:
                if isinstance(item, dict) and _has_lag(item):
                    return True
    return False


def ir_fingerprint(example: dict) -> set[str]:
    """Extract structural tokens from a corpus example dict.

    Reads 'modeling_patterns' and 'ir' keys.
    Returns a set of string tokens covering both structural IR properties
    and canonical modeling pattern tags.
    """
    tokens: set[str] = set()
    ir = example.get("ir", {})
    patterns = example.get("modeling_patterns", [])

    # Objective sense
    sense = ir.get("sense", "")
    if sense:
        tokens.add(f"sense:{sense}")

    # Model type
    model_type = ir.get("model_type", "").lower()
    if "mixed integer" in model_type:
        tokens.add("model:mip")
    elif "integer" in model_type:
        tokens.add("model:ip")
    else:
        tokens.add("model:lp")

    # Sets
    sets = ir.get("sets", {})
    tokens.add(f"n_sets:{_bucket(len(sets), [2, 5])}")
    for s in sets.values():
        if s.get("ordered"):
            tokens.add("has:ordered_set")
            break

    # Variables
    variables = ir.get("variables", {})
    tokens.add(f"n_vars:{_bucket(len(variables), [2, 5])}")
    for var in variables.values():
        vtype = var.get("type", "")
        if vtype:
            tokens.add(f"var_type:{vtype}")
        tokens.add(f"var_dim:{len(var.get('domain', []))}")
        if var.get("domain_filter"):
            tokens.add("has:domain_filter")
        if var.get("exclude_diagonal"):
            tokens.add("has:exclude_diagonal")

    # Constraints
    constraints = ir.get("constraints", {})
    tokens.add(f"n_cons:{_bucket(len(constraints), [3, 8])}")
    for con in constraints.values():
        tokens.add(f"con_dim:{len(con.get('domain', []))}")
        if con.get("sparse_filter"):
            tokens.add("has:sparse_filter")
        if _has_lag(con.get("expression")):
            tokens.add("has:lag")

    # Canonical modeling pattern tags
    for pat in patterns:
        tokens.add(f"pat:{pat}")

    return tokens


def detect_modeling_patterns(problem: dict) -> set[str]:
    """Infer canonical modeling pattern tags from a problem description.

    Returns only pat: tokens (canonical modeling patterns from patterns.VOCAB).
    Works with plain dicts or objects accessible via getattr.
    """
    def _get(key: str, default=""):
        if isinstance(problem, dict):
            return problem.get(key, default)
        return getattr(problem, key, default) or default

    parts = [
        _get("title"),
        _get("description"),
        str(_get("objective")),
        str(_get("objective_description")),
        " ".join(_get("decision_variables") or []),
        " ".join(_get("parameters") or []),
    ]
    for c in (_get("constraints") or []):
        parts.append(c.get("description", str(c)) if isinstance(c, dict) else str(c))
    text = " ".join(parts)

    return {f"pat:{p}" for p in detect_patterns(text)}


def jaccard(a: set[str], b: set[str]) -> float:
    """Jaccard similarity coefficient. Returns 0.0 when both sets are empty."""
    if not a and not b:
        return 0.0
    union = len(a | b)
    return len(a & b) / union if union else 0.0
