"""LLM-based modeling pattern classifier for RAG retrieval.

Replaces the regex-based detect_modeling_patterns() in the production path so that
natural-language problem descriptions (e.g. "production only when site is active")
are correctly mapped to VOCAB patterns (e.g. "big-M linking") without requiring
the exact trigger phrases that regex rules need.

Falls back to detect_modeling_patterns() (regex) on any error so the RAG channel
always works even if the classifier call fails.
"""

from __future__ import annotations

import json
import logging

from orpilot.rag.patterns import VOCAB_SET

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stable system prompt — describes every pattern once; good for caching.
# ---------------------------------------------------------------------------

_SYSTEM = """\
You classify optimization problems into structural modeling patterns.

Return ONLY a JSON object: {"patterns": ["pattern1", "pattern2", ...]}
Use only names from the list below. Return {"patterns": []} if none apply.
Do not explain or add commentary.

PATTERNS — name: what signals it (include ALL that clearly apply):

Variable types
  binary variable: Decision is yes/no, 0/1; select, open, assign, activate, whether to
  integer variable: Decision is a whole number — count of vehicles, batches, workers, trips
  scalar variable: Single auxiliary bound variable used to cap an objective (minimax bound)

Variable structure
  domain filter: Only certain (source, destination) pairs are valid; not all combinations exist
  exclude diagonal: Arcs or variables exclude i=i self-loops; travel between DIFFERENT locations
  duplicate set: A parameter indexed by the same set twice, e.g. distance[Location, Location]
  upper bound set: Variable upper-bounded by set cardinality, e.g. MTZ position ≤ |Customers|

Temporal / sequence
  temporal lag: Multi-period model with ordered time set; state (inventory, stock) carries over from previous period
  init constraint: Separate constraint for first period / initial state (initial inventory = X)

Constraint patterns
  big-M linking: Activity only allowed when binary indicator is 1; gates flow, production, shipment
  mutual exclusion: At most one option from a group; cannot both be selected; either/or
  deviation variables: Backlog, shortage, surplus; penalty for unmet demand; soft constraints
  step cost: Tiered or piecewise pricing; volume discounts; cost changes at thresholds in data
  balance constraint: Flow conservation; stock balance; inflow equals outflow at each node
  set covering: At least one facility/option must cover each requirement
  MTZ subtour elimination: Routing with position/sequence variables to eliminate subtours
  sparse filter: Arc/cost table has far fewer rows than |SetA|×|SetB|; only some pairs are defined
  filter_column subset: CSV has a type/category column that defines a subset of a larger set
  implication constraint: If X selected then Y must also be selected; forces co-selection
  flow conservation: Multi-commodity network flow; in-flow equals out-flow at transshipment nodes
  precedence constraint: Task A must finish before B can start; prerequisite ordering
  ratio constraint: At least X% of a group; proportion or fraction must meet a threshold

Objective
  minimax objective: Minimize the maximum load/cost/penalty; equalize workload; fairest assignment

Data / parameter patterns
  BOM parameter: Bill of materials; recipe ratios; yield rates; ingredient consumption matrix
  scenario weighted sum: Expected cost/profit over scenarios; probability-weighted sum
  two-stage stochastic: Here-and-now decisions before uncertainty resolves; recourse variables
  set size expression: Big-M equal to set cardinality; upper bound equal to number of customers
  pre-enumerated options: Cutting patterns, option bundles, configurations pre-listed as rows

Structural
  subset indexed variable: Variable indexed over a subset (e.g. Proteins ⊂ FoodItems)
  no sets: All-scalar model; no CSV files; no indexed sets — all values are single numbers
"""


def _summarise_csv_schemas(csv_schemas: dict) -> str:
    if not csv_schemas:
        return "(no CSV data files)"
    lines = []
    for stem, info in csv_schemas.items():
        cols = list(info.get("columns", {}).keys())
        row_count = info.get("row_count", "?")
        lines.append(f"  {stem}.csv: {row_count} rows, columns: {cols}")
    return "\n".join(lines)


def _build_user_message(problem_dict: dict, csv_schemas: dict | None) -> str:
    parts = ["PROBLEM:"]
    for key in ("title", "description", "objective", "objective_description"):
        val = problem_dict.get(key) or ""
        if val:
            parts.append(f"{key.replace('_', ' ').title()}: {val}")
    for key in ("decision_variables", "parameters"):
        val = problem_dict.get(key) or []
        if val:
            parts.append(f"{key.replace('_', ' ').title()}: " + "; ".join(str(v) for v in val))
    constraints = problem_dict.get("constraints") or []
    if constraints:
        descs = [
            c.get("description", str(c)) if isinstance(c, dict) else str(c)
            for c in constraints
        ]
        parts.append("Constraints: " + "; ".join(descs))

    parts.append("")
    parts.append("CSV DATA SCHEMAS:")
    parts.append(_summarise_csv_schemas(csv_schemas or {}))
    parts.append("")
    parts.append("Classify.")
    return "\n".join(parts)


def classify_patterns(
    problem_dict: dict,
    llm,
    csv_schemas: dict | None = None,
) -> set[str]:
    """Return pat: tokens for this problem using an LLM classifier.

    Falls back to regex detect_modeling_patterns() on any error.
    llm must implement BaseLLM.chat(messages) -> str.
    """
    try:
        messages = [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": _build_user_message(problem_dict, csv_schemas)},
        ]
        raw = llm.chat(messages)
        raw = llm._strip_json_fences(raw)
        data = json.loads(raw)
        patterns = data.get("patterns", [])
        valid = {p for p in patterns if p in VOCAB_SET}
        invalid = set(patterns) - valid
        if invalid:
            logger.debug("classify_patterns: dropping non-VOCAB patterns: %s", invalid)
        return {f"pat:{p}" for p in valid}
    except Exception as exc:
        logger.warning("classify_patterns failed (%s), falling back to regex", exc)
        from orpilot.rag.structural import detect_modeling_patterns
        return detect_modeling_patterns(problem_dict)
