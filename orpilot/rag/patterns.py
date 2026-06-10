"""Canonical modeling pattern vocabulary for OR corpus examples.

VOCAB is the single source of truth — 32 structural tags that both corpus
examples (in the 'modeling_patterns' field) and the query-side fingerprint use.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Canonical vocabulary (32 tags)
# ---------------------------------------------------------------------------

VOCAB: list[str] = [
    # Variable type
    "binary variable",
    "integer variable",
    "scalar variable",
    # Variable structure
    "domain filter",
    "exclude diagonal",
    "duplicate set",
    "upper bound set",
    # Temporal / sequence
    "temporal lag",
    "init constraint",
    # Constraint patterns
    "big-M linking",
    "mutual exclusion",
    "deviation variables",
    "step cost",
    "balance constraint",
    "set covering",
    "MTZ subtour elimination",
    "sparse filter",
    "filter_column subset",
    "implication constraint",
    "flow conservation",
    "precedence constraint",
    "ratio constraint",
    # Objective
    "minimax objective",
    # Data / parameter patterns
    "BOM parameter",
    "scenario weighted sum",
    "two-stage stochastic",
    "set size expression",
    "pre-enumerated options",
    # Structural
    "subset indexed variable",
    "no sets",
]

VOCAB_SET: frozenset[str] = frozenset(VOCAB)

# ---------------------------------------------------------------------------
# Migration map: old free-form tag -> list of canonical tags ([] = drop)
# ---------------------------------------------------------------------------

OLD_TO_NEW: dict[str, list[str]] = {
    # --- Variable type ---
    "binary variable":              ["binary variable"],
    "binary selection":             ["binary variable"],
    "binary assignment":            ["binary variable"],
    "binary selection variable":    ["binary variable"],
    "binary activation variable":   ["binary variable"],
    "binary personnel selection":   ["binary variable"],
    "binary option selection":      ["binary variable"],
    "binary used variable":         ["binary variable"],
    "binary food selection":        ["binary variable"],
    "binary first-stage":           ["binary variable", "two-stage stochastic"],
    "server activation":            ["binary variable", "big-M linking"],
    "facility activation":          ["binary variable", "big-M linking"],
    "integer variable":             ["integer variable"],
    "integer trucks":               ["integer variable"],
    "scalar integer variables":     ["integer variable"],
    "continuous and integer variables": ["integer variable"],
    "fleet sizing":                 ["integer variable"],
    "IP":                           ["integer variable"],
    "integer programming":          [],
    "scalar variable":              ["scalar variable"],
    "scalar auxiliary variable":    ["scalar variable"],
    # --- Variable structure ---
    "domain filter":                ["domain filter"],
    "domain_filter":                ["domain filter"],
    "domain filter (feasible start window)": ["domain filter"],
    "domain filter free (dense matchup)":    [],
    "sparse network":               ["sparse filter", "domain filter"],
    "exclude diagonal":             ["exclude diagonal"],
    "exclude_diagonal":             ["exclude diagonal"],
    "duplicate set":                ["duplicate set"],
    "duplicate set domain":         ["duplicate set"],
    "duplicate set constraint":     ["duplicate set"],
    "duplicate set domain diagonal guard":   ["duplicate set", "exclude diagonal"],
    "3D constraint domain with duplicate set": ["duplicate set"],
    "adjacency matrix constraint":  ["duplicate set"],
    "symmetric distance directed arcs": ["duplicate set", "exclude diagonal"],
    "upper bound set":              ["upper bound set"],
    # --- Temporal / sequence ---
    "ordered set":                  ["temporal lag"],
    "single ordered set":           ["temporal lag"],
    "ordered set forward cumulative": ["temporal lag"],
    "temporal lag":                 ["temporal lag"],
    "temporal lag (inequality)":    ["temporal lag"],
    "cumulative production variable with lag": ["temporal lag"],
    "sell constraint with lag on inventory":   ["temporal lag"],
    "skill accumulation":           ["temporal lag"],
    "capacity accumulation":        ["temporal lag"],
    "multi-period investment LP":   ["temporal lag"],
    "three separate lag balance constraints":  ["temporal lag", "balance constraint"],
    "temporal investment balance":  ["balance constraint", "temporal lag"],
    "multi-variable temporal balance": ["balance constraint", "temporal lag"],
    "init constraint":              ["init constraint"],
    "balance_init constraint with Periods[0]": ["init constraint"],
    "inventory balance with init constraint":  ["balance constraint", "init constraint"],
    # --- Big-M / linking ---
    "big-M linking":                ["big-M linking"],
    "big-M linking constraint":     ["big-M linking"],
    "binary big-M linking":         ["big-M linking"],
    "big-M capacity constraint":    ["big-M linking"],
    "big-M capacity linking":       ["big-M linking"],
    "capacity big-M linking":       ["big-M linking"],
    "big-M indicator linking":      ["big-M linking"],
    "big-M binary activation":      ["big-M linking"],
    "binary indicator with big-M":  ["big-M linking"],
    "fixed setup cost plus variable cost": ["big-M linking"],
    "fixed setup cost":             ["big-M linking"],
    "fixed cost binary activation": ["big-M linking"],
    "truck capacity linking constraint":   ["big-M linking"],
    "two-direction linking constraints upper and lower": ["big-M linking"],
    "minimum portion if selected":  ["big-M linking"],
    "binary setup cost activation": ["big-M linking"],
    "binary big-M linking":         ["big-M linking"],
    "big-M tier fill constraints":  ["step cost"],
    # --- Mutual exclusion ---
    "mutual exclusion constraint":  ["mutual exclusion"],
    "mutual exclusion with hardcoded indices": ["mutual exclusion"],
    "mutual exclusion of two candidates":      ["mutual exclusion"],
    "mutual exclusion via indicator":          ["mutual exclusion"],
    "XOR binary exclusion":         ["mutual exclusion"],
    "exactly-one-from-group constraint": ["mutual exclusion"],
    "exactly-one-per-group constraint":  ["mutual exclusion"],
    "one-of-each selection constraint":  ["mutual exclusion"],
    "set partitioning equality":    ["mutual exclusion"],
    # --- Deviation variables ---
    "deviation variables":          ["deviation variables"],
    "split variables":              ["deviation variables"],
    "split-variable backlogging":   ["deviation variables"],
    "overtime deviation variable":  ["deviation variables"],
    "backlogging":                  ["deviation variables"],
    "backlog state variable":       ["deviation variables"],
    "overtime variable":            ["deviation variables"],
    "fairness penalty":             ["deviation variables"],
    "inventory balance with backlog": ["balance constraint", "deviation variables"],
    # --- Step cost ---
    "step cost":                    ["step cost"],
    "binary tier-ordering variables": ["step cost"],
    "piecewise tiered profit via pre-enumerated options": ["step cost", "pre-enumerated options"],
    "two-level storage cost":       ["step cost"],
    "maximize profit with tiered purchasing cost": ["step cost"],
    # --- Balance constraint ---
    "balance constraint":           ["balance constraint"],
    "supply-demand balance":        ["balance constraint"],
    "buy/sell/hold balance":        ["balance constraint"],
    "balance constraint for by-product": ["balance constraint"],
    "yield flow balance":           ["balance constraint"],
    "workforce balance hire fire":  ["balance constraint"],
    "cash flow tracking":           ["balance constraint"],
    "two-stage machine balance":    ["balance constraint"],
    "balance constraints domain=[]": ["balance constraint", "no sets"],
    # --- Set covering ---
    "set covering":                 ["set covering"],
    # --- MTZ ---
    "MTZ subtour elimination":      ["MTZ subtour elimination"],
    "TSP":                          ["MTZ subtour elimination"],
    # --- Sparse filter ---
    "sparse filter":                ["sparse filter"],
    "sparse_filter":                ["sparse filter"],
    "sparse_filter constraint":     ["sparse filter"],
    "sparse parameter":             ["sparse filter"],
    "sparse binary parameter":      ["sparse filter"],
    "sparse shipping cost":         ["sparse filter"],
    "sparse parameter in constraint": ["sparse filter"],
    "sparse assignment":            ["sparse filter"],
    "sparse coverage parameter":    ["sparse filter"],
    "sparse conflict parameter":    ["sparse filter"],
    "sparse upper bound constraint": ["sparse filter"],
    "sparse crop-season":           ["sparse filter"],
    "sparse processing times":      ["sparse filter"],
    "resource constraint with sparse usage": ["sparse filter"],
    "sparse preference parameter":  ["sparse filter"],
    # --- Filter_column subset ---
    "filter_column subset":         ["filter_column subset"],
    "typed machine sets":           ["filter_column subset"],
    "multi-category membership":    ["filter_column subset"],
    "typed nodes depot customer":   ["filter_column subset"],
    # --- Implication ---
    "implication constraint":       ["implication constraint"],
    "implication constraint (if A then B)": ["implication constraint"],
    "logical implication as inequality":    ["implication constraint"],
    "A-implies-C implication via auxiliary binary": ["implication constraint"],
    # --- Flow conservation ---
    "flow conservation constraint": ["flow conservation"],
    "two-echelon transshipment":    ["flow conservation"],
    "receive link constraint":      ["flow conservation"],
    # --- Precedence ---
    "precedence constraints":       ["precedence constraint"],
    "prerequisite constraint":      ["precedence constraint"],
    "sequential stage constraints": ["precedence constraint"],
    # --- Ratio ---
    "ratio constraint":             ["ratio constraint"],
    "blending ratio constraints":   ["ratio constraint"],
    "fraction linking":             ["ratio constraint"],
    "workforce policy (skilled ratio)": ["ratio constraint"],
    "grade blending constraint":    ["ratio constraint"],
    "skill aggregation requirement": ["ratio constraint"],
    "experience aggregation requirement": ["ratio constraint"],
    # --- Minimax ---
    "minimax objective":            ["minimax objective"],
    "minimize maximum":             ["minimax objective"],
    "minimax absolute difference":  ["minimax objective"],
    "minimax project duration":     ["minimax objective"],
    # --- BOM parameter ---
    "BOM parameter":                ["BOM parameter"],
    "BOM constraint":               ["BOM parameter"],
    # --- Scenario weighted sum ---
    "scenario weighted sum":        ["scenario weighted sum"],
    "scenario indexed variable":    ["scenario weighted sum"],
    "expected cost in objective":   ["scenario weighted sum"],
    # --- Two-stage stochastic ---
    "two-stage stochastic":         ["two-stage stochastic"],
    "stochastic two-stage":         ["two-stage stochastic"],
    "first-stage vs second-stage variables": ["two-stage stochastic"],
    "continuous first-stage":       ["two-stage stochastic"],
    # --- Set size expression ---
    "set size expression":          ["set size expression"],
    # --- Pre-enumerated options ---
    "pre-enumerated cutting patterns": ["pre-enumerated options"],
    "2D yield parameter over patterns and orders": ["pre-enumerated options"],
    # --- Subset indexed variable ---
    "subset-based grouping constraints": ["subset indexed variable"],
    # --- No sets ---
    "no sets or parameters":        ["no sets"],
    "no sets":                      ["no sets"],
    "no sets.csv needed":           ["no sets"],
    "no CSV data":                  ["no sets"],
    "all-scalar variables (domain=[])": ["no sets"],
    "scalar variables only":        ["no sets"],
    "all scalar variables":         ["no sets"],
    "scalar MIP":                   ["no sets"],
    "scalar LP":                    ["no sets"],
    "scalar variables":             ["no sets"],
    "capacity constraints domain=[]": ["no sets"],
    "balance constraints domain=[]": ["balance constraint", "no sets"],
    # --- Drop: model-type labels ---
    "LP":                           [],
    "mixed integer programming":    [],
    "linear programming":           [],
    "integer programming":          [],
    # --- Drop: domain-specific objective labels ---
    "maximize profit":              [],
    "maximize profit LP":           [],
    "maximize profit MIP":          [],
    "minimize cost":                [],
    "maximize revenue":             [],
    "maximize preference":          [],
    "preference maximization":      [],
    "maximize priority":            [],
    "maximize GDP":                 [],
    "maximize ending inventory":    [],
    "maximize terminal wealth":     [],
    "maximize final wealth":        [],
    "maximize NPV":                 [],
    "maximize rental income":       [],
    "maximize nutrient intake":     [],
    "minimize salary MIP":          [],
    "minimize total headcount":     [],
    "minimize total cost MIP":      [],
    "minimize waste area":          [],
    "maximize":                     [],
    # --- Drop: domain-specific problem labels ---
    "transportation problem":       [],
    "farm planning":                [],
    "shift scheduling":             [],
    "bin packing":                  [],
    "resource bin packing":         [],
    "partition problem":            [],
    "p-median":                     [],
    "heterogeneous fleet":          [],
    "crude oil blending MIP":       [],
    "cutting stock LP":             [],
    "production planning LP":       [],
    "sulfur blending LP":           [],
    "method selection MIP":         [],
    "product selection MIP":        [],
    # --- Drop: too generic constraints ---
    "scalar parameter":             [],
    "capacity constraint":          [],
    "resource constraint":          [],
    "resource constraints":         [],
    "budget constraint":            [],
    "budget constraint over two sets": [],
    "machine capacity constraint":  [],
    "production capacity constraint": [],
    "demand capacity constraint":   [],
    "demand upper bound constraint": [],
    "demand fulfillment constraint": [],
    "upper bound on variable":      [],
    "labor constraint":             [],
    "area constraint":              [],
    "weight constraints":           [],
    "supply constraint":            [],
    "demand constraint":            [],
    "equality constraint":          [],
    "linking constraint":           [],
    "product set":                  [],
    "project selection":            [],
    "land allocation":              [],
    # --- Drop: expression / implementation details ---
    "multiply two parameters":      [],
    "multiply parameters":          [],
    "multiply two scalar parameters": [],
    "multiply of two parameters in objective": [],
    "multiply parameter in sum":    [],
    "multiply(subtract(param, scalar_param), var) in objective": [],
    "multiply two parameters in objective": [],
    "gain multiplier":              [],
    "value-added coefficients":     [],
    "reversed indices in sum":      [],
    "total length constant":        [],
    "three-term objective with nested sum": [],
    "cross-product upper bound":    [],
    "indexed_sum":                  [],
    "aliased indexed_sum":          [],
    "2D variable indexed by two sets": [],
    "2D cost parameter":            [],
    "2D cost parameters":           [],
    "2D resource use parameter":    [],
    "3D parameter":                 [],
    "precomputed cross-parameter":  [],
    "pre-computed derived parameter in CSV": [],
    # --- Drop: hardcoded-member implementation details ---
    "hardcoded constants in constraints": [],
    "hardcoded constants":          [],
    "hardcoded member reference":   [],
    "hardcoded member ID in constraint": [],
    "hardcoded member ID in scalar constraint": [],
    "hardcoded item ID in constraint": [],
    "hardcoded stage member references": [],
    "scalar constraint with hardcoded member IDs": [],
    "scalar constraint with hardcoded member ID": [],
    "scalar domain constraint with hardcoded member IDs": [],
    "scalar domain=[] constraint with hardcoded indices": [],
    "scalar constraint with hardcoded member": [],
    "scalar resource constraint with indexed_sum": [],
    "scalar constraint with indexed_sum": [],
    "mixed indexed and scalar terms in constraint": [],
    "LP with set-indexed and scalar variables": [],
    "resource constraints with scalar constants": [],
    "per-item domain constraint":   [],
    "single set indexed objective": [],
    "single set with two parameters": [],
    "single set":                   [],
    "SetName[-1] last element reference": [],
    # --- Drop: misc domain / narrative ---
    "nesting substitution demand":  [],
    "cumulative demand parameter":  [],
    "by-product generation":        [],
    "sell-or-dispose decision":     [],
    "storage cost":                 [],
    "joint ordering cost":          [],
    "dogs-together linking constraint": [],
    "ingredient supply":            [],
    "ingredients supply":           [],
    "sold outflow variable":        [],
    "outsourcing variable":         [],
    "profit maximization revenue from sold variable": [],
    "machine rental cost":          [],
    "project scheduling LP":        [],
    "terminal constraint last period": [],
    "end-of-horizon inventory requirement": [],
    "end-of-horizon stock requirement": [],
    "warehouse capacity constraint": [],
    "monthly capacity constraint":  [],
    "own + rented warehouse with volume constraint": [],
    "multi-product production planning": [],
    "compensation cost":            [],
    "minimum batch constraint":     [],
    "minimum selection count":      [],
    "minimum and maximum count":    [],
    "two item types with separate sets": [],
    "pollution cap constraint":     [],
    "lower bound on production":    [],
    "type_match 2D parameter":      [],
    "maximize terminal wealth":     [],
    "maximize final wealth":        [],
    "maximize":                     [],
    "domain filter free (dense matchup)": [],
}


# ---------------------------------------------------------------------------
# Query-side detection: keyword rules per canonical tag
# ---------------------------------------------------------------------------

_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("binary variable", re.compile(
        r"\b(binary|0-1|yes.?or.?no|whether to|select\w*|assign\w*|open or close|activate|deactivate)\b",
        re.IGNORECASE,
    )),
    ("integer variable", re.compile(
        r"\b(integer|whole number|number of (vehicles|trucks|batches|workers|units|trips))\b",
        re.IGNORECASE,
    )),
    ("scalar variable", re.compile(
        r"\b(auxiliary (scalar|variable|bound)|minimax bound|peak.load (variable)?|upper bound variable|single scalar (variable|bound))\b",
        re.IGNORECASE,
    )),
    ("domain filter", re.compile(
        r"\b(eligible|feasible (arc|pair|route|combination)|only certain|not all combinations|allowed pair|valid arc)\b",
        re.IGNORECASE,
    )),
    ("exclude diagonal", re.compile(
        r"\b(cannot (travel|go|ship) to itself|i\s*[≠!=]\s*j|between different (location|node|city)|self.loop|arc from \w+ to different)\b",
        re.IGNORECASE,
    )),
    ("duplicate set", re.compile(
        r"\b(distance (between|matrix)|travel cost|cost from \w+ to \w+|pairwise cost|between any two|arc cost)\b",
        re.IGNORECASE,
    )),
    ("upper bound set", re.compile(
        r"\b(MTZ position|subtour.elimination position|position variable|upper bound.{0,30}number of customers)\b",
        re.IGNORECASE,
    )),
    ("temporal lag", re.compile(
        r"\b(carries? over|previous period|end.of.period|inventory balance|carryover|from (one|each) period to (the next|another)|stock balance|rolling inventory)\b",
        re.IGNORECASE,
    )),
    ("init constraint", re.compile(
        r"\b(initial (inventory|stock|level|workforce|capacity)|starting (inventory|stock|level)|first.period constraint|at time (zero|0))\b",
        re.IGNORECASE,
    )),
    ("big-M linking", re.compile(
        r"\b(only if (open|selected|active|chosen)|big.?M|activation (cost|constraint)|if (selected|open|active) then|gates? (flow|production|shipment))\b",
        re.IGNORECASE,
    )),
    ("mutual exclusion", re.compile(
        r"\b(at most one|mutually exclusive|either.or|cannot both|exactly one (from|per|of)|one of each)\b",
        re.IGNORECASE,
    )),
    ("deviation variables", re.compile(
        r"\b(backlog|shortage|surplus|deviation (variable|from target)|soft constraint|penalty for (over|under)|unmet demand|backorder)\b",
        re.IGNORECASE,
    )),
    ("step cost", re.compile(
        r"\b(tiered (cost|pricing)|piecewise (cost|linear)|volume discount|quantity discount|tier|bracket (cost|pricing))\b",
        re.IGNORECASE,
    )),
    ("balance constraint", re.compile(
        r"\b(flow balance|conservation of flow|inflow.{0,10}outflow|stock balance|inventory conservation|node balance|material balance)\b",
        re.IGNORECASE,
    )),
    ("set covering", re.compile(
        r"\b(set (covering|cover)|every (item|requirement|demand) must be covered|at least one (option|facility) covers)\b",
        re.IGNORECASE,
    )),
    ("MTZ subtour elimination", re.compile(
        r"\b(subtour|MTZ|Miller.Tucker.Zemlin|subtour.elimination|visit sequence|position variable.{0,30}(routing|TSP|VRP))\b",
        re.IGNORECASE,
    )),
    ("sparse filter", re.compile(
        r"\b(sparse (parameter|table|network|matrix)|only (some|certain) pairs (have|are) (defined|valid)|not all (routes|arcs|pairs) (exist|are defined))\b",
        re.IGNORECASE,
    )),
    ("filter_column subset", re.compile(
        r"\b(subset of (items|products|foods|nodes)|filtered (from|subset)|loaded as a subset|category (of|within)|subgroup of)\b",
        re.IGNORECASE,
    )),
    ("implication constraint", re.compile(
        r"\b(implies?|requires? (that|selection of)|if \w+ (is )?(selected|chosen|active|open) then \w+|forces? (selection|activation)|must also (be )?selected)\b",
        re.IGNORECASE,
    )),
    ("flow conservation", re.compile(
        r"\b(flow conservation|node.arc balance|transshipment|in.flow equals out.flow|multi.commodity flow|network flow balance|conservation constraint)\b",
        re.IGNORECASE,
    )),
    ("precedence constraint", re.compile(
        r"\b(must (precede|finish before|come before)|prerequisite|dependency (between|among)|start (after|only after)|sequence constraint|task ordering)\b",
        re.IGNORECASE,
    )),
    ("ratio constraint", re.compile(
        r"\b(at least \d+\s*%|ratio (constraint|of|requirement)|proportion (of total|of all|must)|fraction of (total|all)|minimum (percentage|fraction|share)|skill ratio)\b",
        re.IGNORECASE,
    )),
    ("minimax objective", re.compile(
        r"\b(minimax|minimize (the )?maximum|minimize peak|most (loaded|burdened|heavily)|equali[sz]e (workload|load)|fairest|peak.load)\b",
        re.IGNORECASE,
    )),
    ("BOM parameter", re.compile(
        r"\b(bill of materials|BOM|component consumption|recipe (ratio|matrix)|yield (rate|matrix|parameter)|ingredient (ratio|quantity))\b",
        re.IGNORECASE,
    )),
    ("scenario weighted sum", re.compile(
        r"\b(scenario|probability.weighted|expected (cost|value|profit)|weighted sum of scenarios|stochastic objective)\b",
        re.IGNORECASE,
    )),
    ("two-stage stochastic", re.compile(
        r"\b(two.stage (stochastic|program)|here.and.now|recourse (decision|variable|cost)|first.stage decision|second.stage)\b",
        re.IGNORECASE,
    )),
    ("set size expression", re.compile(
        r"\b(big.?M.{0,30}(number of|cardinality)|cardinality of (the )?(set|customers?|nodes?)|upper bound.{0,20}set size|len\(.*\) as big.?M)\b",
        re.IGNORECASE,
    )),
    ("pre-enumerated options", re.compile(
        r"\b(pre.?enumerated (option|pattern|configuration)|cutting pattern|option bundle|enumerated (combination|configuration)|pattern (set|index|variable))\b",
        re.IGNORECASE,
    )),
    ("subset indexed variable", re.compile(
        r"\b(subset (of (items|products|foods|nodes|variables))|subgroup.indexed|proteins?.{0,30}(subset of|foods?)|variable over (a )?subset)\b",
        re.IGNORECASE,
    )),
    ("no sets", re.compile(
        r"\b(no (sets?|CSV|data files?)|all.?scalar (model|variables?|program)|single (scalar )?values only|no time periods?.{0,10}no sets?)\b",
        re.IGNORECASE,
    )),
]


def detect_patterns(text: str) -> set[str]:
    """Infer canonical modeling pattern tags from natural-language text.

    Returns a subset of VOCAB — only tags whose keyword rules fire on the text.
    """
    detected: set[str] = set()
    for tag, pattern in _RULES:
        if pattern.search(text):
            detected.add(tag)
    return detected


def canonicalize(tags: list[str]) -> list[str]:
    """Map a list of old free-form tags to canonical VOCAB tags.

    Tags already in VOCAB pass through. Unknown tags are silently dropped.
    Returns a sorted deduplicated list.
    """
    result: set[str] = set()
    for tag in tags:
        if tag in VOCAB_SET:
            result.add(tag)
        elif tag in OLD_TO_NEW:
            result.update(OLD_TO_NEW[tag])
        # unknown tags not in OLD_TO_NEW are dropped
    return sorted(result)
