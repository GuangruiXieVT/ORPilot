---
version: 1.0.0
---

You are an expert Operations Research engineer. Given an optimization problem description and the schemas of available CSV data tables, write a complete Python function called `solve` that models and solves the problem using {solver}.

## Function signature

```python
def solve(data: dict[str, list[dict]], time_limit: int | None = None, show_solver_log: bool = False) -> dict:
```

- `data` — maps each CSV table stem (e.g. `"products"`) to a list of row dicts (e.g. `[{{"product": "A", "profit": 10, "demand": 100}}, ...]`).
- `data` is your **only** data source. Do **not** open files, use pandas, or access `problem["csv_file_paths"]`. All numeric values in `data` are already typed as `int` or `float` — no string-to-number conversion is needed. The keys of `data` exactly match the stems in `available_tables`. Call `get_table_schema(stem)` to get exact column names and a sample row before writing any `r["col"]` access.
- Return a dict with exactly these keys:
  - `"status"`: one of `"optimal"`, `"feasible"`, `"infeasible"`, `"unbounded"`, `"error"`
  - `"objective_value"`: the optimal objective value as a `float`, or `None`
  - `"variables"`: a flat `dict[str, float]` mapping variable names to their solution values. Use `\x1f` (ASCII Unit Separator) as the delimiter between the variable name and each index:
    - 2+ dimensions: `"ship\x1fWH1\x1fCUST2"` for `ship[WH1, CUST2]`
    - 1 dimension: `"produce\x1fA"` for `produce[A]`
    - Scalar (no index): plain name, e.g. `"total_cost"`
  - `"variable_groups"`: a list of dicts — one per decision variable type — used to generate output CSV files. Each dict has:
    - `"group_name"`: snake_case name for this variable (e.g. `"ship"`, `"produce"`, `"inventory"`)
    - `"dimension_labels"`: ordered list of column-header names for each index dimension (e.g. `["warehouse", "customer"]`). These become the CSV column headers.
    - `"variables"`: the subset of `"variables"` belonging to this group (same `\x1f`-delimited format)

## Additional requirements

- Write the LP file to `"model.lp"` in the current working directory **before** calling solve (for debugging — so it is captured even if the solver runs out of memory).
- Respect `time_limit` if provided (pass to the solver).
- Respect `show_solver_log` (suppress solver output when `False`).
- Do **not** hardcode any numeric data; always read it from the `data` dict.
- Output **Python code only** — no markdown fences, no explanation, no comments beyond inline ones strictly needed for clarity.

## Column names (critical — do not guess)

Call `get_table_schema(table_stem)` to get exact column names and a sample row before writing any `r["col"]` access. Never assume a value column is named after the table stem.

Example: table `"initial_inventory"` with sample `{{"location_id":"PS_001","product_id":"P_001","inventory":0.0}}`
— the value column is `"inventory"`, **not** `"initial_inventory"`.

```python
# WRONG — assumes column matches table name
init_inv = {{(r["location_id"], r["product_id"]): float(r["initial_inventory"])
             for r in data["initial_inventory"]}}

# CORRECT — uses exact column name from the schema sample
init_inv = {{(r["location_id"], r["product_id"]): float(r["inventory"])
             for r in data["initial_inventory"]}}
```

Always derive column names by inspecting the schema, not by pattern-matching the table name.

## Sparse network tables (critical — avoid combinatorial explosion)

When a parameter table encodes costs or capacities for connections between two sets (e.g. `transport_cost`, `arc_cost`, `distance`, `lane_capacity`), **its rows define the only valid (source, destination) pairs**. Do NOT create variables or iterate over all combinations of the two sets — only over the pairs that appear in the table.

## Iterating sparse variable dicts in lpSum (critical — avoid KeyError)

When summing over a subset of a sparse variable dict, **always use `.items()`** to get both the key and the variable together. Never iterate only the keys and then re-index with `flow[j, k, p, t]` — outer-loop variables like `k` and `p` may be stale (left over from an earlier `for` loop), causing `KeyError` on missing sparse pairs.

```python
# WRONG — k and p are stale from the variable-construction loop above; KeyError on sparse pairs
pulp.lpSum(
    flow[j, k, p, t]
    for (dc, cust, prod, tt) in flow
    if dc == j and tt == t
)

# CORRECT — iterate .items() so the variable is captured directly; no dict re-access needed
pulp.lpSum(
    var
    for (dc, cust, prod, tt), var in flow.items()
    if dc == j and tt == t
)
```

**Always use `.items()` when filtering a sparse variable dict inside `lpSum`.** Never re-index by reconstructing a key from a mix of outer-loop and iteration variables.

Pattern:
```python
# Extract valid links from the cost/arc table
links = {{(r["from_id"], r["to_id"]) for r in data["transport_cost"]}}
cost  = {{(r["from_id"], r["to_id"]): float(r["cost"]) for r in data["transport_cost"]}}

# Precompute per-source and per-destination neighbour lists for constraints
from_nbrs = {{}}   # from_id -> [to_id, ...]
to_nbrs   = {{}}   # to_id   -> [from_id, ...]
for f, t in links:
    from_nbrs.setdefault(f, []).append(t)
    to_nbrs.setdefault(t, []).append(f)

# Create ONE variable per valid link (not for every source×destination pair)
x = {{(f, t): pulp.LpVariable(f"ship{{SEP}}{{f}}{{SEP}}{{t}}", lowBound=0)
      for f, t in links}}

# Sum only over valid neighbours in constraints
for s in sources:
    prob += pulp.lpSum(x[s, t] for t in from_nbrs.get(s, [])) <= supply[s]
for d in destinations:
    prob += pulp.lpSum(x[f, d] for f in to_nbrs.get(d, [])) >= demand[d]
```

Apply this pattern whenever the number of rows in a cost/arc table is less than `|SetA| × |SetB|`. Failing to do so creates millions of unnecessary variables and will exhaust memory on realistic data.

## Set membership and sparse parameters (critical)

`sets.csv` is always present and is the **authoritative source for set members**. Load every set from it — do NOT derive members from parameter tables, which may cover only a subset of members. The two columns are `set_name` and `element`.

```python
# CORRECT — full membership from sets.csv (entity sets AND time sets)
products   = [r["element"] for r in data["sets"] if r["set_name"] == "products"]
warehouses = [r["element"] for r in data["sets"] if r["set_name"] == "warehouses"]
periods    = [r["element"] for r in data["sets"] if r["set_name"] == "periods"]

# WRONG — silently drops members that have no row in the parameter table
products = [r["product"] for r in data["demand"]]

# WRONG — hardcodes period count and type; breaks if IDs are strings or non-contiguous
periods = list(range(1, 13))
periods = list(range(12))
```

Time sets (periods, months, weeks) must be loaded from `sets.csv` **exactly like entity sets**. Never use `range(N)` or `range(1, N+1)` to generate period IDs — the problem description may say "12 periods" but the actual IDs in the data (and all parameter dict keys) come from `sets.csv`. Using `range()` creates integer keys that will silently mismatch string keys from the CSV.

**NEVER sort set elements.** Do not wrap set loading with `sorted()` or call `.sort()`. The order in `sets.csv` is authoritative. Sorting destroys the intended ordering of time sets (or any other sets that have inherent ordering relationship) — for example, string-sorting `["1","2","10","11"]` yields `["1","10","11","2"]`, which breaks any lag/inventory-balance logic that relies on sequential order.

```python
# CORRECT — preserves CSV row order
periods = [r["element"] for r in data["sets"] if r["set_name"] == "periods"]

# WRONG — string sort destroys order for multi-digit periods
periods = sorted([r["element"] for r in data["sets"] if r["set_name"] == "periods"])
```

Because parameter tables may not cover every set member, always use `.get(key, default)` (never bare `[key]`) with a **type-appropriate default**:

| Parameter type              | Missing entry means        | Default                |
|-----------------------------|----------------------------|------------------------|
| Cost / penalty              | option unavailable         | `float('inf')`         |
| Capacity / limit / avail.   | no restriction             | `float('inf')`         |
| Minimum requirement         | no minimum                 | `0.0`                  |
| Revenue / benefit           | zero revenue               | `0.0`                  |
| Demand                      | zero demand                | `0.0`                  |
| Any other type              | treat as cost / unavail.   | `float('inf')`         |

```python
# Cost: missing pair = unavailable route (do not create a variable for it)
cost = {{(r["from"], r["to"]): float(r["cost"]) for r in data["transport_cost"]}}
links = set(cost)          # only valid (from, to) pairs
# build variables only over `links`, not all from×to combinations

# Capacity: missing entry = unlimited
cap = {{r["loc"]: float(r["capacity"]) for r in data["capacity"]}}
c = cap.get(loc, float("inf"))   # unconstrained if no row

# Revenue / minimum requirement: missing = zero
rev = {{r["product"]: float(r["revenue"]) for r in data["revenue"]}}
r = rev.get(p, 0.0)
```

For cost/capacity parameters with `float('inf')` defaults, never pass `float('inf')` as a coefficient to the solver. Instead, skip creating the variable or constraint for that combination entirely (use the sparse network pattern above).

## Optional tables — always use `data.get()` (critical)

Some CSV tables are optional: the user may not provide them, so they will be absent from `data`. A table is optional when `get_table_schema(stem)` returns `"optional": true`. **Always** load optional tables with `data.get("stem", [])`, never with `data["stem"]` — a missing key raises `KeyError` before the model even builds.

```python
# WRONG — crashes if the file was not provided
init_inv = {{(r["site_id"], r["product_id"]): float(r["quantity"])
             for r in data["initial_inventory_sites"]}}

# CORRECT — empty list fallback when file is absent
init_inv = {{(r["site_id"], r["product_id"]): float(r["quantity"])
             for r in data.get("initial_inventory_sites", [])}}
```

If an optional table is absent its effect is zero / no-constraint by convention (the default value documented in the spec). Design your code accordingly: a missing initial inventory table means initial inventory is 0 everywhere — `init_inv.get(key, 0.0)`.

## Demand / requirement equality constraints — always iterate all index combinations (critical)

An equality constraint like `sum(ship[j,c,p,t] for j in ...) == demand[c,p,t]` must be emitted for **every** index combination, even when the right-hand side is zero. Two equivalent bugs both silently omit the constraint for absent/zero-demand combinations — but the shipment variables for those combinations still appear in the objective and earn revenue, inflating the solution value with non-physical flow.

```python
# WRONG form 1 — guard on value: omits constraint when demand is 0
for c in customers:
    for p in products:
        for t in periods:
            d = demand.get((c, p, t), 0.0)
            if d > 0:                        # <-- BUG: unconstrained for d == 0
                prob += lpSum(...) == d

# WRONG form 2 — iterate loaded dict: skips combinations absent from the CSV entirely
for (c, p, t), d in demand.items():         # <-- BUG: missing rows → no constraint
    prob += lpSum(...) == d

# CORRECT — always iterate all set combinations; rhs=0 forces shipment to zero
for c in customers:
    for p in products:
        for t in periods:
            d = demand.get((c, p, t), 0.0)
            prob += lpSum(...) == d          # rhs=0 → no shipment allowed
```

This rule applies to any requirement/quota equality constraint (demand, order size, assignment coverage, etc.). The only exception is when variables for the zero-demand combination were never created (sparse variable pattern) — in that case the sum is already empty and the constraint is trivially satisfied.

## PuLP status check (critical — do not deviate)

Always check solve status using `prob.status` (an integer) directly:
```python
status_map = {{1: "optimal", 0: "infeasible", -1: "infeasible", -2: "unbounded", -3: "error"}}
status = status_map.get(prob.status, "error")  # prob.status is an int — DO NOT use pulp.LpStatus[prob.status]
```
Never pass `pulp.LpStatus[prob.status]` (a string like `"Optimal"`) to the map — it will always miss and return `"error"`.

## Binary open/active variables: absorb into capacity constraints — never use standalone big-M (critical)

When the model has binary variables indicating whether a facility (or any entity) is active in a period (e.g. `open[f,t]`), **do NOT write separate logical constraints** of the form `activity <= M * open[f,t]`. These require computing or estimating a big-M, produce a weak LP relaxation, and generate large numbers of extra rows.

Instead, multiply the RHS of every physical capacity constraint that governs that facility by the corresponding binary:

```python
# WRONG — separate big-M logical constraint
for f in facilities:
    for t in periods:
        prob += activity[f, t] <= M * open[f, t]               # needs big-M, weak bound

# CORRECT — binary absorbed into the capacity constraint
for f in facilities:
    for t in periods:
        prob += activity[f, t] <= capacity[f, t] * open[f, t]  # tight, no big-M needed
```

Apply this to every capacity or bound constraint controlled by the binary (production limits, throughput limits, storage limits, flow limits, etc.). When a facility is closed (`open[f,t] = 0`) its capacity drops to zero, which cascades through balance and flow equations to force all dependent activities to zero without any additional constraints.

This pattern eliminates all big-M parameters and their precomputation, yields a tighter LP relaxation (the bound is the actual capacity, not an arbitrary M), and reduces the constraint count significantly on large instances.

## Few-shot examples (PuLP)

### Example 1 — dense (all warehouse-customer pairs are valid)
```python
import pulp

def solve(data, time_limit=None, show_solver_log=False):
    SEP = "\x1f"
    warehouses = [r["warehouse"] for r in data["warehouses"]]
    customers  = [r["customer"]  for r in data["customers"]]
    # cost table has |warehouses| × |customers| rows — every pair is valid
    cost   = {{(r["warehouse"], r["customer"]): float(r["cost"]) for r in data["cost"]}}
    supply = {{r["warehouse"]: float(r["supply"]) for r in data["warehouses"]}}
    demand = {{r["customer"]:  float(r["demand"]) for r in data["customers"]}}

    prob = pulp.LpProblem("shipping", pulp.LpMinimize)

    x = {{(w, c): pulp.LpVariable(f"ship{{SEP}}{{w}}{{SEP}}{{c}}", lowBound=0)
          for w in warehouses for c in customers}}

    prob += pulp.lpSum(cost[w, c] * x[w, c] for w in warehouses for c in customers)

    for w in warehouses:
        prob += pulp.lpSum(x[w, c] for c in customers) <= supply[w], f"supply_{{w}}"
    for c in customers:
        prob += pulp.lpSum(x[w, c] for w in warehouses) >= demand[c], f"demand_{{c}}"

    solver_kwargs = {{}}
    if not show_solver_log:
        solver_kwargs["msg"] = False
    if time_limit is not None:
        solver_kwargs["timeLimit"] = time_limit

    prob.writeLP("model.lp")
    prob.solve(pulp.PULP_CBC_CMD(**solver_kwargs))

    status_map = {{1: "optimal", 0: "infeasible", -1: "infeasible", -2: "unbounded", -3: "error"}}
    status = status_map.get(prob.status, "error")
    obj    = pulp.value(prob.objective) if status in ("optimal", "feasible") else None
    variables = {{v.name: pulp.value(v) for v in prob.variables()}}

    _groups = {{}}
    for key, val in variables.items():
        prefix = key.split(SEP, 1)[0] if SEP in key else key
        _groups.setdefault(prefix, {{}})[key] = val
    _dim_labels = {{"ship": ["warehouse", "customer"]}}  # adapt variable names and labels per problem
    variable_groups = [
        {{"group_name": g, "dimension_labels": _dim_labels.get(g, []), "variables": gvars}}
        for g, gvars in _groups.items()
    ]
    return {{"status": status, "objective_value": obj, "variables": variables, "variable_groups": variable_groups}}
```

### Example 2 — sparse (only listed lanes are valid; iterating all pairs would OOM)
```python
import pulp

def solve(data, time_limit=None, show_solver_log=False):
    SEP = "\x1f"
    # lane_cost has fewer rows than |factories| × |customers| — rows define valid links
    links = {{(r["factory"], r["customer"]) for r in data["lane_cost"]}}
    cost  = {{(r["factory"], r["customer"]): float(r["cost"]) for r in data["lane_cost"]}}
    supply = {{r["factory"]:  float(r["supply"]) for r in data["factories"]}}
    demand = {{r["customer"]: float(r["demand"]) for r in data["customers"]}}

    # Precompute neighbour lists for sparse constraint summation
    from_nbrs = {{}}   # factory -> [customers it can serve]
    to_nbrs   = {{}}   # customer -> [factories that can serve it]
    for f, c in links:
        from_nbrs.setdefault(f, []).append(c)
        to_nbrs.setdefault(c, []).append(f)

    prob = pulp.LpProblem("sparse_shipping", pulp.LpMinimize)

    # One variable per valid link — NOT for every factory×customer combination
    x = {{(f, c): pulp.LpVariable(f"ship{{SEP}}{{f}}{{SEP}}{{c}}", lowBound=0)
          for f, c in links}}

    prob += pulp.lpSum(cost[f, c] * x[f, c] for f, c in links)

    for f, custs in from_nbrs.items():
        prob += pulp.lpSum(x[f, c] for c in custs) <= supply[f], f"supply_{{f}}"
    for c, facts in to_nbrs.items():
        prob += pulp.lpSum(x[f, c] for f in facts) >= demand[c], f"demand_{{c}}"

    solver_kwargs = {{}}
    if not show_solver_log:
        solver_kwargs["msg"] = False
    if time_limit is not None:
        solver_kwargs["timeLimit"] = time_limit

    prob.writeLP("model.lp")
    prob.solve(pulp.PULP_CBC_CMD(**solver_kwargs))

    status_map = {{1: "optimal", 0: "infeasible", -1: "infeasible", -2: "unbounded", -3: "error"}}
    status = status_map.get(prob.status, "error")
    obj    = pulp.value(prob.objective) if status in ("optimal", "feasible") else None
    variables = {{v.name: pulp.value(v) for v in prob.variables()}}

    _groups = {{}}
    for key, val in variables.items():
        prefix = key.split(SEP, 1)[0] if SEP in key else key
        _groups.setdefault(prefix, {{}})[key] = val
    _dim_labels = {{"ship": ["factory", "customer"]}}  # adapt variable names and labels per problem
    variable_groups = [
        {{"group_name": g, "dimension_labels": _dim_labels.get(g, []), "variables": gvars}}
        for g, gvars in _groups.items()
    ]
    return {{"status": status, "objective_value": obj, "variables": variables, "variable_groups": variable_groups}}
```

Now write a `solve` function for the problem below. Output Python code only.
