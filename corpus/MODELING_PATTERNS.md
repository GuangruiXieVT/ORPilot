# Modeling Patterns Reference

Canonical structural patterns used in the OR corpus. Each example's `modeling_patterns`
field uses exactly the 32 tags defined here. The same vocabulary drives query-side
fingerprint detection in `orpilot/rag/patterns.py`.

---

## Variable Type Patterns

### `binary variable`
A decision variable with `"type": "binary"` (0 or 1).
**Use when:** any yes/no decision — open a facility, assign a worker, activate a route.

### `integer variable`
A decision variable with `"type": "integer"` (whole number, not binary).
**Use when:** quantities that must be whole numbers — truck count, worker headcount, batches.

### `scalar variable`
A variable with `"domain": []` — a single unnamed auxiliary scalar, not indexed over any set.
**Use when:** the objective needs a bound variable for minimax reformulation.

---

## Variable Structure Patterns

### `domain filter`
Variable field `"domain_filter": "param_name"` — the variable is only created for index
combinations where that parameter has a CSV entry.
**Use when:** `row_count < full Cartesian product` of the index columns (sparse network/arc set).

### `exclude diagonal`
Variable field `"exclude_diagonal": true` — indexed over the same set twice and self-loops
`(i, i)` are structurally forbidden.
**Use when:** arc variables in routing where a node cannot travel to itself.

### `duplicate set`
A parameter or variable whose domain contains the same set twice (e.g. `["Locations", "Locations"]`).
**Use when:** self-referential matrices such as travel distance or pairwise cost.

### `upper bound set`
Variable field `"upper_bound_set": "SetName"` — upper bound is `len(SetName)` at solve time.
**Use when:** integer position variables bounded by set cardinality (e.g. MTZ position `u[c,v]`).

---

## Temporal / Sequence Patterns

### `ordered set`
A set with `"ordered": true` — members have a meaningful sequence supporting lag references.
**Use when:** time sets (Periods, Months) where state carries over between steps.

### `temporal lag`
A variable or parameter node in a constraint expression with `"lag": -1` (or `+1`).
**Requires:** `ordered set`. **Always paired with:** `init constraint`.
**Use when:** state variables carry over between periods (inventory, workforce, capacity).

### `init constraint`
A boundary constraint for the first period using `"Periods[0]"` as the time index.
**Always paired with:** `temporal lag` — omitting it leaves the first period unconstrained.

---

## Constraint Patterns

### `big-M linking`
A binary variable multiplied by a parameter to gate whether a continuous quantity is allowed.
**Use when:** an open/close or yes/no decision controls whether a flow or production can be non-zero.

### `mutual exclusion`
At most one (or exactly one) binary variable in a group may equal 1.
**Use when:** either/or decisions, XOR choices, exactly-one-from-group constraints.

### `deviation variables`
A pair of non-negative split variables (surplus + shortfall) representing signed deviation.
**Use when:** goal programming, inventory backlogging, or soft targets with over/under penalties.

### `step cost`
Binary tier-selection variables with an exactly-one-tier constraint and tier-indexed cost.
**Use when:** piecewise/tiered costs — quantity discounts, production tiers, step-function rates.

### `balance constraint`
Flow or material conservation equality: inflow = outflow (possibly + demand).
**Use when:** inventory balance, workforce level balance, node-arc flow conservation.

### `set covering`
Every item must be covered by at least one selected option (classic set covering IP).
**Use when:** facility placement, course scheduling, or any cover-all requirement.

### `MTZ subtour elimination`
Miller-Tucker-Zemlin position variables and constraints to eliminate subtours in routing.
**Use when:** TSP, VRP, or any model requiring a single connected tour without subtours.

### `sparse filter`
Constraint field `"sparse_filter": "param_name"` — skips index combinations absent from the
parameter's CSV.
**Use when:** topology or availability constraints where a missing row means no constraint.

### `filter_column subset`
A set defined by filtering a column of a CSV (e.g. only rows where `set_name = "proteins"`).
**Use when:** the model has subset sets (Proteins ⊂ Foods) loaded from a shared sets.csv.

### `implication constraint`
If binary A = 1 then binary B = 1, encoded as `B ≥ A`.
**Use when:** selection dependencies — choosing X requires Y, activating A forces B.

### `flow conservation`
Node-arc balance in a network: sum(inflow arcs) = sum(outflow arcs) + demand at each node.
**Use when:** network flow, transshipment, multi-commodity flow problems.

### `precedence constraint`
Task A must complete before task B starts: `start[B] ≥ start[A] + duration[A]`.
**Use when:** project scheduling, job-shop, assembly sequences with task dependencies.

### `ratio constraint`
A constraint of the form: at least k% of total must satisfy a condition.
**Use when:** skill ratios, blending purity requirements, portfolio allocation limits.

---

## Objective Pattern

### `minimax objective`
Minimize the maximum value across a set, reformulated with a scalar auxiliary bound variable.
**Use when:** fairest assignment, minimize peak load, equalize workload.

---

## Data / Parameter Patterns

### `BOM parameter`
A bill-of-materials: component consumption rate matrix indexed over (Products, Components).
**Use when:** production requires component inputs in fixed proportions per finished unit.

### `scenario weighted sum`
Probability-weighted sum over named scenarios in the objective.
**Use when:** explicit probability weights on scenarios, expected-value objectives.

### `two-stage stochastic`
Here-and-now first-stage decisions + scenario-dependent recourse second stage.
**Use when:** the model has a clear first/second stage split with recourse variables.

### `set size expression`
Expression node `{"type": "set_size", "set": "SetName"}` compiling to `len(SetName)`.
**Use when:** big-M equals set cardinality (MTZ subtour elimination position bounds).

### `pre-enumerated options`
Feasible combinations listed explicitly as CSV rows; variable indexed over an options set.
**Use when:** cutting stock patterns, option bundles — any model where valid combinations
are pre-computed rather than constructed from set products.

---

## Structural Patterns

### `subset indexed variable`
A variable or constraint domain uses a proper subset set (e.g. Proteins ⊂ Foods).
**Use when:** different granularities of the same item type appear in the same model.

### `no sets`
All-scalar model — no set-indexed variables or parameters, no CSV set data.
**Use when:** small algebraic LP/IP problems solvable with only scalar constants.

---

