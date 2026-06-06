# IR Patterns Reference

Structural patterns used in the IR schema. Each corpus example's `ir_patterns` field
is populated from this vocabulary by `scripts/update_corpus_ir_patterns.py`.

---

## Variable Patterns

### `binary variable`
A decision variable with `"type": "binary"` (0 or 1).  
**Use when:** any yes/no decision — open a facility, assign a worker, activate a route.  
**Examples (29):** assignment, facility_location, lot_sizing, vrp, network_*_open_close, …

### `integer variable`
A decision variable with `"type": "integer"` (whole number, not binary).  
**Use when:** quantities that must be whole numbers — number of workers per shift, truck count, shipment batches.  
**Examples (8):** production, production_simple, scheduling_shift_planning, transportation_shipment_sizing, vrp, …

### `scalar variable`
A variable with `"domain": []` — a single unnamed auxiliary scalar, not indexed over any set.  
**Use when:** the objective needs a bound variable for min-max reformulation (e.g. minimax workload).  
**Examples (1):** minimax_workload

### `exclude diagonal`
Variable field `"exclude_diagonal": true` — the variable is indexed over the same set twice
and self-loops `(i, i)` are structurally forbidden.  
**Use when:** arc variables in routing where a location cannot travel to itself.  
**Examples (2):** vrp, vrp_time_windows

### `upper bound set`
Variable field `"upper_bound_set": "SetName"` — the variable's upper bound is `len(SetName)`
at solve time, not a hardcoded constant.  
**Use when:** integer position variables whose natural ceiling is the cardinality of a set
(e.g. MTZ subtour-elimination position `u[c,v]` bounded by number of customers).  
**Examples (2):** vrp, vrp_time_windows

### `domain filter`
Variable field `"domain_filter": "param_name"` — the variable is only created for index
combinations where that parameter has a CSV entry (sparse network).  
**Use when:** `row_count < full Cartesian product` of the parameter's index columns, and the
variable should not exist for invalid arcs/pairs.  
**Examples (17):** assignment_skill_based, facility_location, network_transportation,
network_production_inventory_transportation, single_source, …

---

## Parameter Patterns

### `scalar parameter`
A parameter with `"domain": []` — a single global value, not indexed over any set.  
**Use when:** problem-wide constants such as warehouse capacity, holding cost rate,
or order quantity limit.  
**Examples (14):** inventory_single_product, inventory_multi_products, knapsack,
bin_packing, lot_sizing, …

### `duplicate set`
A parameter or constraint whose domain contains the same set twice
(e.g. `domain: ["Locations", "Locations"]`).  
**Use when:** self-referential matrices such as travel distance or cost between members
of the same set; also required for MTZ constraints over `[Customers, Customers, Trips]`.  
**Examples (3):** transportation_flow_allocation, vrp, vrp_time_windows

### `BOM parameter`
A parameter encoding a bill-of-materials — a yield, recipe, or consumption ratio
indexed over two different set types (e.g. `yield[Products, Components]`).  
**Use when:** production requires component inputs in fixed proportions per finished unit.  
**Examples (4):** production_bom, production_multi_boms,
network_production_inventory_transportation_bom,
network_production_inventory_transportation_bom_wc

### `scenario weighted sum`
A probability parameter (named `prob_*` or `scenario_*`) used to weight recourse terms
in a two-stage stochastic objective: `sum_s prob[s] * recourse_cost[s]`.  
**Use when:** the problem has explicit named scenarios each with a probability weight.  
**Examples (1):** demand_scenarios

---

## Set Patterns

### `ordered set`
A set with `"ordered": true` — members have a meaningful sequence and support
lag references (`"lag": -1` or `+1`).  
**Use when:** time sets (Periods, Months, Weeks) where state carries over between steps.  
**Examples (14):** all multi-period inventory, lot_sizing, capacity_expansion,
scheduling_workforce_adjustment, network_production_inventory_transportation, …

---

## Constraint / Expression Patterns

### `temporal lag`
A variable or parameter node in a constraint expression with `"lag": -1` (or `+1`) —
references the value from the previous (or next) period.  
**Requires:** `ordered set` on the time dimension.  
**Use when:** state variables carry over between periods (inventory balance, workforce level,
cumulative capacity).  
**Examples (14):** all multi-period inventory, lot_sizing, capacity_expansion,
scheduling_workforce_adjustment, network_production_inventory_transportation, …

### `init constraint`
A boundary constraint for the first period using `"Periods[0]"` (SetName[N] syntax) as the
time index instead of a loop variable.  
**Always paired with:** `temporal lag` — every lag-based constraint needs a corresponding
init constraint; omitting it leaves the first period unconstrained.  
**Use when:** inventory balance, workforce adjustment, or any state variable with lag.  
**Examples (16):** same as temporal lag plus vrp, vrp_time_windows

### `sparse filter`
Constraint field `"sparse_filter": "param_name"` — the compiler emits
`if key not in param: continue` before the constraint body, skipping combinations
absent from the parameter's CSV.  
**Use when:** topology or availability constraints where a missing row means
"no constraint" (not "RHS = 0"). Never use on demand, balance, or equality constraints.  
**Examples (10):** network_transportation, network_production_inventory_transportation family,
transportation_flow_count, assignment_operator_job_period, …

### `big-M linking`
A binary variable multiplied by a parameter in a constraint expression
(`multiply(param, binary_var)`) to gate whether a continuous quantity is allowed.  
**Use when:** an open/close or yes/no decision controls whether a flow, production, or
assignment can be non-zero.  
**Examples (21):** facility_location, single_source, lot_sizing, make_or_buy,
capacity_expansion, vrp_time_windows, network_*_open_close, …

### `step cost`
Binary tier-selection variables with a `one_tier` (or similar) equality constraint ensuring
exactly one tier is active per entity per period, plus a tier-indexed cost parameter.  
**Use when:** piecewise/tiered operating costs — quantity discounts, production tiers,
step-function shipping rates.  
**Examples (4):** procurement_quantity_discounts, network_production_inventory_transportation_step_cost,
network_production_inventory_transportation_open_close,
network_production_inventory_transportation_step_transport

### `set size expression`
An expression node `{"type": "set_size", "set": "SetName"}` that compiles to `len(SetName)`
at solve time — used as a big-M whose value equals a set's cardinality.  
**Use when:** MTZ subtour-elimination position variables need an upper bound equal to
the number of customers (not a hardcoded integer).  
**Examples (2):** vrp, vrp_time_windows

### `deviation variables`
A pair of non-negative split variables (surplus + shortfall, or backlog + inventory)
representing the signed deviation from a target or balance equation.  
**Use when:** goal programming (soft targets with penalties), inventory backlogging
(unmet demand carried forward), or any model where both over- and under-achievement
must be tracked separately.  
**Examples (3):** goal_programming, inventory_backlogging, demand_scenarios

---

## Pattern Co-occurrence

Patterns that always or almost always appear together:

| Pattern | Always comes with |
|---|---|
| `temporal lag` | `ordered set`, `init constraint` |
| `init constraint` | `temporal lag`, `ordered set` |
| `exclude diagonal` | `duplicate set`, `upper bound set`, `set size expression` (VRP family) |
| `upper bound set` | `exclude diagonal`, `set size expression` |
| `set size expression` | `exclude diagonal`, `upper bound set` |
| `step cost` | `binary variable`, `big-M linking` |
| `BOM parameter` | typically `domain filter`, `sparse filter` in network models |
