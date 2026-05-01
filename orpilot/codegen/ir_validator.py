"""Semantic validation of IR dicts beyond Pydantic schema checks.

These checks catch modeling errors that are structurally valid JSON but physically
wrong — for example, inventory balance constraints where outflow variables have the
wrong sign, making the model infeasible.
"""

from __future__ import annotations

from collections import defaultdict


# ---------------------------------------------------------------------------
# Expression tree helpers
# ---------------------------------------------------------------------------

def _compute_coefficients(expr: dict, sign: int = 1) -> dict[str, int]:
    """Walk an expression tree and return the net integer coefficient for each
    variable name referenced anywhere in the tree.

    sign propagation:
      sum(A, B)      → A gets sign, B gets sign
      subtract(A, B) → A gets sign, B gets -sign
      indexed_sum    → body gets sign
      multiply(C, V) → variable V gets sign (constant C ignored)
      variable node  → variable.name gets sign
      parameter/constant/set_size → ignored
    """
    result: dict[str, int] = defaultdict(int)

    node_type = expr.get("type")
    op = expr.get("operation")

    if node_type == "variable":
        result[expr["name"]] += sign

    elif node_type in ("parameter", "constant", "set_size"):
        pass

    elif op == "sum":
        for k, v in _compute_coefficients(expr.get("left", {}), sign).items():
            result[k] += v
        for k, v in _compute_coefficients(expr.get("right", {}), sign).items():
            result[k] += v

    elif op == "subtract":
        for k, v in _compute_coefficients(expr.get("left", {}), sign).items():
            result[k] += v
        for k, v in _compute_coefficients(expr.get("right", {}), -sign).items():
            result[k] += v

    elif op == "indexed_sum":
        for k, v in _compute_coefficients(expr.get("body", {}), sign).items():
            result[k] += v

    elif op == "multiply":
        left = expr.get("left", {})
        right = expr.get("right", {})
        # Walk both sides — constants contribute nothing, variables get the sign
        for k, v in _compute_coefficients(left, sign).items():
            result[k] += v
        for k, v in _compute_coefficients(right, sign).items():
            result[k] += v

    return dict(result)


def _has_lag(expr: dict) -> bool:
    """Return True if any variable/parameter node in the tree has a non-zero lag."""
    if expr.get("lag", 0) != 0:
        return True
    for key in ("left", "right", "body"):
        child = expr.get(key)
        if child and _has_lag(child):
            return True
    return False


def _find_variable_nodes(expr: dict) -> list[str]:
    """Return names of all variable nodes found anywhere in an expression tree."""
    result: list[str] = []
    if expr.get("type") == "variable":
        result.append(expr.get("name", "?"))
    for key in ("left", "right", "body"):
        child = expr.get(key)
        if child:
            result.extend(_find_variable_nodes(child))
    return result


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def _check_inventory_balance_signs(ir: dict) -> list[str]:
    """Detect inventory balance constraints where outflow variables have wrong sign.

    Correct identity:  ending_inv = beginning_inv + inflow - outflow
    Encoded as LHS=0:  inv[t] - inv[t-1] - inflow[t] + outflow[t] = 0
    Or with non-zero initial inventory on RHS:
                       inv[t0] + outflow[t0] - inflow[t0] = initial_inventory

    The common LLM mistake: all flow variables are subtracted:
        inv[t] - inv[t-1] - inflow[t] - outflow[t] = 0   (WRONG)
        inv[t0] - outflow[t0] - inflow[t0] = initial_inventory  (WRONG)

    We apply this check to two constraint shapes:
    1. Lag-based (subsequent periods): has a lag: -1 variable in the expression.
       The inventory var has net coeff 0 (appears as +1 current and -1 lag).
       All other decision variables should NOT all be negative.
    2. Init (first period, no lag): equality with "init" in the constraint name.
       Has exactly one decision variable with positive coefficient (the inventory)
       and 2+ others with negative coefficient.
       Works regardless of whether initial_inventory is on the RHS or moved to LHS.
    """
    errors = []
    variables = set(ir.get("variables", {}).keys())

    for cname, cspec in ir.get("constraints", {}).items():
        if cspec.get("sense") != "=":
            continue

        expr = cspec.get("expression", {})
        has_lag = _has_lag(expr)
        is_init = "init" in cname.lower()

        if not has_lag and not is_init:
            continue

        coeffs = _compute_coefficients(expr)
        neg_vars = [n for n, c in coeffs.items() if c < 0 and n in variables]
        pos_vars = [n for n, c in coeffs.items() if c > 0 and n in variables]
        zero_vars = [n for n, c in coeffs.items() if c == 0 and n in variables]
        # zero_vars: appear as both current (+1) and lagged (-1) — this IS the
        # inventory variable in lag-based constraints.

        # Identify inventory variables by name heuristic
        inv_name_match = lambda n: "inventory" in n.lower() or n.lower().startswith("inv")

        # --- Pattern A: init constraints (no lag) ---
        # Wrong: inventory(+1), inflow(-1), outflow(-1)
        # The inventory variable is in pos_vars; 2+ flow vars are negative.
        inv_vars_pos = [n for n in pos_vars if inv_name_match(n)]
        inv_vars_neg = [n for n in neg_vars if inv_name_match(n)]
        non_inv_neg = [n for n in neg_vars if n not in inv_vars_neg]

        if not has_lag and is_init and inv_vars_pos and len(non_inv_neg) >= 2:
            errors.append(
                f"Constraint '{cname}': inventory balance has wrong signs. "
                f"Inventory variable(s) {inv_vars_pos} have coefficient +1, "
                f"but flow variables {non_inv_neg} all have coefficient -1. "
                f"This encodes inv = inflow + outflow (WRONG). "
                f"Outflow variables must be positive on the LHS: "
                f"subtract(sum(inventory[t0], indexed_sum(outflow)), inflow[t0]) = 0 "
                f"→ inv = inflow - outflow (correct)."
            )

        # --- Pattern B: lag-based constraints ---
        # The inventory variable cancels to net 0 (current +1, lag -1).
        # Wrong: zero_var=inventory, neg_vars has both inflow and outflow (both subtracted).
        # Correct: zero_var=inventory, exactly one of {inflow,outflow} is negative,
        #          the other (outflow) is positive.
        if has_lag:
            inv_zero = [n for n in zero_vars if inv_name_match(n)]
            flow_neg = [n for n in neg_vars if n not in inv_vars_neg]
            flow_pos = [n for n in pos_vars if not inv_name_match(n)]
            # Wrong: inventory nets to 0, and ALL flow vars are negative (2+)
            if inv_zero and len(flow_neg) >= 2 and len(flow_pos) == 0:
                errors.append(
                    f"Constraint '{cname}': inventory balance has wrong signs. "
                    f"Inventory variable(s) {inv_zero} net to 0 (current - lag), "
                    f"but ALL flow variables {flow_neg} have coefficient -1. "
                    f"This encodes inv[t] = inv[t-1] + inflow + outflow (WRONG). "
                    f"Outflow must be positive: "
                    f"subtract(sum(subtract(inv[t], inv[t-1,lag]), indexed_sum(outflow)), inflow[t]) = 0 "
                    f"→ inv[t] = inv[t-1] + inflow - outflow (correct)."
                )

    return errors


def _check_lag_without_init(ir: dict) -> list[str]:
    """Check that every lag-based inventory balance has a corresponding _init constraint."""
    errors = []
    constraints = ir.get("constraints", {})
    lag_constraints = [
        name for name, spec in constraints.items()
        if spec.get("sense") == "=" and _has_lag(spec.get("expression", {}))
    ]
    for cname in lag_constraints:
        init_name = cname + "_init"
        if init_name not in constraints:
            errors.append(
                f"Constraint '{cname}' uses lag: -1 but no corresponding init constraint "
                f"('{init_name}') was found. The lag-based constraint skips the first period "
                f"(via 'if _idx_t < 1: continue'), leaving period-0 inventory unconstrained "
                f"and the model potentially unbounded. Add '{init_name}' with domain that "
                f"excludes the time set and uses 'Periods[0]' as the time index."
            )
    return errors


def _check_period_set_with_hardcoded_size(ir: dict) -> list[str]:
    """Warn when a time set uses hardcoded size instead of CSV member list."""
    errors = []
    for sname, sspec in ir.get("sets", {}).items():
        if sspec.get("ordered") and sspec.get("size") is not None and not sspec.get("source"):
            errors.append(
                f"Set '{sname}' is ordered (time set) but uses hardcoded size={sspec['size']} "
                f"with no CSV source. The compiler emits list(range({sspec['size']})) = "
                f"[0..{sspec['size']-1}] (integers). If period IDs in your data are strings "
                f"like '1','2',...,'{sspec['size']}' or 'Jan','Feb',..., all period-indexed "
                f"parameters will have keys that never match, silently skipping every "
                f"period-indexed constraint and making the model unbounded. "
                f"Use MEMBER LIST FROM CSV (source + column, with filter_column/filter_value "
                f"if using a shared sets.csv) so the set contains the actual string IDs."
            )
    return errors


def _check_rhs_variable_nodes(ir: dict) -> list[str]:
    """Check that no constraint RHS is a bare variable node at the top level.

    A standalone variable as the entire RHS (e.g. rhs = {type: variable, name: y})
    means the constraint is encoded as `expr = y` — but the compiler expects the RHS
    to be a constant or parameter expression. Move the variable to the LHS:
      expression = subtract(lhs_expr, y), rhs = constant 0.

    Note: parameter × variable on the RHS (big-M pattern: capacity * open_site)
    is valid linear MIP and is NOT flagged — only a bare root-level variable is flagged.
    """
    errors = []
    for cname, cspec in ir.get("constraints", {}).items():
        rhs = cspec.get("rhs", {})
        if rhs.get("type") == "variable":
            errors.append(
                f"Constraint '{cname}': rhs is a bare variable node ('{rhs.get('name')}'). "
                f"Move the variable to the LHS expression via subtract: "
                f"expression = subtract(lhs_expr, {rhs.get('name')}[...]), rhs = constant 0."
            )
    return errors


def _check_variable_times_variable(ir: dict) -> list[str]:
    """Check that no multiply node multiplies two variable-containing sub-expressions.

    Only constant×variable, constant×parameter, and parameter×variable are linear.
    Variable×variable is quadratic and forbidden.
    """
    errors: list[str] = []

    def check_expr(expr: dict, location: str) -> None:
        if expr.get("operation") == "multiply":
            left_vars = _find_variable_nodes(expr.get("left", {}))
            right_vars = _find_variable_nodes(expr.get("right", {}))
            if left_vars and right_vars:
                errors.append(
                    f"Nonlinear term in {location}: multiply node has variable(s) "
                    f"{left_vars} on the left and {right_vars} on the right. "
                    f"Variable × variable is forbidden — only constant×variable "
                    f"or parameter×variable are allowed."
                )
        for key in ("left", "right", "body"):
            child = expr.get(key)
            if child:
                check_expr(child, location)

    for cname, cspec in ir.get("constraints", {}).items():
        check_expr(cspec.get("expression", {}), f"constraint '{cname}' expression")
        check_expr(cspec.get("rhs", {}), f"constraint '{cname}' rhs")
    obj = ir.get("objective", {})
    check_expr(obj.get("expression", {}), "objective")
    return errors


def _check_lag_in_objective(ir: dict) -> list[str]:
    """Check that the objective expression contains no lagged nodes.

    Lag is only meaningful in constraint expressions where the compiler can emit
    an enumerate()-based loop with a boundary guard. In the objective, there is
    no such loop, so lag is undefined.
    """
    errors = []
    obj_expr = ir.get("objective", {}).get("expression", {})
    if _has_lag(obj_expr):
        errors.append(
            "Objective contains a lagged variable or parameter (lag != 0). "
            "Lag is only valid in constraint expressions (the compiler emits a "
            "boundary-guarded enumerate loop there). Remove all lag references "
            "from the objective."
        )
    return errors


def _check_alias_in_domain(ir: dict) -> list[str]:
    """Check that no domain field uses the 'SetName:alias' syntax.

    The alias syntax (e.g. 'Locations:k') is only valid inside indexed_sum 'over'
    arrays. Using it in a variable, constraint, or parameter domain field is
    a schema error that will produce a KeyError at compile time.
    """
    errors = []
    for vname, vspec in ir.get("variables", {}).items():
        for s in vspec.get("domain", []):
            if ":" in str(s):
                errors.append(
                    f"Variable '{vname}' domain entry '{s}' uses 'SetName:alias' syntax. "
                    f"Alias notation is only valid inside indexed_sum 'over' arrays. "
                    f"Use plain set names (e.g. 'Locations', not 'Locations:k') in domain."
                )
    for cname, cspec in ir.get("constraints", {}).items():
        for s in cspec.get("domain", []):
            if ":" in str(s):
                errors.append(
                    f"Constraint '{cname}' domain entry '{s}' uses 'SetName:alias' syntax. "
                    f"Use plain set names in domain fields."
                )
    for pname, pspec in ir.get("parameters", {}).items():
        for s in pspec.get("domain", []):
            if ":" in str(s):
                errors.append(
                    f"Parameter '{pname}' domain entry '{s}' uses 'SetName:alias' syntax. "
                    f"Use plain set names in domain fields."
                )
    return errors


def _check_shared_source_without_filter(ir: dict) -> list[str]:
    """Check that sets sharing the same source file + column all use filter_column.

    When a single CSV stores members for multiple sets distinguished by a category
    column, every set that reads from it must specify filter_column and filter_value.
    Without a filter every set gets all rows, making them identical.
    """
    errors = []
    source_col_groups: dict = defaultdict(list)
    for sname, sspec in ir.get("sets", {}).items():
        source = sspec.get("source")
        column = sspec.get("column")
        if source and column:
            source_col_groups[(source, column)].append(sname)

    for (source, column), snames in source_col_groups.items():
        if len(snames) > 1:
            missing_filter = [n for n in snames if not ir["sets"][n].get("filter_column")]
            if missing_filter:
                errors.append(
                    f"Sets {missing_filter} share source '{source}' + column '{column}' "
                    f"without filter_column/filter_value. This gives all these sets the same "
                    f"members (all rows of that column). Add filter_column and filter_value "
                    f"to each set so only the rows belonging to it are selected."
                )
    return errors


def _check_duplicate_domain_index_columns(ir: dict) -> list[str]:
    """Check that parameters with duplicate sets in their domain supply index_columns.

    When the same set appears more than once in a parameter's domain (e.g.
    distance[Locations, Locations]), the compiler cannot infer which CSV column
    identifies each domain position. index_columns must be provided explicitly.
    """
    errors = []
    for pname, pspec in ir.get("parameters", {}).items():
        domain = pspec.get("domain", [])
        if len(domain) != len(set(domain)) and not pspec.get("index_columns"):
            dup = [s for s in set(domain) if domain.count(s) > 1]
            errors.append(
                f"Parameter '{pname}' has duplicate set(s) {dup} in its domain "
                f"but index_columns is null. When the same set appears more than once, "
                f"index_columns must specify the CSV column name for each domain position "
                f"(e.g. ['from_id', 'to_id'] for a distance[Locations, Locations] parameter)."
            )
    return errors


def _check_sparse_filter_domain(ir: dict) -> list[str]:
    """Check that a constraint's sparse_filter parameter domain is a subset of the constraint domain.

    The compiler emits 'if key not in param: continue' using a key built from the
    constraint's loop variables. If the parameter is indexed over a set that is not
    in the constraint's domain, the corresponding loop variable is never defined and
    the guard key would be malformed at runtime.
    """
    errors = []
    parameters = ir.get("parameters", {})
    for cname, cspec in ir.get("constraints", {}).items():
        sf = cspec.get("sparse_filter")
        if not sf:
            continue
        param = parameters.get(sf)
        if param is None:
            errors.append(
                f"Constraint '{cname}' has sparse_filter='{sf}' but no parameter "
                f"named '{sf}' exists in the IR."
            )
            continue
        constraint_domain_set = set(cspec.get("domain", []))
        param_domain = param.get("domain", [])
        missing = [s for s in param_domain if s not in constraint_domain_set]
        if missing:
            errors.append(
                f"Constraint '{cname}' sparse_filter='{sf}': parameter '{sf}' has "
                f"domain set(s) {missing} that are not in the constraint domain "
                f"{sorted(constraint_domain_set)}. The filter guard key would reference "
                f"undefined loop variables at runtime. Remove sparse_filter or add the "
                f"missing sets to the constraint's domain."
            )
    return errors


def _check_domain_filter_domain(ir: dict) -> list[str]:
    """Check that a variable's domain_filter parameter domain is a subset of the variable domain.

    The compiler emits 'if (i, j) in param' during variable creation using the
    variable's own loop variables. If the parameter is indexed over a set not in
    the variable's domain, the membership test key would be incomplete.
    """
    errors = []
    parameters = ir.get("parameters", {})
    for vname, vspec in ir.get("variables", {}).items():
        df = vspec.get("domain_filter")
        if not df:
            continue
        param = parameters.get(df)
        if param is None:
            errors.append(
                f"Variable '{vname}' has domain_filter='{df}' but no parameter "
                f"named '{df}' exists in the IR."
            )
            continue
        variable_domain_set = set(vspec.get("domain", []))
        param_domain = param.get("domain", [])
        missing = [s for s in param_domain if s not in variable_domain_set]
        if missing:
            errors.append(
                f"Variable '{vname}' domain_filter='{df}': parameter '{df}' has "
                f"domain set(s) {missing} not in the variable domain "
                f"{sorted(variable_domain_set)}. The filter membership key would be "
                f"incomplete. Add the missing sets to the variable's domain."
            )
    return errors


def _check_objective_subtract_from_zero(ir: dict) -> list[str]:
    """Detect 'subtract(constant 0, expr)' anywhere in the objective expression.

    Two common LLM mistakes share this root cause:

    Bug 1 — starting from zero:
      subtract(constant 0, revenue_sum) = 0 - revenue = -revenue
      The LLM chains costs from 0 instead of starting from the first positive term.

    Bug 2 — subtract nested on the right side of another subtract:
      subtract(A, subtract(B, C)) = A - (B - C) = A - B + C
      C (e.g. holding_cost) ends up with a POSITIVE coefficient — added instead of
      subtracted.  The LLM may have written subtract(cost2, cost3) intending both
      to be subtracted, but only cost2 is subtracted; cost3 is added.

    Both produce 'subtract(constant 0, ...)' somewhere in the tree because the LLM
    pads an expression with "0 - term" rather than using the term directly.

    Correct pattern for maximize profit = revenue - cost1 - cost2 - cost3:
      subtract(subtract(subtract(revenue, cost1), cost2), cost3)  ← left-to-right chain
      NEVER: subtract(A, subtract(cost2, cost3))  ← flips cost3 sign to positive
      NEVER: subtract(constant 0, revenue)        ← negates revenue
    """
    errors: list[str] = []

    def _find_zero_subtracts(expr: dict, path: str) -> list[str]:
        found: list[str] = []
        if (expr.get("operation") == "subtract"
                and expr.get("left", {}).get("type") == "constant"
                and expr.get("left", {}).get("value") == 0):
            found.append(path)
        for key in ("left", "right", "body"):
            child = expr.get(key)
            if child:
                found.extend(_find_zero_subtracts(child, f"{path}.{key}"))
        return found

    obj_expr = ir.get("objective", {}).get("expression", {})
    bad_paths = _find_zero_subtracts(obj_expr, "objective.expression")
    if bad_paths:
        errors.append(
            f"Objective contains subtract(constant 0, ...) at: {bad_paths}. "
            f"This negates a term: '0 - revenue' subtracts revenue instead of adding it, "
            f"and 'subtract(A, subtract(B, C))' adds C instead of subtracting it. "
            f"Build the objective as a left-to-right chain of subtracts: "
            f"subtract(subtract(subtract(revenue, cost1), cost2), cost3) "
            f"= revenue - cost1 - cost2 - cost3. Never start from constant 0."
        )
    return errors


def _check_objective_nested_subtract(ir: dict) -> list[str]:
    """Detect subtract(A, subtract(B, C)) anywhere in the objective expression.

    subtract(A, subtract(B, C)) = A - B + C  — C ends up with a positive
    coefficient, i.e. it is added instead of subtracted.  The correct pattern
    for a chain of subtractions is a left-associative tree:
      subtract(subtract(subtract(A, B), C), D) = A - B - C - D

    This is a distinct bug from subtract(constant 0, ...) and is not caught by
    _check_objective_subtract_from_zero.
    """
    errors: list[str] = []

    def _find_nested(expr: dict, path: str) -> list[str]:
        found: list[str] = []
        if (expr.get("operation") == "subtract"
                and expr.get("right", {}).get("operation") == "subtract"):
            # Describe what flips sign
            right = expr["right"]
            right_right = right.get("right", {})
            # Summarise what's in the inner right subtree
            inner_vars = _find_variable_nodes(right_right)
            inner_params = []

            def _collect_params(e: dict) -> None:
                if e.get("type") == "parameter":
                    inner_params.append(e.get("name", "?"))
                for k in ("left", "right", "body"):
                    ch = e.get(k)
                    if ch:
                        _collect_params(ch)

            _collect_params(right_right)
            flipped = inner_vars + inner_params
            found.append(f"{path} (term(s) with flipped sign: {flipped})")
        for key in ("left", "right", "body"):
            child = expr.get(key)
            if child:
                found.extend(_find_nested(child, f"{path}.{key}"))
        return found

    obj_expr = ir.get("objective", {}).get("expression", {})
    bad_paths = _find_nested(obj_expr, "objective.expression")
    if bad_paths:
        errors.append(
            f"Objective contains subtract(A, subtract(B, C)) at: {bad_paths}. "
            f"This evaluates to A - B + C — the innermost term C ends up ADDED instead of "
            f"subtracted. Build the objective as a strictly left-to-right chain: "
            f"subtract(subtract(subtract(revenue, cost1), cost2), cost3) "
            f"= revenue - cost1 - cost2 - cost3. "
            f"Never place a subtract node on the RIGHT side of another subtract."
        )
    return errors


def _collect_param_names_in_expr(expr: dict) -> set[str]:
    """Walk an expression tree and return all parameter names referenced."""
    names: set[str] = set()
    if not isinstance(expr, dict):
        return names
    if expr.get("type") == "parameter":
        names.add(expr["name"])
    for child_key in ("left", "right", "body"):
        child = expr.get(child_key)
        if isinstance(child, dict):
            names |= _collect_param_names_in_expr(child)
    return names


def _check_cost_params_not_in_objective(ir: dict) -> list[str]:
    """Detect cost/revenue/holding parameters declared in IR but absent from the objective.

    A parameter whose name contains a cost/revenue keyword that never appears in the
    objective expression is almost always an omission — the LLM declared it but forgot
    to add the corresponding term (e.g. dropped holding_cost_sites from the maximize
    profit objective while still using it in inventory constraints).
    """
    OBJECTIVE_KEYWORDS = ("cost", "holding", "revenue", "profit", "penalty", "price")
    obj_expr = ir.get("objective", {}).get("expression", {})
    referenced = _collect_param_names_in_expr(obj_expr)
    errors: list[str] = []
    for pname in ir.get("parameters", {}):
        if any(kw in pname.lower() for kw in OBJECTIVE_KEYWORDS):
            if pname not in referenced:
                errors.append(
                    f"Parameter '{pname}' contains a cost/revenue keyword but is not "
                    f"referenced anywhere in the objective expression. This is almost "
                    f"certainly an omission — add a term for '{pname}' to the objective "
                    f"(subtract for costs, add for revenues/prices)."
                )
    return errors


def _check_cost_capacity_missing_default(ir: dict) -> list[str]:
    """Flag cost and capacity parameters that have missing_default='zero'.

    Missing entries in cost/capacity tables mean the route or option does not
    exist — not that it is free. Using 'zero' as the default silently makes
    unavailable routes/capacities appear free or unlimited, producing an
    infeasible or incorrect model.

    Rule:
      - Parameter name contains 'cost', 'capacity', 'time', or 'distance'
        → missing_default MUST be 'inf'
      - Parameters named for demand, revenue, etc. may use 'zero'
    """
    COST_KEYWORDS = ("cost", "capacity", "distance", "penalty", "duration")
    errors: list[str] = []
    for pname, pspec in ir.get("parameters", {}).items():
        # Scalar parameters (domain=[]) have no "missing combinations" — skip
        if not pspec.get("domain"):
            continue
        name_lower = pname.lower()
        if any(kw in name_lower for kw in COST_KEYWORDS):
            if pspec.get("missing_default", "zero") == "zero":
                errors.append(
                    f"Parameter '{pname}' looks like a cost/capacity parameter but has "
                    f"missing_default='zero'. A missing entry means the route or option "
                    f"does not exist — not that it is free. Set missing_default='inf' so "
                    f"the compiler uses float('inf') for absent combinations, making them "
                    f"prohibitively expensive and naturally excluded by the optimizer."
                )
    return errors


def _check_init_lag_sign_consistency(ir: dict) -> list[str]:
    """Check that init and lag-based inventory balance constraints use consistent flow signs.

    The init constraint (period 0) and the lag constraint (periods 1..T) must assign
    the same sign to each shared flow variable. Swapping inflows and outflows between
    the two constraints is a common LLM error:

      Lag (correct):  (inv[t] - inv[t-1]) - inflows + outflows = 0
        → inflows: -1, outflows: +1
      Init (WRONG):   inv[t0] + inflows - outflows = init_inv
        → inflows: +1, outflows: -1  ← sign-flipped vs lag

    We detect this by comparing each flow variable's net coefficient in the lag
    constraint against its coefficient in the corresponding init constraint.
    """
    errors = []
    constraints = ir.get("constraints", {})
    variables = set(ir.get("variables", {}).keys())
    inv_name_match = lambda n: "inventory" in n.lower() or n.lower().startswith("inv")

    for cname, cspec in constraints.items():
        if cspec.get("sense") != "=":
            continue
        if not _has_lag(cspec.get("expression", {})):
            continue

        # Find corresponding init constraint
        init_name = cname + "_init"
        base_init = (cname.replace("_balance_dc", "_balance_dc_init")
                         .replace("_balance_site", "_balance_site_init"))
        init_cname = init_name if init_name in constraints else (
            base_init if base_init in constraints else None
        )
        if not init_cname:
            continue  # _check_lag_without_init handles this

        lag_coeffs = _compute_coefficients(cspec["expression"])
        init_coeffs = _compute_coefficients(constraints[init_cname]["expression"])

        # Flow variables: appear in lag constraint with non-zero net coefficient
        # (inventory variables net to 0 because of current - lag)
        lag_flow_signs = {
            n: c for n, c in lag_coeffs.items()
            if c != 0 and n in variables and not inv_name_match(n)
        }

        mismatches = []
        for vname, lag_sign in lag_flow_signs.items():
            if vname in init_coeffs:
                init_sign = init_coeffs[vname]
                if init_sign != lag_sign:
                    mismatches.append((vname, lag_sign, init_sign))

        if mismatches:
            detail = ", ".join(
                f"'{v}' (lag={ls:+d}, init={is_:+d})"
                for v, ls, is_ in mismatches
            )
            errors.append(
                f"Constraint '{init_cname}': flow variable signs are inconsistent with "
                f"lag constraint '{cname}'. Mismatches: {detail}. "
                f"The init and lag constraints must use the same sign for each flow "
                f"variable. Correct pattern: "
                f"subtract(subtract(sum(inv[t0], outflow), inflow), init_inv) = 0 "
                f"→ inv[t0] = init_inv + inflow - outflow. "
                f"Fix: swap the inflow and outflow terms in '{init_cname}'."
            )

    return errors


def _check_sparse_filter_on_equality(ir: dict) -> list[str]:
    """Flag equality constraints that use sparse_filter.

    When a constraint has sense "=" and sparse_filter is set, the compiler
    emits `if key not in param: continue`, skipping the constraint entirely
    for entries absent from the sparse parameter.  For topology constraints
    (routes, arcs) this is correct — variables are also filtered to those
    pairs, so no phantom flow is possible.  But for DEMAND SATISFACTION
    constraints the variables exist for the full domain (all periods, all
    products) regardless of demand sparsity.  For the 87-90% of
    (customer, product, period) triples with no demand entry, the equality
    is skipped → those flow variables are unconstrained → the optimizer
    ships phantom goods that generate revenue without real demand,
    inflating the objective above the true optimum.

    Heuristic: flag any equality constraint that has sparse_filter set AND
    whose LHS expression contains at least one variable node.  The LLM
    should remove sparse_filter and let the compiler use `.get(key, 0.0)`
    for missing entries (producing RHS = 0, which forces flows to zero).
    """
    errors: list[str] = []
    constraints = ir.get("constraints", {})
    for cname, cspec in constraints.items():
        if cspec.get("sense") != "=":
            continue
        if not cspec.get("sparse_filter"):
            continue
        expr = cspec.get("expression", {})
        if not _find_variable_nodes(expr):
            continue
        sf = cspec["sparse_filter"]
        errors.append(
            f"Constraint '{cname}' has sense '=' and sparse_filter='{sf}'. "
            f"The compiler skips this constraint for (key) not in {sf}, leaving "
            f"those flow variables UNCONSTRAINED. If flows appear in the objective "
            f"with positive revenue, the optimizer will generate phantom shipments "
            f"to entries with no {sf} data, inflating the objective. "
            f"Fix: remove sparse_filter (or set null). The compiler automatically "
            f"uses {sf}.get(key, 0.0) for missing entries, giving RHS=0 and "
            f"forcing flows to zero for absent demand/balance entries."
        )
    return errors


def _check_exclude_diagonal_self_loop(ir: dict) -> list[str]:
    """Detect variable references where two indices are identical strings on an exclude_diagonal var.

    When an indexed_sum iterates over a set that is already in the outer domain, the loop
    variable shadows the outer domain variable — e.g. over=["Customers"] inside a
    domain=["Customers","Vehicles"] constraint gives inner var 'c' which shadows outer 'c'.
    The body then uses y["c","c","v"] which is always the diagonal (0 for exclude_diagonal).
    The constraint silently evaluates as if the sum didn't exist → typically infeasible.
    """
    variables = ir.get("variables", {})
    excl_vars: set[str] = {
        name for name, meta in variables.items()
        if meta.get("exclude_diagonal", False)
    }

    errors: list[str] = []
    seen_pairs: set[tuple] = set()

    def _walk(expr: object, context: str) -> None:
        if not isinstance(expr, dict):
            return
        if expr.get("type") == "variable":
            vname = expr.get("name", "")
            indices = expr.get("indices", [])
            if vname in excl_vars:
                for i in range(len(indices)):
                    for j in range(i + 1, len(indices)):
                        if (isinstance(indices[i], str) and isinstance(indices[j], str)
                                and indices[i] == indices[j]):
                            key = (context, vname, indices[i])
                            if key not in seen_pairs:
                                seen_pairs.add(key)
                                errors.append(
                                    f"{context}: variable '{vname}' (exclude_diagonal=true) "
                                    f"is referenced with index '{indices[i]}' in two positions "
                                    f"({indices}) — this is always the excluded diagonal (0). "
                                    f"The inner indexed_sum likely uses over: [\"SetName\"] "
                                    f"where SetName is already the outer domain set, shadowing "
                                    f"the loop variable. Fix: use an alias, e.g. "
                                    f"over: [\"Customers:c2\"] and body indices [\"c2\",\"c\",...]."
                                )
        for key in ("left", "right", "body"):
            _walk(expr.get(key), context)

    for cname, cmeta in ir.get("constraints", {}).items():
        if isinstance(cmeta, dict):
            _walk(cmeta.get("expression"), f"constraint '{cname}'")
    _walk(ir.get("objective", {}).get("expression"), "objective")

    return errors


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def validate_ir_semantics(ir: dict) -> list[str]:
    """Run all semantic checks on an IR dict.  Returns a list of error strings.
    An empty list means no issues were found.
    """
    errors: list[str] = []
    # Inventory modeling checks
    errors.extend(_check_inventory_balance_signs(ir))
    errors.extend(_check_lag_without_init(ir))
    errors.extend(_check_init_lag_sign_consistency(ir))
    errors.extend(_check_period_set_with_hardcoded_size(ir))
    # Expression / linearity checks
    errors.extend(_check_exclude_diagonal_self_loop(ir))
    errors.extend(_check_rhs_variable_nodes(ir))
    errors.extend(_check_variable_times_variable(ir))
    errors.extend(_check_lag_in_objective(ir))
    errors.extend(_check_objective_subtract_from_zero(ir))
    errors.extend(_check_objective_nested_subtract(ir))
    # Schema / cross-reference checks
    errors.extend(_check_alias_in_domain(ir))
    errors.extend(_check_shared_source_without_filter(ir))
    errors.extend(_check_duplicate_domain_index_columns(ir))
    errors.extend(_check_sparse_filter_domain(ir))
    errors.extend(_check_domain_filter_domain(ir))
    errors.extend(_check_sparse_filter_on_equality(ir))
    errors.extend(_check_cost_capacity_missing_default(ir))
    errors.extend(_check_cost_params_not_in_objective(ir))
    return errors
