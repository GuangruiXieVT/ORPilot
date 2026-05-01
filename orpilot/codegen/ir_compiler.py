"""Deterministic IR → Python compiler (PuLP, Pyomo, OR-Tools backends)."""

from __future__ import annotations

import re
from pathlib import Path

from orpilot.codegen.ir_validator import validate_ir_semantics

# Matches indices like "Periods[0]" or "Months[-1]" — these are runtime set member
# accesses and must be emitted literally, not wrapped in repr().
_SET_MEMBER_RE = re.compile(r'^\w+\[-?\d+\]$')


def _fmt_index(idx: str, known: set[str]) -> str:
    """Format a single index for variable/parameter dict access.

    - Loop variables (in *known*) → returned as-is.
    - Set-member syntax ``SetName[N]`` → returned as-is (runtime access).
    - Anything else → wrapped in repr() as a string literal.
    """
    if idx in known:
        return idx
    if _SET_MEMBER_RE.match(idx):
        return idx  # e.g. Periods[0] → first element at runtime
    return repr(idx)


class IRCompiler:
    """Compiles a JSON IR dict into a solver-specific Python solve(data) function."""

    @staticmethod
    def _normalize_ir(ir: dict) -> dict:
        """Strip ':alias' suffixes from domain arrays in variables, constraints, and parameters.

        The LLM occasionally writes the 'Set:alias' notation (only valid inside
        indexed_sum.over arrays) into variable/constraint/parameter domain fields.
        This causes a KeyError in _domain_idx_vars because 'Locations:l' is not
        a key in index_map.  Normalise before compiling so the compiler always
        receives plain set names in domain fields.
        """
        def _strip(items: list[str]) -> list[str]:
            return [s.split(":")[0].strip() if ":" in s else s for s in items]

        for meta in ir.get("variables", {}).values():
            meta["domain"] = _strip(meta.get("domain", []))
        for meta in ir.get("constraints", {}).values():
            meta["domain"] = _strip(meta.get("domain", []))
        for meta in ir.get("parameters", {}).values():
            meta["domain"] = _strip(meta.get("domain", []))
        return ir

    def compile(self, ir: dict, solver_framework: str = "pulp") -> str:
        ir = self._normalize_ir(ir)
        errors = validate_ir_semantics(ir)
        if errors:
            raise ValueError(
                "IR semantic validation failed — fix these issues before compiling:\n"
                + "\n".join(f"  • {e}" for e in errors)
            )
        if solver_framework == "pulp":
            return self._compile_pulp(ir)
        if solver_framework == "pyomo":
            return self._compile_pyomo(ir)
        if solver_framework in ("ortools", "or-tools"):
            return self._compile_ortools(ir)
        if solver_framework == "gurobi":
            return self._compile_gurobi(ir)
        if solver_framework in ("cplex", "cplex_cmd"):
            return self._compile_cplex(ir)
        raise NotImplementedError(f"Solver framework '{solver_framework}' is not yet supported.")

    # ------------------------------------------------------------------
    # Shared helpers — set/parameter loading, variable groups
    # ------------------------------------------------------------------

    def _emit_set_loading(self, sets: dict, set_column: dict) -> list[str]:
        """Emit lines that load each set's members from data.

        Supports an optional row filter via ``filter_column`` / ``filter_value`` in the
        set metadata.  This handles the common pattern where a single CSV stores multiple
        sets distinguished by a category column, e.g.:

            set_name             element
            production_sites     PS_001
            distribution_centers DC_002

        Specify in the IR:
            "source": "sets.csv", "column": "element",
            "filter_column": "set_name", "filter_value": "production_sites"
        """
        lines = []
        for set_name, meta in sets.items():
            source = meta.get("source")
            column = meta.get("column")
            filter_column = meta.get("filter_column")
            filter_value = meta.get("filter_value")
            table_stem = Path(source).stem if source else None

            if table_stem and column:
                if filter_column and filter_value is not None:
                    row_filter = f" if str(row[{filter_column!r}]) == {str(filter_value)!r}"
                else:
                    row_filter = ""
                lines.append(
                    f"    {set_name} = list(dict.fromkeys("
                    f"str(row[{column!r}]) for row in data[{table_stem!r}]{row_filter}))"
                )
            elif table_stem:
                lines.append(
                    f"    {set_name} = list(dict.fromkeys("
                    f"str(next(iter(row.values()))) for row in data[{table_stem!r}]))"
                )
            else:
                size = meta.get("size")
                size_source = meta.get("size_source")
                size_column = meta.get("size_column")
                if size_source and size_column:
                    size_stem = Path(size_source).stem
                    lines.append(
                        f"    {set_name} = list(range(int(float("
                        f"data[{size_stem!r}][0][{size_column!r}]))))"
                    )
                elif size is not None:
                    lines.append(f"    {set_name} = list(range({size}))")
                else:
                    lines.append(
                        f"    {set_name} = []  # TODO: no source specified for set {set_name!r}"
                    )
        return lines

    @staticmethod
    def _val_assign(lhs: str, val_expr: str, cast: str, missing_default: str) -> list[str]:
        """Return line(s) that assign the loaded value to *lhs*.

        Uses a None-check rather than `or 0` / `or float('inf')` so that an actual
        zero in the CSV is never silently converted to infinity.
        """
        if not cast:
            return [f"        {lhs} = {val_expr}"]
        if missing_default == "inf":
            return [
                f"        _v = {val_expr}",
                f"        {lhs} = float(_v) if _v is not None else float('inf')",
            ]
        # missing_default == "zero" (default)
        return [f"        {lhs} = float({val_expr}) if {val_expr} is not None else 0.0"]

    def _emit_parameter_loading(
        self, parameters: dict, set_column: dict, index_map: dict
    ) -> list[str]:
        """Emit lines that load each parameter as a dict keyed by set indices."""
        lines = []
        for param_name, meta in parameters.items():
            domain = meta.get("domain", [])
            source = meta.get("source")
            table_stem = Path(source).stem if source else None

            if not table_stem:
                lines.append(
                    f"    {param_name} = {{}}  # TODO: no source for parameter {param_name!r}"
                )
                continue

            # Per-index key columns: use explicit index_columns if provided, else fall back to
            # the set's index_symbol (never the set's "column" field, which is the column name
            # in sets.csv and is irrelevant to the parameter CSV's own key columns).
            index_columns = meta.get("index_columns")
            if index_columns:
                col_names = list(index_columns)
            else:
                col_names = [index_map.get(s, s.lower()) for s in domain]
            # Column that holds the parameter's value in the CSV.
            # Use the explicit "column" from the IR if present; fall back to param_name.
            value_col = meta.get("column") or param_name

            is_str = meta.get("type") == "string"
            cast = "" if is_str else "float"
            missing_default = meta.get("missing_default", "zero")

            # Scalar parameter (no domain): read directly from the first CSV row.
            if not domain:
                val_expr = f"data[{table_stem!r}][0].get({value_col!r})"
                if cast:
                    if missing_default == "inf":
                        lines.append(f"    _sv = {val_expr}")
                        lines.append(f"    {param_name} = float(_sv) if _sv is not None else float('inf')")
                    else:
                        lines.append(f"    _sv = {val_expr}")
                        lines.append(f"    {param_name} = float(_sv) if _sv is not None else 0.0")
                else:
                    lines.append(f"    {param_name} = {val_expr}")
                continue

            lines.append(f"    {param_name} = {{}}")
            lines.append(f"    for _row in data[{table_stem!r}]:")

            val_expr = f"_row.get({value_col!r})"
            if len(domain) == 1:
                c0 = col_names[0]
                lines.append(
                    f"        _key = str(_row.get({c0!r}) or next("
                    f"v for k, v in _row.items() if k != {value_col!r}))"
                )
                lines += self._val_assign(f"{param_name}[_key]", val_expr, cast, missing_default)
            elif len(domain) == 2:
                c0, c1 = col_names[0], col_names[1]
                lines.append(
                    f"        _k1 = str(_row.get({c0!r}) or _row.get({domain[0].lower()!r}))"
                )
                lines.append(
                    f"        _k2 = str(_row.get({c1!r}) or _row.get({domain[1].lower()!r}))"
                )
                lines += self._val_assign(f"{param_name}[(_k1, _k2)]", val_expr, cast, missing_default)
            else:
                for k, c in enumerate(col_names):
                    lines.append(
                        f"        _k{k} = str(_row.get({c!r}) or _row.get({domain[k].lower()!r}))"
                    )
                key_tuple = ", ".join(f"_k{k}" for k in range(len(col_names)))
                lines += self._val_assign(f"{param_name}[({key_tuple},)]", val_expr, cast, missing_default)
        return lines

    def _emit_variable_groups(self, variables: dict) -> list[str]:
        """Emit lines that build result['variable_groups'] from result['variables']."""
        lines = []
        for var_name, meta in variables.items():
            dim_labels = [d.lower() for d in meta.get("domain", [])]
            group_name = meta.get("label") or var_name
            lines.extend([
                f"    _grp_{var_name} = {{",
                f"        k: v for k, v in result['variables'].items()",
                f"        if k.startswith({var_name!r} + '\\x1f')",
                f"    }}",
                f"    result['variable_groups'].append({{",
                f"        'group_name': {group_name!r},",
                f"        'dimension_labels': {dim_labels!r},",
                f"        'variables': _grp_{var_name},",
                f"    }})",
            ])
        return lines

    # ------------------------------------------------------------------
    # Expression tree walkers
    # ------------------------------------------------------------------

    def _var_ref(
        self,
        name: str,
        indices: list[str],
        domain: list[str],
        known_symbols: set[str] | None = None,
        use_get: bool = False,
    ) -> str:
        """Return the dict-style variable reference used by PuLP and OR-Tools.

        Any index that is not a recognised loop-variable symbol is treated as a
        string literal and emitted with ``repr()`` (e.g. the hard-coded depot
        label ``"depot"`` in depot_start_end constraints).

        When *use_get* is True, emits ``name.get(key, 0)`` instead of
        ``name[key]``, so that variables with ``exclude_diagonal`` never raise
        a KeyError for self-loop indices.
        """
        if not indices or not domain:
            return name

        def _fmt(idx: str) -> str:
            return _fmt_index(idx, known_symbols or set())

        if len(domain) == 1:
            key = _fmt(indices[0])
            return f"{name}.get({key}, 0)" if use_get else f"{name}[{key}]"
        idx_parts = ", ".join(_fmt(i) for i in indices[: len(domain)])
        key = f"({idx_parts})"
        return f"{name}.get({key}, 0)" if use_get else f"{name}[{key}]"

    @staticmethod
    def _domain_idx_vars(domain: list[str], index_map: dict[str, str]) -> list[str]:
        """Return a unique loop-variable name for each set in *domain*.

        When the same set appears more than once, a 1-based numeric suffix is
        appended so the generated Python code has distinct loop variables:
        e.g. ``["Locations", "Locations", "Trips"]`` → ``["l1", "l2", "t"]``.
        Single-occurrence sets keep their plain symbol (``"t"``).
        """
        counts: dict[str, int] = {}
        for s in domain:
            counts[s] = counts.get(s, 0) + 1
        seen: dict[str, int] = {}
        result: list[str] = []
        for s in domain:
            base = index_map[s]
            seen[s] = seen.get(s, 0) + 1
            result.append(f"{base}{seen[s]}" if counts[s] > 1 else base)
        return result

    @staticmethod
    def _excl_diag_pair(domain: list[str], idx_vars: list[str]) -> tuple[str, str]:
        """Return the two index-variable names for the first duplicate set pair in domain.

        For domain ["Vehicles","Locations","Locations"] with idx_vars ["v","i1","i2"]
        returns ("i1","i2") — not ("v","i1").  Falls back to last two if no duplicate.
        """
        seen: dict[str, int] = {}
        for i, s in enumerate(domain):
            if s in seen:
                return idx_vars[seen[s]], idx_vars[i]
            seen[s] = i
        return idx_vars[-2], idx_vars[-1]

    @classmethod
    def _excl_diag_guard(cls, domain: list[str], idx_vars: list[str]) -> str:
        """Return ' if idxA != idxB' for the first duplicate set pair in domain.

        Used in variable creation list comprehensions when exclude_diagonal=True on a
        3+ dimensional domain (e.g. ["Vehicles","Locations","Locations"]).  The duplicate
        pair may not be at positions 0 and 1, so we search for it explicitly.
        """
        a, b = cls._excl_diag_pair(domain, idx_vars)
        return f" if {a} != {b}"

    @staticmethod
    def _constraint_diagonal_guard(domain: list[str], idx_vars: list[str]) -> str | None:
        """Return an 'idxA == idxB' guard expression if the same set appears twice in domain.

        Used to skip trivial self-pair iterations in constraint loops (e.g. MTZ where
        c1==c2 produces 0 <= N-1, wasting solver time on junk constraints).
        Returns the first duplicate pair found, or None if all sets are distinct.
        """
        seen: dict[str, int] = {}
        for i, s in enumerate(domain):
            if s in seen:
                return f"{idx_vars[seen[s]]} == {idx_vars[i]}"
            seen[s] = i
        return None

    @staticmethod
    def _sparse_filter_guard(
        sparse_filter: str | None,
        parameters: dict,
        index_map: dict[str, str],
        constraint_domain: list[str],
    ) -> tuple[str | None, str | None]:
        """Return (precompute_stmt, guard_expr) for sparse constraint filtering.

        Full coverage (every dim of filter param is in constraint domain):
          precompute_stmt = None
          guard_expr      = "(key) not in param"

        Partial coverage (only some dims of filter param are in constraint domain):
          precompute_stmt = "    _valid_<vars>_<param> = {k[pos] for k in param}"
          guard_expr      = "(key) not in _valid_<vars>_<param>"
          The precompute line is emitted BEFORE the constraint loops so the set is
          built once, keeping the guard O(1) at runtime.

        No usable overlap or unknown param:
          (None, None)
        """
        if not sparse_filter or sparse_filter not in parameters:
            return None, None
        filter_domain = parameters[sparse_filter].get("domain", [])
        if not filter_domain:
            return None, None
        constraint_sets = set(constraint_domain)

        # Collect filter dims that are also loop variables in this constraint
        in_scope: list[tuple[str, str]] = [
            (s, index_map[s])
            for s in filter_domain
            if s in constraint_sets and s in index_map
        ]
        if not in_scope:
            return None, None

        total = len(filter_domain)
        if len(in_scope) == total:
            # Full coverage — emit a direct key lookup
            key_vars = [iv for (_, iv) in in_scope]
            key_expr = key_vars[0] if len(key_vars) == 1 else "(" + ", ".join(key_vars) + ")"
            return None, f"{key_expr} not in {sparse_filter}"
        else:
            # Partial coverage — project onto in-scope dims and precompute a set
            positions = [filter_domain.index(s) for (s, _) in in_scope]
            in_scope_vars = [iv for (_, iv) in in_scope]
            set_var = f"_valid_{'_'.join(in_scope_vars)}_{sparse_filter}"
            if total == 1:
                key_inner = "k"
            elif len(positions) == 1:
                key_inner = f"k[{positions[0]}]"
            else:
                key_inner = "(" + ", ".join(f"k[{p}]" for p in positions) + ")"
            precompute = f"    {set_var} = {{{key_inner} for k in {sparse_filter}}}"
            guard_key = in_scope_vars[0] if len(in_scope_vars) == 1 else "(" + ", ".join(in_scope_vars) + ")"
            return precompute, f"{guard_key} not in {set_var}"

    @staticmethod
    def _domain_filter_cond(
        var_meta: dict,
        parameters: dict,
        idx_vars: list[str],
        var_domain: list[str],
    ) -> str | None:
        """Return an 'if key in param' condition for sparse variable creation, or None.

        When a variable should only exist for combinations where a corresponding
        parameter entry exists (e.g. shipment2 only for valid DC→customer routes),
        set ``domain_filter`` on the variable to the parameter name.

        The filter key is built from the positions in *var_domain* that match the
        parameter's domain — in the order the parameter's domain lists them.
        Returns None if the parameter is unknown or has no overlap with var_domain.
        """
        domain_filter = var_meta.get("domain_filter")
        if not domain_filter or domain_filter not in parameters:
            return None
        filter_domain = parameters[domain_filter].get("domain", [])
        filter_key_vars: list[str] = []
        for s in filter_domain:
            try:
                pos = var_domain.index(s)
                filter_key_vars.append(idx_vars[pos])
            except ValueError:
                pass  # set not in variable's domain — skip
        if not filter_key_vars:
            return None
        key = filter_key_vars[0] if len(filter_key_vars) == 1 else "(" + ", ".join(filter_key_vars) + ")"
        # Returns just the boolean expression, without "if" prefix.
        # Call sites format as: "if {cond}" (creation) or "if not {cond}: continue" (extraction).
        return f"{key} in {domain_filter}"

    @staticmethod
    def _collect_sparse_filters(
        body: dict,
        alias_vars: set[str],
        extra_known: set[str] | None,
        variables: dict,
        parameters: dict,
    ) -> list[str]:
        """Scan *body* for multi-dim parameters/variables-with-domain-filter and return
        a list of ``if`` conditions like ``"(c, p, t) in demand"`` suitable for appending
        to a quicksum/lpSum comprehension to skip zero-valued terms.

        A condition is included when:
        - All key variables are in scope (alias_vars ∪ extra_known), AND
        - At least one key variable is a local loop var (alias_vars), so the filter
          meaningfully changes per iteration of this indexed_sum.
        Duplicate conditions (same parameter) are suppressed via *seen*.
        """
        all_known = alias_vars | (extra_known or set())
        conditions: list[str] = []
        seen: set[str] = set()

        def _scan(node: dict) -> None:
            if not isinstance(node, dict):
                return
            ntype = node.get("type")
            op = node.get("operation")

            if ntype == "parameter":
                name = node["name"]
                if name not in seen:
                    pmeta = parameters.get(name, {})
                    domain = pmeta.get("domain", [])
                    indices = node.get("indices", [])
                    if len(domain) >= 2:
                        key_vars = indices[: len(domain)]
                        if (
                            key_vars
                            and all(iv in all_known for iv in key_vars)
                            and any(iv in alias_vars for iv in key_vars)
                        ):
                            key = "(" + ", ".join(key_vars) + ")" if len(key_vars) > 1 else key_vars[0]
                            conditions.append(f"{key} in {name}")
                            seen.add(name)
                return  # leaf — no children

            if ntype == "variable":
                name = node["name"]
                vmeta = variables.get(name, {})
                df = vmeta.get("domain_filter")
                if df and df not in seen:
                    pmeta = parameters.get(df, {})
                    filter_domain = pmeta.get("domain", [])
                    var_domain = vmeta.get("domain", [])
                    indices = node.get("indices", [])
                    key_vars: list[str] = []
                    for s in filter_domain:
                        try:
                            pos = var_domain.index(s)
                            if pos < len(indices):
                                key_vars.append(indices[pos])
                        except ValueError:
                            pass
                    if (
                        key_vars
                        and all(iv in all_known for iv in key_vars)
                        and any(iv in alias_vars for iv in key_vars)
                    ):
                        key = "(" + ", ".join(key_vars) + ")" if len(key_vars) > 1 else key_vars[0]
                        conditions.append(f"{key} in {df}")
                        seen.add(df)
                return  # leaf — no children

            for child_key in ("left", "right", "body"):
                if child_key in node:
                    _scan(node[child_key])

        _scan(body)
        return conditions

    @staticmethod
    def _collect_lag_symbols(
        node: dict,
        variables: dict,
        parameters: dict,
        sets: dict,
    ) -> dict[str, int]:
        """Scan an expression tree and return {index_symbol: lag_value} for every
        ordered-set index that appears with a non-zero 'lag' field on a variable or
        parameter node.  When a symbol appears with multiple lag values, the one with
        the largest absolute value (most boundary-sensitive) is kept.
        """
        result: dict[str, int] = {}
        lag = node.get("lag", 0)
        ntype = node.get("type")
        if lag and ntype in ("variable", "parameter"):
            if ntype == "variable":
                var_domain = variables.get(node.get("name", ""), {}).get("domain", [])
            else:
                var_domain = parameters.get(node.get("name", ""), {}).get("domain", [])
            for idx, set_name in zip(node.get("indices", [])[: len(var_domain)], var_domain):
                set_meta = sets.get(set_name)
                if set_meta is None:
                    continue
                ordered = (
                    set_meta.get("ordered", False)
                    if isinstance(set_meta, dict)
                    else getattr(set_meta, "ordered", False)
                )
                if ordered:
                    if idx not in result or abs(lag) > abs(result[idx]):
                        result[idx] = lag
        for key in ("left", "right", "body"):
            child = node.get(key)
            if child:
                child_lags = IRCompiler._collect_lag_symbols(child, variables, parameters, sets)
                for sym, lag_val in child_lags.items():
                    if sym not in result or abs(lag_val) > abs(result[sym]):
                        result[sym] = lag_val
        return result

    @staticmethod
    def _build_lag_context(lag_syms: dict[str, int], sets: dict) -> dict[str, tuple[str, str]]:
        """Build a lag_context dict from lag_syms.

        Returns {index_symbol: (set_name, enumerate_var_name)} so the expression
        emitters know which Python variable to use for the enumerate position index.
        """
        result: dict[str, tuple[str, str]] = {}
        for sym in lag_syms:
            for set_name, set_meta in sets.items():
                s_sym = (
                    set_meta.get("index_symbol")
                    if isinstance(set_meta, dict)
                    else getattr(set_meta, "index_symbol", None)
                )
                if s_sym == sym:
                    result[sym] = (set_name, f"_idx_{sym}")
                    break
        return result

    def _emit_lagged_ref(
        self,
        name: str,
        indices: list[str],
        domain: list[str],
        lag: int,
        lag_context: dict[str, tuple[str, str]],
        known: set[str],
        use_get: bool,
    ) -> str:
        """Emit a variable or parameter reference with temporal lag applied.

        For each index that maps to an ordered set in lag_context, the reference is
        shifted: ``var[t]`` with lag=-1 becomes ``var[Months[_idx_t - 1]]``.
        Non-lagged indices are emitted as plain symbols (or quoted literals if not in known).
        """
        modified: list[str] = []
        for idx, _set_name in zip(indices[: len(domain)], domain):
            if idx in lag_context:
                ctx_set, enum_var = lag_context[idx]
                if lag >= 0:
                    modified.append(f"{ctx_set}[{enum_var} + {lag}]")
                else:
                    modified.append(f"{ctx_set}[{enum_var} - {-lag}]")
            else:
                modified.append(_fmt_index(idx, known))
        if len(domain) == 1:
            key = modified[0]
        else:
            key = f"({', '.join(modified)})"
        if use_get:
            return f"{name}.get({key}, 0)"
        return f"{name}[{key}]"

    @staticmethod
    def _index_key(indices: list[str], domain: list[str], known: set[str]) -> str:
        """Return the dict-key expression for a variable reference (without the variable name).

        Used by the OR-Tools emitter to call ``var.get(key)`` separately from the
        variable name so the result can be guarded with ``if _v is not None``.
        """
        if len(domain) == 1:
            return _fmt_index(indices[0], known)
        idx_parts = ", ".join(_fmt_index(i, known) for i in indices[: len(domain)])
        return f"({idx_parts})"

    @staticmethod
    def _parse_over_item(item: str, index_map: dict[str, str]) -> tuple[str, str]:
        """Parse a single ``over`` entry, returning ``(set_name, loop_var)``.

        Supports the extended ``"SetName:alias"`` syntax to allow the same set
        to appear twice with different loop-variable names (e.g. when summing
        over Locations × Locations with distinct symbols ``l1`` and ``l2``).
        """
        if ":" in item:
            set_name, alias = item.split(":", 1)
            return set_name.strip(), alias.strip()
        return item, index_map[item]

    def _flatten_obj_terms(self, node: dict, sign: int = 1) -> list[tuple[int, dict]]:
        """Flatten a tree of sum/subtract nodes into [(sign, leaf_node)] pairs.

        Leaf nodes are anything that is not a bare sum/subtract:
        indexed_sum, variable, parameter, constant, multiply, set_size.
        """
        op = node.get("operation")
        if op == "subtract":
            return (
                self._flatten_obj_terms(node["left"], sign)
                + self._flatten_obj_terms(node["right"], -sign)
            )
        if op == "sum":
            return (
                self._flatten_obj_terms(node["left"], sign)
                + self._flatten_obj_terms(node["right"], sign)
            )
        return [(sign, node)]

    def _emit_expr(
        self,
        node: dict,
        index_map: dict[str, str],
        variables: dict,
        parameters: dict,
        extra_known: set[str] | None = None,
        lag_context: dict[str, tuple[str, str]] | None = None,
        _sum_fn: str = "pulp.lpSum",
    ) -> str:
        """Emit a Python expression string (PuLP lpSum / plain Python for OR-Tools RHS).

        *_sum_fn* controls which aggregation function is emitted for ``indexed_sum``
        nodes.  Defaults to ``"pulp.lpSum"``; use ``"gp.quicksum"`` for the native
        Gurobi backend and ``"mdl.sum"`` for the native docplex (CPLEX) backend.
        """
        node_type = node.get("type")
        operation = node.get("operation")

        if node_type == "constant":
            return str(node["value"])

        known = set(index_map.values())
        if extra_known:
            known |= extra_known

        if node_type == "variable":
            name = node["name"]
            indices = node.get("indices", [])
            vmeta = variables.get(name, {})
            domain = vmeta.get("domain", [])
            use_get = bool(vmeta.get("exclude_diagonal", False)) or bool(vmeta.get("domain_filter"))
            lag = node.get("lag", 0)
            if lag and lag_context:
                return self._emit_lagged_ref(name, indices, domain, lag, lag_context, known, use_get)
            return self._var_ref(name, indices, domain, known, use_get=use_get)

        if node_type == "parameter":
            name = node["name"]
            indices = node.get("indices", [])
            pmeta = parameters.get(name, {})
            domain = pmeta.get("domain", [])
            if not indices or not domain:
                return name
            lag = node.get("lag", 0)
            missing_default = "float('inf')" if pmeta.get("missing_default") == "inf" else "0.0"
            if lag and lag_context:
                # Multi-dim parameters always use .get() so lagged refs are safe too.
                use_get = len(domain) >= 2
                return self._emit_lagged_ref(name, indices, domain, lag, lag_context, known, use_get)

            if len(domain) == 1:
                return f"{name}.get({_fmt_index(indices[0], known)}, {missing_default})"
            # Always use .get() for multi-dimensional parameters. This handles both sparse
            # tables (fewer rows than the full Cartesian product) and the diagonal case
            # (same set twice) without requiring any explicit flag in the IR.
            idx_tuple = ", ".join(_fmt_index(i, known) for i in indices[: len(domain)])
            return f"{name}.get(({idx_tuple}), {missing_default})"

        if node_type == "set_size":
            return f"len({node['set']})"

        if operation in ("sum", "subtract", "multiply"):
            left = self._emit_expr(node["left"], index_map, variables, parameters, extra_known, lag_context, _sum_fn)
            right = self._emit_expr(node["right"], index_map, variables, parameters, extra_known, lag_context, _sum_fn)
            if operation == "multiply":
                return f"{left} * {right}"
            op = "+" if operation == "sum" else "-"
            return f"({left} {op} {right})"

        if operation == "indexed_sum":
            over = node.get("over", [])
            alias_vars: set[str] = set()
            iter_parts: list[str] = []
            for item in over:
                set_name, loop_var = self._parse_over_item(item, index_map)
                iter_parts.append(f"for {loop_var} in {set_name}")
                alias_vars.add(loop_var)
            new_extra = (extra_known or set()) | alias_vars
            body = self._emit_expr(node["body"], index_map, variables, parameters, new_extra, lag_context, _sum_fn)
            filter_conds = self._collect_sparse_filters(node["body"], alias_vars, extra_known, variables, parameters)
            cond_str = "".join(f" if {c}" for c in filter_conds)
            iterators = " ".join(iter_parts)
            return f"{_sum_fn}({body} {iterators}{cond_str})"

        return "0"

    def _emit_expr_gurobi(
        self,
        node: dict,
        index_map: dict[str, str],
        variables: dict,
        parameters: dict,
        extra_known: set[str] | None = None,
        lag_context: dict[str, tuple[str, str]] | None = None,
    ) -> str:
        """Emit an expression string for the native gurobipy backend (uses gp.quicksum)."""
        return self._emit_expr(node, index_map, variables, parameters, extra_known, lag_context, "gp.quicksum")

    def _emit_expr_cplex(
        self,
        node: dict,
        index_map: dict[str, str],
        variables: dict,
        parameters: dict,
        extra_known: set[str] | None = None,
        lag_context: dict[str, tuple[str, str]] | None = None,
    ) -> str:
        """Emit an expression string for the native docplex (CPLEX) backend (uses mdl.sum)."""
        return self._emit_expr(node, index_map, variables, parameters, extra_known, lag_context, "mdl.sum")

    def _emit_pyomo_expr(
        self,
        node: dict,
        index_map: dict[str, str],
        variables: dict,
        parameters: dict,
        extra_known: set[str] | None = None,
        lag_context: dict[str, tuple[str, str]] | None = None,
    ) -> str:
        """Emit a Pyomo-compatible Python expression string.

        Differences from _emit_expr:
        - Variables are referenced as ``model.x[i, j]`` (no tuple, comma-separated)
        - indexed_sum uses ``sum(...)`` instead of ``pulp.lpSum(...)``
        """
        node_type = node.get("type")
        operation = node.get("operation")

        if node_type == "constant":
            return str(node["value"])

        known = set(index_map.values())
        if extra_known:
            known |= extra_known

        if node_type == "variable":
            name = node["name"]
            indices = node.get("indices", [])
            vmeta = variables.get(name, {})
            domain = vmeta.get("domain", [])
            use_get_var = bool(vmeta.get("exclude_diagonal", False)) or bool(vmeta.get("domain_filter"))
            lag = node.get("lag", 0)
            if not indices or not domain:
                return f"model.{name}"
            if lag and lag_context:
                # Build lagged reference; Pyomo uses comma-separated indices, not tuples
                modified: list[str] = []
                for idx, _set_name in zip(indices[: len(domain)], domain):
                    if idx in lag_context:
                        ctx_set, enum_var = lag_context[idx]
                        modified.append(
                            f"{ctx_set}[{enum_var} + {lag}]" if lag >= 0
                            else f"{ctx_set}[{enum_var} - {-lag}]"
                        )
                    else:
                        modified.append(_fmt_index(idx, known))
                idx_str = ", ".join(modified)
                return f"model.{name}[{idx_str}]"
            if len(domain) == 1:
                key = _fmt_index(indices[0], known)
                return f"model.{name}.get({key}, 0)" if use_get_var else f"model.{name}[{key}]"
            idx_str = ", ".join(_fmt_index(i, known) for i in indices[: len(domain)])
            # Pyomo Var with a filtered index set uses dict-style access; .get() is safe.
            return f"model.{name}.get(({idx_str}), 0)" if use_get_var else f"model.{name}[{idx_str}]"

        if node_type == "parameter":
            name = node["name"]
            indices = node.get("indices", [])
            pmeta = parameters.get(name, {})
            domain = pmeta.get("domain", [])
            if not indices or not domain:
                return name
            lag = node.get("lag", 0)
            missing_default = "float('inf')" if pmeta.get("missing_default") == "inf" else "0.0"
            if lag and lag_context:
                use_get = len(domain) >= 2
                return self._emit_lagged_ref(name, indices, domain, lag, lag_context, known, use_get)
            if len(domain) == 1:
                return f"{name}.get({_fmt_index(indices[0], known)}, {missing_default})"
            # Always use .get() for multi-dimensional parameters.
            idx_tuple = ", ".join(_fmt_index(i, known) for i in indices[: len(domain)])
            return f"{name}.get(({idx_tuple}), {missing_default})"

        if node_type == "set_size":
            return f"len({node['set']})"

        if operation in ("sum", "subtract", "multiply"):
            left = self._emit_pyomo_expr(node["left"], index_map, variables, parameters, extra_known, lag_context)
            right = self._emit_pyomo_expr(node["right"], index_map, variables, parameters, extra_known, lag_context)
            if operation == "multiply":
                return f"{left} * {right}"
            op = "+" if operation == "sum" else "-"
            return f"({left} {op} {right})"

        if operation == "indexed_sum":
            over = node.get("over", [])
            alias_vars: set[str] = set()
            iter_parts: list[str] = []
            for item in over:
                set_name, loop_var = self._parse_over_item(item, index_map)
                iter_parts.append(f"for {loop_var} in {set_name}")
                alias_vars.add(loop_var)
            new_extra = (extra_known or set()) | alias_vars
            body = self._emit_pyomo_expr(node["body"], index_map, variables, parameters, new_extra, lag_context)
            filter_conds = self._collect_sparse_filters(node["body"], alias_vars, extra_known, variables, parameters)
            cond_str = "".join(f" if {c}" for c in filter_conds)
            iterators = " ".join(iter_parts)
            return f"sum({body} {iterators}{cond_str})"

        return "0"

    def _emit_ortools_coefficients(
        self,
        node: dict,
        target: str,
        index_map: dict[str, str],
        variables: dict,
        parameters: dict,
        lines: list[str],
        indent: int,
        sign: int = 1,
        extra_known: set[str] | None = None,
        lag_context: dict[str, tuple[str, str]] | None = None,
    ) -> None:
        """Append OR-Tools SetCoefficient calls to *lines* for a linear expression node.

        *target* is the Python name of the OR-Tools objective or constraint object.
        *indent* is the current indentation level (1 = 4 spaces = inside solve()).
        *sign* is +1 or -1, accumulated through subtract nodes.
        *extra_known* carries loop-variable aliases from enclosing indexed_sum nodes.
        """
        pad = "    " * indent
        op = node.get("operation")
        ntype = node.get("type")

        if op == "indexed_sum":
            alias_vars: set[str] = set()
            for item in node["over"]:
                set_name, loop_var = self._parse_over_item(item, index_map)
                pad = "    " * indent
                lines.append(f"{pad}for {loop_var} in {set_name}:")
                indent += 1
                alias_vars.add(loop_var)
            # Emit sparse-filter guards to skip zero terms
            filter_conds = self._collect_sparse_filters(node["body"], alias_vars, extra_known, variables, parameters)
            pad = "    " * indent
            for cond in filter_conds:
                lines.append(f"{pad}if not ({cond}): continue")
            new_extra = (extra_known or set()) | alias_vars
            self._emit_ortools_coefficients(
                node["body"], target, index_map, variables, parameters, lines, indent, sign, new_extra, lag_context
            )
            return

        known = set(index_map.values()) | (extra_known or set())

        if ntype == "variable":
            name = node["name"]
            indices = node.get("indices", [])
            domain = variables.get(name, {}).get("domain", [])
            excl = bool(variables.get(name, {}).get("exclude_diagonal", False))
            lag = node.get("lag", 0)
            if lag and lag_context:
                var_ref = self._emit_lagged_ref(name, indices, domain, lag, lag_context, known, excl)
            else:
                var_ref = self._var_ref(name, indices, domain, known)
            coeff = "1.0" if sign == 1 else "-1.0"
            if excl:
                lines.append(f"{pad}_v = {name}.get({self._index_key(indices, domain, known)})")
                lines.append(f"{pad}if _v is not None: {target}.SetCoefficient(_v, {coeff})")
            else:
                lines.append(f"{pad}{target}.SetCoefficient({var_ref}, {coeff})")
            return

        if op == "multiply":
            left, right = node["left"], node["right"]
            # Identify which operand is the variable
            if right.get("type") == "variable":
                coeff_node, var_node = left, right
            else:
                coeff_node, var_node = right, left
            name = var_node["name"]
            indices = var_node.get("indices", [])
            domain = variables.get(name, {}).get("domain", [])
            excl = bool(variables.get(name, {}).get("exclude_diagonal", False))
            lag = var_node.get("lag", 0)
            coeff_str = self._emit_expr(coeff_node, index_map, variables, parameters, extra_known, lag_context)
            if sign == -1:
                coeff_str = f"-({coeff_str})"
            if lag and lag_context:
                var_ref = self._emit_lagged_ref(name, indices, domain, lag, lag_context, known, excl)
            else:
                var_ref = self._var_ref(name, indices, domain, known)
            if excl:
                lines.append(f"{pad}_v = {name}.get({self._index_key(indices, domain, known)})")
                lines.append(f"{pad}if _v is not None: {target}.SetCoefficient(_v, {coeff_str})")
            else:
                lines.append(f"{pad}{target}.SetCoefficient({var_ref}, {coeff_str})")
            return

        if op == "sum":
            self._emit_ortools_coefficients(
                node["left"], target, index_map, variables, parameters, lines, indent, sign, extra_known, lag_context
            )
            self._emit_ortools_coefficients(
                node["right"], target, index_map, variables, parameters, lines, indent, sign, extra_known, lag_context
            )
            return

        if op == "subtract":
            self._emit_ortools_coefficients(
                node["left"], target, index_map, variables, parameters, lines, indent, sign, extra_known, lag_context
            )
            self._emit_ortools_coefficients(
                node["right"], target, index_map, variables, parameters, lines, indent, -sign, extra_known, lag_context
            )
            return

        lines.append(f"{pad}# TODO: unsupported expression node type={ntype!r} op={op!r}")

    # ------------------------------------------------------------------
    # PuLP backend
    # ------------------------------------------------------------------

    def _compile_pulp(self, ir: dict) -> str:
        sets = ir.get("sets", {})
        parameters = ir.get("parameters", {})
        variables = ir.get("variables", {})
        constraints = ir.get("constraints", {})
        objective = ir.get("objective", {})
        sense = ir.get("sense", "minimize")
        problem_class = ir.get("problem_class", "Model")

        index_map: dict[str, str] = {n: m["index_symbol"] for n, m in sets.items()}
        set_column: dict[str, str | None] = {n: m.get("column") for n, m in sets.items()}

        lines: list[str] = [
            "import pulp",
            "",
            "",
            "def solve(data: dict, time_limit: int | None = None, show_solver_log: bool = False) -> dict:",
            "    # --- Load sets ---",
        ]
        lines += self._emit_set_loading(sets, set_column)
        lines.append("")
        lines.append("    # --- Load parameters ---")
        lines += self._emit_parameter_loading(parameters, set_column, index_map)

        lp_sense = "pulp.LpMinimize" if sense == "minimize" else "pulp.LpMaximize"
        lines += [
            "",
            "    # --- Build model ---",
            f"    prob = pulp.LpProblem({problem_class!r}, {lp_sense})",
            "",
            "    # --- Decision variables ---",
        ]

        cat_map = {
            "continuous": "pulp.LpContinuous",
            "integer": "pulp.LpInteger",
            "binary": "pulp.LpBinary",
        }
        for var_name, meta in variables.items():
            domain = meta.get("domain", [])
            cat = cat_map.get(meta.get("type", "continuous"), "pulp.LpContinuous")
            lb = meta.get("lower_bound")
            ub = meta.get("upper_bound")
            lb_str = str(lb) if lb is not None else "0"
            if ub is not None:
                ub_str = str(ub)
            elif meta.get("upper_bound_set"):
                ub_str = f"len({meta['upper_bound_set']})"
            else:
                ub_str = "None"
            excl = meta.get("exclude_diagonal", False)

            if not domain:
                lines.append(
                    f"    {var_name} = pulp.LpVariable({var_name!r}, "
                    f"lowBound={lb_str}, upBound={ub_str}, cat={cat})"
                )
            elif len(domain) == 1:
                idx0 = index_map[domain[0]]
                idx0 = index_map[domain[0]]
                df_cond = self._domain_filter_cond(meta, parameters, [idx0], domain)
                if df_cond:
                    lines.append(
                        f"    {var_name} = pulp.LpVariable.dicts("
                        f"{var_name!r}, [{idx0} for {idx0} in {domain[0]} if {df_cond}], "
                        f"lowBound={lb_str}, upBound={ub_str}, cat={cat})"
                    )
                else:
                    lines.append(
                        f"    {var_name} = pulp.LpVariable.dicts("
                        f"{var_name!r}, {domain[0]}, lowBound={lb_str}, upBound={ub_str}, cat={cat})"
                    )
            elif len(domain) == 2:
                s0, s1 = domain[0], domain[1]
                idx0, idx1 = self._domain_idx_vars(domain, index_map)
                diag_guard = f" if {idx0} != {idx1}" if excl else ""
                df_cond = self._domain_filter_cond(meta, parameters, [idx0, idx1], domain)
                df_guard = f" if {df_cond}" if df_cond else ""
                lines.append(
                    f"    {var_name} = pulp.LpVariable.dicts("
                    f"{var_name!r}, [({idx0}, {idx1}) for {idx0} in {s0} for {idx1} in {s1}{diag_guard}{df_guard}], "
                    f"lowBound={lb_str}, upBound={ub_str}, cat={cat})"
                )
            else:
                idx_vars = self._domain_idx_vars(domain, index_map)
                iterators = " ".join(
                    f"for {iv} in {s}" for iv, s in zip(idx_vars, domain)
                )
                idx_tuple = "(" + ", ".join(idx_vars) + ")"
                diag_guard = self._excl_diag_guard(domain, idx_vars) if excl else ""
                df_cond = self._domain_filter_cond(meta, parameters, idx_vars, domain)
                df_guard = f" if {df_cond}" if df_cond else ""
                lines.append(
                    f"    {var_name} = pulp.LpVariable.dicts("
                    f"{var_name!r}, [{idx_tuple} {iterators}{diag_guard}{df_guard}], "
                    f"lowBound={lb_str}, upBound={ub_str}, cat={cat})"
                )

        lines.append("")
        lines.append("    # --- Objective ---")
        terms = self._flatten_obj_terms(objective["expression"])
        lines.append("    prob += (")
        for i, (sign, node) in enumerate(terms):
            expr = self._emit_expr(node, index_map, variables, parameters)
            if i == 0:
                prefix = "        " if sign == 1 else "        -"
            else:
                prefix = "        + " if sign == 1 else "        - "
            lines.append(f"{prefix}{expr}")
        lines.append("    ), 'objective'")

        lines.append("")
        lines.append("    # --- Constraints ---")
        for cname, cmeta in constraints.items():
            domain = cmeta.get("domain", [])
            sense_op = {"<=": "<=", ">=": ">=", "=": "=="}.get(cmeta.get("sense", "<="), "<=")
            sparse_filter = cmeta.get("sparse_filter")
            domain_loop_vars = set(self._domain_idx_vars(domain, index_map)) if domain else set()
            lag_syms: dict[str, int] = {}
            lag_syms.update(self._collect_lag_symbols(cmeta["expression"], variables, parameters, sets))
            lag_syms.update(self._collect_lag_symbols(cmeta["rhs"], variables, parameters, sets))
            lag_ctx = self._build_lag_context(lag_syms, sets) if lag_syms else {}
            lhs = self._emit_expr(cmeta["expression"], index_map, variables, parameters, extra_known=domain_loop_vars, lag_context=lag_ctx or None)
            rhs = self._emit_expr(cmeta["rhs"], index_map, variables, parameters, extra_known=domain_loop_vars, lag_context=lag_ctx or None)
            sf_precompute, sf_guard = self._sparse_filter_guard(sparse_filter, parameters, index_map, domain)
            if sf_precompute:
                lines.append(sf_precompute)

            if not domain:
                lines.append(f"    prob += {lhs} {sense_op} {rhs}, {cname!r}")
            elif len(domain) == 1:
                idx0 = index_map[domain[0]]
                if idx0 in lag_ctx:
                    _, _ev0 = lag_ctx[idx0]
                    _lv0 = lag_syms[idx0]
                    lines.append(f"    for {_ev0}, {idx0} in enumerate({domain[0]}):")
                    if _lv0 < 0:
                        lines.append(f"        if {_ev0} < {-_lv0}: continue")
                    else:
                        lines.append(f"        if {_ev0} + {_lv0} >= len({domain[0]}): continue")
                else:
                    lines.append(f"    for {idx0} in {domain[0]}:")
                if sf_guard:
                    lines.append(f"        if {sf_guard}: continue")
                lines.append(f"        prob += {lhs} {sense_op} {rhs}, f\"{cname}_{{{idx0}}}\"")
            elif len(domain) == 2:
                idx0, idx1 = self._domain_idx_vars(domain, index_map)
                if idx0 in lag_ctx:
                    _, _ev0 = lag_ctx[idx0]
                    _lv0 = lag_syms[idx0]
                    lines.append(f"    for {_ev0}, {idx0} in enumerate({domain[0]}):")
                    if _lv0 < 0:
                        lines.append(f"        if {_ev0} < {-_lv0}: continue")
                    else:
                        lines.append(f"        if {_ev0} + {_lv0} >= len({domain[0]}): continue")
                else:
                    lines.append(f"    for {idx0} in {domain[0]}:")
                if idx1 in lag_ctx:
                    _, _ev1 = lag_ctx[idx1]
                    _lv1 = lag_syms[idx1]
                    lines.append(f"        for {_ev1}, {idx1} in enumerate({domain[1]}):")
                    if _lv1 < 0:
                        lines.append(f"            if {_ev1} < {-_lv1}: continue")
                    else:
                        lines.append(f"            if {_ev1} + {_lv1} >= len({domain[1]}): continue")
                else:
                    lines.append(f"        for {idx1} in {domain[1]}:")
                guard = self._constraint_diagonal_guard(domain, [idx0, idx1])
                if guard:
                    lines.append(f"            if {guard}: continue")
                if sf_guard:
                    lines.append(f"            if {sf_guard}: continue")
                lines.append(
                    f"            prob += {lhs} {sense_op} {rhs}, "
                    f"f\"{cname}_{{{idx0}}}_{{{idx1}}}\""
                )
            else:
                idx_vars = self._domain_idx_vars(domain, index_map)
                for k, (iv, s) in enumerate(zip(idx_vars, domain)):
                    if iv in lag_ctx:
                        _, _ev = lag_ctx[iv]
                        _lv = lag_syms[iv]
                        lines.append(f"    {'    ' * k}for {_ev}, {iv} in enumerate({s}):")
                        if _lv < 0:
                            lines.append(f"    {'    ' * (k + 1)}if {_ev} < {-_lv}: continue")
                        else:
                            lines.append(f"    {'    ' * (k + 1)}if {_ev} + {_lv} >= len({s}): continue")
                    else:
                        lines.append(f"    {'    ' * k}for {iv} in {s}:")
                name_parts = "\\x1f".join(f"{{{iv}}}" for iv in idx_vars)
                inner = "    " * (len(domain) + 1)
                guard = self._constraint_diagonal_guard(domain, idx_vars)
                if guard:
                    lines.append(f"{inner}if {guard}: continue")
                if sf_guard:
                    lines.append(f"{inner}if {sf_guard}: continue")
                lines.append(
                    f"{inner}prob += {lhs} {sense_op} {rhs}, f\"{cname}_{name_parts}\""
                )

        lines += [
            "",
            "    # --- Solve ---",
            "    prob.writeLP('model.lp')",
            "    _cbc_kwargs = {'msg': 1 if show_solver_log else 0}",
            "    if time_limit is not None:",
            "        _cbc_kwargs['timeLimit'] = time_limit",
            "    prob.solve(pulp.PULP_CBC_CMD(**_cbc_kwargs))",
            "",
            "    _sol_status_map = {",
            "        pulp.constants.LpSolutionOptimal:         'optimal',",
            "        pulp.constants.LpSolutionIntegerFeasible: 'feasible',",
            "        pulp.constants.LpSolutionInfeasible:      'infeasible',",
            "        pulp.constants.LpSolutionUnbounded:       'unbounded',",
            "    }",
            "    result = {",
            "        'status': _sol_status_map.get(prob.sol_status, 'error'),",
            "        'objective_value': None,",
            "        'variables': {},",
            "        'variable_groups': [],",
            "    }",
            "    if prob.sol_status in (pulp.constants.LpSolutionOptimal, pulp.constants.LpSolutionIntegerFeasible):",
            "        result['objective_value'] = pulp.value(prob.objective)",
        ]

        for var_name, meta in variables.items():
            domain = meta.get("domain", [])
            excl = meta.get("exclude_diagonal", False)
            lines.append(f"    # extract {var_name}")
            if not domain:
                lines.append(
                    f"    result['variables'][{var_name!r}] = pulp.value({var_name})"
                )
            elif len(domain) == 1:
                idx0 = index_map[domain[0]]
                lines.append(f"    for {idx0} in {domain[0]}:")
                lines.append(
                    f"        result['variables'][f\"{var_name}\\x1f{{{idx0}}}\"] = "
                    f"pulp.value({var_name}[{idx0}])"
                )
            elif len(domain) == 2:
                idx0, idx1 = self._domain_idx_vars(domain, index_map)
                lines.append(f"    for {idx0} in {domain[0]}:")
                lines.append(f"        for {idx1} in {domain[1]}:")
                if excl:
                    lines.append(f"            if {idx0} == {idx1}: continue")
                lines.append(
                    f"            result['variables'][f\"{var_name}\\x1f{{{idx0}}}\\x1f{{{idx1}}}\"] = "
                    f"pulp.value({var_name}[({idx0}, {idx1})])"
                )
            else:
                idx_vars = self._domain_idx_vars(domain, index_map)
                for k, (iv, s) in enumerate(zip(idx_vars, domain)):
                    lines.append(f"    {'    ' * k}for {iv} in {s}:")
                inner = "    " * (len(domain) + 1)
                if excl and len(idx_vars) >= 2:
                    _da, _db = self._excl_diag_pair(domain, idx_vars)
                    lines.append(f"{inner}if {_da} == {_db}: continue")
                idx_tuple = "(" + ", ".join(idx_vars) + ")"
                name_parts = "\\x1f".join(f"{{{iv}}}" for iv in idx_vars)
                lines.append(
                    f"{inner}result['variables'][f\"{var_name}\\x1f{name_parts}\"] = "
                    f"pulp.value({var_name}[{idx_tuple}])"
                )

        lines += self._emit_variable_groups(variables)
        lines.append("    return result")
        return "\n".join(lines) + "\n"

    # ------------------------------------------------------------------
    # Gurobi backend (native gurobipy)
    # ------------------------------------------------------------------

    def _compile_gurobi(self, ir: dict) -> str:
        """Compile IR to native gurobipy code (no PuLP dependency)."""
        sets = ir.get("sets", {})
        parameters = ir.get("parameters", {})
        variables = ir.get("variables", {})
        constraints = ir.get("constraints", {})
        objective = ir.get("objective", {})
        sense = ir.get("sense", "minimize")
        problem_class = ir.get("problem_class", "Model")

        index_map: dict[str, str] = {n: m["index_symbol"] for n, m in sets.items()}
        set_column: dict[str, str | None] = {n: m.get("column") for n, m in sets.items()}

        lines: list[str] = [
            "import gurobipy as gp",
            "from gurobipy import GRB",
            "",
            "",
            "def solve(data: dict, time_limit: int | None = None, show_solver_log: bool = False) -> dict:",
            "    # --- Load sets ---",
        ]
        lines += self._emit_set_loading(sets, set_column)
        lines.append("")
        lines.append("    # --- Load parameters ---")
        lines += self._emit_parameter_loading(parameters, set_column, index_map)

        lines += [
            "",
            "    # --- Build model ---",
            f"    m = gp.Model({problem_class!r})",
            "    m.setParam('OutputFlag', 1 if show_solver_log else 0)",
            "    if time_limit is not None:",
            "        m.setParam('TimeLimit', time_limit)",
            "",
            "    # --- Decision variables ---",
        ]

        vtype_map = {
            "continuous": "GRB.CONTINUOUS",
            "integer": "GRB.INTEGER",
            "binary": "GRB.BINARY",
        }

        for var_name, meta in variables.items():
            domain = meta.get("domain", [])
            vtype = vtype_map.get(meta.get("type", "continuous"), "GRB.CONTINUOUS")
            lb = meta.get("lower_bound")
            ub = meta.get("upper_bound")
            lb_str = str(float(lb)) if lb is not None else "0.0"
            if ub is not None:
                ub_str = str(float(ub))
            elif meta.get("upper_bound_set"):
                ub_str = f"len({meta['upper_bound_set']})"
            else:
                ub_str = "GRB.INFINITY"
            excl = meta.get("exclude_diagonal", False)

            if not domain:
                lines.append(
                    f"    {var_name} = m.addVar(lb={lb_str}, ub={ub_str}, "
                    f"vtype={vtype}, name={var_name!r})"
                )
            elif len(domain) == 1:
                idx0 = index_map[domain[0]]
                df_cond = self._domain_filter_cond(meta, parameters, [idx0], domain)
                if df_cond:
                    lines.append(
                        f"    {var_name} = m.addVars("
                        f"[{idx0} for {idx0} in {domain[0]} if {df_cond}], "
                        f"lb={lb_str}, ub={ub_str}, vtype={vtype}, name={var_name!r})"
                    )
                else:
                    lines.append(
                        f"    {var_name} = m.addVars({domain[0]}, lb={lb_str}, ub={ub_str}, "
                        f"vtype={vtype}, name={var_name!r})"
                    )
            elif len(domain) == 2:
                idx0, idx1 = self._domain_idx_vars(domain, index_map)
                s0, s1 = domain[0], domain[1]
                diag_guard = f" if {idx0} != {idx1}" if excl else ""
                df_cond = self._domain_filter_cond(meta, parameters, [idx0, idx1], domain)
                df_guard = f" if {df_cond}" if df_cond else ""
                lines.append(
                    f"    {var_name} = m.addVars("
                    f"[({idx0}, {idx1}) for {idx0} in {s0} for {idx1} in {s1}{diag_guard}{df_guard}], "
                    f"lb={lb_str}, ub={ub_str}, vtype={vtype}, name={var_name!r})"
                )
            else:
                idx_vars = self._domain_idx_vars(domain, index_map)
                iterators = " ".join(f"for {iv} in {s}" for iv, s in zip(idx_vars, domain))
                idx_tuple = "(" + ", ".join(idx_vars) + ")"
                diag_guard = self._excl_diag_guard(domain, idx_vars) if excl else ""
                df_cond = self._domain_filter_cond(meta, parameters, idx_vars, domain)
                df_guard = f" if {df_cond}" if df_cond else ""
                lines.append(
                    f"    {var_name} = m.addVars("
                    f"[{idx_tuple} {iterators}{diag_guard}{df_guard}], "
                    f"lb={lb_str}, ub={ub_str}, vtype={vtype}, name={var_name!r})"
                )

        lines.append("    m.update()")

        lines.append("")
        lines.append("    # --- Objective ---")
        grb_sense = "GRB.MINIMIZE" if sense == "minimize" else "GRB.MAXIMIZE"
        terms = self._flatten_obj_terms(objective["expression"])
        lines.append("    m.setObjective(")
        for i, (sign, node) in enumerate(terms):
            expr = self._emit_expr_gurobi(node, index_map, variables, parameters)
            is_last = i == len(terms) - 1
            suffix = "," if is_last else ""
            if i == 0:
                prefix = "        " if sign == 1 else "        -"
            else:
                prefix = "        + " if sign == 1 else "        - "
            lines.append(f"{prefix}{expr}{suffix}")
        lines.append(f"    {grb_sense})")

        lines.append("")
        lines.append("    # --- Constraints ---")
        for cname, cmeta in constraints.items():
            domain = cmeta.get("domain", [])
            sense_op = {"<=": "<=", ">=": ">=", "=": "=="}.get(cmeta.get("sense", "<="), "<=")
            sparse_filter = cmeta.get("sparse_filter")
            domain_loop_vars = set(self._domain_idx_vars(domain, index_map)) if domain else set()
            lag_syms: dict[str, int] = {}
            lag_syms.update(self._collect_lag_symbols(cmeta["expression"], variables, parameters, sets))
            lag_syms.update(self._collect_lag_symbols(cmeta["rhs"], variables, parameters, sets))
            lag_ctx = self._build_lag_context(lag_syms, sets) if lag_syms else {}
            lhs = self._emit_expr_gurobi(cmeta["expression"], index_map, variables, parameters, extra_known=domain_loop_vars, lag_context=lag_ctx or None)
            rhs = self._emit_expr_gurobi(cmeta["rhs"], index_map, variables, parameters, extra_known=domain_loop_vars, lag_context=lag_ctx or None)
            sf_precompute, sf_guard = self._sparse_filter_guard(sparse_filter, parameters, index_map, domain)
            if sf_precompute:
                lines.append(sf_precompute)

            if not domain:
                lines.append(f"    m.addConstr({lhs} {sense_op} {rhs}, name={cname!r})")
            elif len(domain) == 1:
                idx0 = index_map[domain[0]]
                if idx0 in lag_ctx:
                    _, _ev0 = lag_ctx[idx0]
                    _lv0 = lag_syms[idx0]
                    lines.append(f"    for {_ev0}, {idx0} in enumerate({domain[0]}):")
                    if _lv0 < 0:
                        lines.append(f"        if {_ev0} < {-_lv0}: continue")
                    else:
                        lines.append(f"        if {_ev0} + {_lv0} >= len({domain[0]}): continue")
                else:
                    lines.append(f"    for {idx0} in {domain[0]}:")
                if sf_guard:
                    lines.append(f"        if {sf_guard}: continue")
                lines.append(
                    f"        m.addConstr({lhs} {sense_op} {rhs}, name=f\"{cname}_{{{idx0}}}\")"
                )
            elif len(domain) == 2:
                idx0, idx1 = self._domain_idx_vars(domain, index_map)
                if idx0 in lag_ctx:
                    _, _ev0 = lag_ctx[idx0]
                    _lv0 = lag_syms[idx0]
                    lines.append(f"    for {_ev0}, {idx0} in enumerate({domain[0]}):")
                    if _lv0 < 0:
                        lines.append(f"        if {_ev0} < {-_lv0}: continue")
                    else:
                        lines.append(f"        if {_ev0} + {_lv0} >= len({domain[0]}): continue")
                else:
                    lines.append(f"    for {idx0} in {domain[0]}:")
                if idx1 in lag_ctx:
                    _, _ev1 = lag_ctx[idx1]
                    _lv1 = lag_syms[idx1]
                    lines.append(f"        for {_ev1}, {idx1} in enumerate({domain[1]}):")
                    if _lv1 < 0:
                        lines.append(f"            if {_ev1} < {-_lv1}: continue")
                    else:
                        lines.append(f"            if {_ev1} + {_lv1} >= len({domain[1]}): continue")
                else:
                    lines.append(f"        for {idx1} in {domain[1]}:")
                guard = self._constraint_diagonal_guard(domain, [idx0, idx1])
                if guard:
                    lines.append(f"            if {guard}: continue")
                if sf_guard:
                    lines.append(f"            if {sf_guard}: continue")
                lines.append(
                    f"            m.addConstr({lhs} {sense_op} {rhs}, "
                    f"name=f\"{cname}_{{{idx0}}}_{{{idx1}}}\")"
                )
            else:
                idx_vars = self._domain_idx_vars(domain, index_map)
                for k, (iv, s) in enumerate(zip(idx_vars, domain)):
                    if iv in lag_ctx:
                        _, _ev = lag_ctx[iv]
                        _lv = lag_syms[iv]
                        lines.append(f"    {'    ' * k}for {_ev}, {iv} in enumerate({s}):")
                        if _lv < 0:
                            lines.append(f"    {'    ' * (k + 1)}if {_ev} < {-_lv}: continue")
                        else:
                            lines.append(f"    {'    ' * (k + 1)}if {_ev} + {_lv} >= len({s}): continue")
                    else:
                        lines.append(f"    {'    ' * k}for {iv} in {s}:")
                name_parts = "\\x1f".join(f"{{{iv}}}" for iv in idx_vars)
                inner = "    " * (len(domain) + 1)
                guard = self._constraint_diagonal_guard(domain, idx_vars)
                if guard:
                    lines.append(f"{inner}if {guard}: continue")
                if sf_guard:
                    lines.append(f"{inner}if {sf_guard}: continue")
                lines.append(
                    f"{inner}m.addConstr({lhs} {sense_op} {rhs}, "
                    f"name=f\"{cname}_{name_parts}\")"
                )

        lines += [
            "",
            "    # --- Solve ---",
            "    try:",
            "        m.write('model.lp')",
            "    except Exception:",
            "        pass  # LP write is best-effort",
            "    m.optimize()",
            "",
            "    _status_map = {",
            "        GRB.OPTIMAL: 'optimal',",
            "        GRB.SUBOPTIMAL: 'feasible',",
            "        GRB.INFEASIBLE: 'infeasible',",
            "        GRB.UNBOUNDED: 'unbounded',",
            "    }",
            "    result = {",
            "        'status': _status_map.get(m.Status, 'error'),",
            "        'objective_value': None,",
            "        'variables': {},",
            "        'variable_groups': [],",
            "    }",
            "    if m.Status in (GRB.OPTIMAL, GRB.SUBOPTIMAL):",
            "        result['objective_value'] = m.ObjVal",
        ]

        ext_lines: list[str] = []
        for var_name, meta in variables.items():
            domain = meta.get("domain", [])
            excl = meta.get("exclude_diagonal", False)
            ext_lines.append(f"    # extract {var_name}")
            if not domain:
                ext_lines.append(f"    result['variables'][{var_name!r}] = {var_name}.X")
            elif len(domain) == 1:
                idx0 = index_map[domain[0]]
                df_cond = self._domain_filter_cond(meta, parameters, [idx0], domain)
                ext_lines.append(f"    for {idx0} in {domain[0]}:")
                if df_cond:
                    ext_lines.append(f"        if not {df_cond}: continue")
                ext_lines.append(
                    f"        result['variables'][f\"{var_name}\\x1f{{{idx0}}}\"] = "
                    f"{var_name}[{idx0}].X"
                )
            elif len(domain) == 2:
                idx0, idx1 = self._domain_idx_vars(domain, index_map)
                df_cond = self._domain_filter_cond(meta, parameters, [idx0, idx1], domain)
                ext_lines.append(f"    for {idx0} in {domain[0]}:")
                ext_lines.append(f"        for {idx1} in {domain[1]}:")
                if excl:
                    ext_lines.append(f"            if {idx0} == {idx1}: continue")
                if df_cond:
                    ext_lines.append(f"            if not {df_cond}: continue")
                ext_lines.append(
                    f"            result['variables'][f\"{var_name}\\x1f{{{idx0}}}\\x1f{{{idx1}}}\"] = "
                    f"{var_name}[({idx0}, {idx1})].X"
                )
            else:
                idx_vars = self._domain_idx_vars(domain, index_map)
                for k, (iv, s) in enumerate(zip(idx_vars, domain)):
                    ext_lines.append(f"    {'    ' * k}for {iv} in {s}:")
                inner = "    " * (len(domain) + 1)
                if excl and len(idx_vars) >= 2:
                    _da, _db = self._excl_diag_pair(domain, idx_vars)
                    ext_lines.append(f"{inner}if {_da} == {_db}: continue")
                df_cond = self._domain_filter_cond(meta, parameters, idx_vars, domain)
                if df_cond:
                    ext_lines.append(f"{inner}if not {df_cond}: continue")
                idx_tuple = "(" + ", ".join(idx_vars) + ")"
                name_parts = "\\x1f".join(f"{{{iv}}}" for iv in idx_vars)
                ext_lines.append(
                    f"{inner}result['variables'][f\"{var_name}\\x1f{name_parts}\"] = "
                    f"{var_name}[{idx_tuple}].X"
                )

        # Nest extraction inside the status check to prevent GurobiError on failed solves
        for _el in ext_lines:
            lines.append("    " + _el)

        lines += self._emit_variable_groups(variables)
        lines.append("    return result")
        return "\n".join(lines) + "\n"

    # ------------------------------------------------------------------
    # CPLEX backend (native docplex)
    # ------------------------------------------------------------------

    def _compile_cplex(self, ir: dict) -> str:
        """Compile IR to native docplex (CPLEX Python API) code (no PuLP dependency)."""
        sets = ir.get("sets", {})
        parameters = ir.get("parameters", {})
        variables = ir.get("variables", {})
        constraints = ir.get("constraints", {})
        objective = ir.get("objective", {})
        sense = ir.get("sense", "minimize")
        problem_class = ir.get("problem_class", "Model")

        index_map: dict[str, str] = {n: m["index_symbol"] for n, m in sets.items()}
        set_column: dict[str, str | None] = {n: m.get("column") for n, m in sets.items()}

        lines: list[str] = [
            "from docplex.mp.model import Model",
            "",
            "",
            "def solve(data: dict, time_limit: int | None = None, show_solver_log: bool = False) -> dict:",
            "    # --- Load sets ---",
        ]
        lines += self._emit_set_loading(sets, set_column)
        lines.append("")
        lines.append("    # --- Load parameters ---")
        lines += self._emit_parameter_loading(parameters, set_column, index_map)

        lines += [
            "",
            "    # --- Build model ---",
            f"    mdl = Model(name={problem_class!r})",
            "",
            "    # --- Decision variables ---",
        ]

        for var_name, meta in variables.items():
            domain = meta.get("domain", [])
            vtype = meta.get("type", "continuous")
            lb = meta.get("lower_bound")
            ub = meta.get("upper_bound")
            lb_str = str(float(lb)) if lb is not None else "0"
            if ub is not None:
                ub_str = str(float(ub))
            elif meta.get("upper_bound_set"):
                ub_str = f"len({meta['upper_bound_set']})"
            else:
                ub_str = "None"
            excl = meta.get("exclude_diagonal", False)

            if vtype == "binary":
                dict_factory = "binary_var_dict"
                scalar_factory = "binary_var"
                bounds_kw = ""
            elif vtype == "integer":
                dict_factory = "integer_var_dict"
                scalar_factory = "integer_var"
                bounds_kw = f", lb={lb_str}, ub={ub_str}"
            else:
                dict_factory = "continuous_var_dict"
                scalar_factory = "continuous_var"
                bounds_kw = f", lb={lb_str}, ub={ub_str}"

            if not domain:
                if vtype == "binary":
                    lines.append(f"    {var_name} = mdl.{scalar_factory}(name={var_name!r})")
                else:
                    lines.append(
                        f"    {var_name} = mdl.{scalar_factory}(lb={lb_str}, ub={ub_str}, name={var_name!r})"
                    )
            elif len(domain) == 1:
                idx0 = index_map[domain[0]]
                df_cond = self._domain_filter_cond(meta, parameters, [idx0], domain)
                if df_cond:
                    lines.append(
                        f"    {var_name} = mdl.{dict_factory}("
                        f"[{idx0} for {idx0} in {domain[0]} if {df_cond}], "
                        f"name={var_name!r}{bounds_kw})"
                    )
                else:
                    lines.append(
                        f"    {var_name} = mdl.{dict_factory}({domain[0]}, name={var_name!r}{bounds_kw})"
                    )
            elif len(domain) == 2:
                idx0, idx1 = self._domain_idx_vars(domain, index_map)
                s0, s1 = domain[0], domain[1]
                diag_guard = f" if {idx0} != {idx1}" if excl else ""
                df_cond = self._domain_filter_cond(meta, parameters, [idx0, idx1], domain)
                df_guard = f" if {df_cond}" if df_cond else ""
                lines.append(
                    f"    {var_name} = mdl.{dict_factory}("
                    f"[({idx0}, {idx1}) for {idx0} in {s0} for {idx1} in {s1}{diag_guard}{df_guard}], "
                    f"name={var_name!r}{bounds_kw})"
                )
            else:
                idx_vars = self._domain_idx_vars(domain, index_map)
                iterators = " ".join(f"for {iv} in {s}" for iv, s in zip(idx_vars, domain))
                idx_tuple = "(" + ", ".join(idx_vars) + ")"
                diag_guard = self._excl_diag_guard(domain, idx_vars) if excl else ""
                df_cond = self._domain_filter_cond(meta, parameters, idx_vars, domain)
                df_guard = f" if {df_cond}" if df_cond else ""
                lines.append(
                    f"    {var_name} = mdl.{dict_factory}("
                    f"[{idx_tuple} {iterators}{diag_guard}{df_guard}], "
                    f"name={var_name!r}{bounds_kw})"
                )

        lines.append("")
        lines.append("    # --- Objective ---")
        terms = self._flatten_obj_terms(objective["expression"])
        cplex_fn = "mdl.minimize" if sense == "minimize" else "mdl.maximize"
        lines.append(f"    {cplex_fn}(")
        for i, (sign, node) in enumerate(terms):
            expr = self._emit_expr_cplex(node, index_map, variables, parameters)
            is_last = i == len(terms) - 1
            suffix = "," if is_last else ""
            if i == 0:
                prefix = "        " if sign == 1 else "        -"
            else:
                prefix = "        + " if sign == 1 else "        - "
            lines.append(f"{prefix}{expr}{suffix}")
        lines.append("    )")

        lines.append("")
        lines.append("    # --- Constraints ---")
        for cname, cmeta in constraints.items():
            domain = cmeta.get("domain", [])
            sense_op = {"<=": "<=", ">=": ">=", "=": "=="}.get(cmeta.get("sense", "<="), "<=")
            sparse_filter = cmeta.get("sparse_filter")
            domain_loop_vars = set(self._domain_idx_vars(domain, index_map)) if domain else set()
            lag_syms: dict[str, int] = {}
            lag_syms.update(self._collect_lag_symbols(cmeta["expression"], variables, parameters, sets))
            lag_syms.update(self._collect_lag_symbols(cmeta["rhs"], variables, parameters, sets))
            lag_ctx = self._build_lag_context(lag_syms, sets) if lag_syms else {}
            lhs = self._emit_expr_cplex(cmeta["expression"], index_map, variables, parameters, extra_known=domain_loop_vars, lag_context=lag_ctx or None)
            rhs = self._emit_expr_cplex(cmeta["rhs"], index_map, variables, parameters, extra_known=domain_loop_vars, lag_context=lag_ctx or None)
            sf_precompute, sf_guard = self._sparse_filter_guard(sparse_filter, parameters, index_map, domain)
            if sf_precompute:
                lines.append(sf_precompute)

            if not domain:
                lines.append(f"    mdl.add_constraint({lhs} {sense_op} {rhs}, ctname={cname!r})")
            elif len(domain) == 1:
                idx0 = index_map[domain[0]]
                if idx0 in lag_ctx:
                    _, _ev0 = lag_ctx[idx0]
                    _lv0 = lag_syms[idx0]
                    lines.append(f"    for {_ev0}, {idx0} in enumerate({domain[0]}):")
                    if _lv0 < 0:
                        lines.append(f"        if {_ev0} < {-_lv0}: continue")
                    else:
                        lines.append(f"        if {_ev0} + {_lv0} >= len({domain[0]}): continue")
                else:
                    lines.append(f"    for {idx0} in {domain[0]}:")
                if sf_guard:
                    lines.append(f"        if {sf_guard}: continue")
                lines.append(
                    f"        mdl.add_constraint({lhs} {sense_op} {rhs}, ctname=f\"{cname}_{{{idx0}}}\")"
                )
            elif len(domain) == 2:
                idx0, idx1 = self._domain_idx_vars(domain, index_map)
                if idx0 in lag_ctx:
                    _, _ev0 = lag_ctx[idx0]
                    _lv0 = lag_syms[idx0]
                    lines.append(f"    for {_ev0}, {idx0} in enumerate({domain[0]}):")
                    if _lv0 < 0:
                        lines.append(f"        if {_ev0} < {-_lv0}: continue")
                    else:
                        lines.append(f"        if {_ev0} + {_lv0} >= len({domain[0]}): continue")
                else:
                    lines.append(f"    for {idx0} in {domain[0]}:")
                if idx1 in lag_ctx:
                    _, _ev1 = lag_ctx[idx1]
                    _lv1 = lag_syms[idx1]
                    lines.append(f"        for {_ev1}, {idx1} in enumerate({domain[1]}):")
                    if _lv1 < 0:
                        lines.append(f"            if {_ev1} < {-_lv1}: continue")
                    else:
                        lines.append(f"            if {_ev1} + {_lv1} >= len({domain[1]}): continue")
                else:
                    lines.append(f"        for {idx1} in {domain[1]}:")
                guard = self._constraint_diagonal_guard(domain, [idx0, idx1])
                if guard:
                    lines.append(f"            if {guard}: continue")
                if sf_guard:
                    lines.append(f"            if {sf_guard}: continue")
                lines.append(
                    f"            mdl.add_constraint({lhs} {sense_op} {rhs}, "
                    f"ctname=f\"{cname}_{{{idx0}}}_{{{idx1}}}\")"
                )
            else:
                idx_vars = self._domain_idx_vars(domain, index_map)
                for k, (iv, s) in enumerate(zip(idx_vars, domain)):
                    if iv in lag_ctx:
                        _, _ev = lag_ctx[iv]
                        _lv = lag_syms[iv]
                        lines.append(f"    {'    ' * k}for {_ev}, {iv} in enumerate({s}):")
                        if _lv < 0:
                            lines.append(f"    {'    ' * (k + 1)}if {_ev} < {-_lv}: continue")
                        else:
                            lines.append(f"    {'    ' * (k + 1)}if {_ev} + {_lv} >= len({s}): continue")
                    else:
                        lines.append(f"    {'    ' * k}for {iv} in {s}:")
                name_parts = "\\x1f".join(f"{{{iv}}}" for iv in idx_vars)
                inner = "    " * (len(domain) + 1)
                guard = self._constraint_diagonal_guard(domain, idx_vars)
                if guard:
                    lines.append(f"{inner}if {guard}: continue")
                if sf_guard:
                    lines.append(f"{inner}if {sf_guard}: continue")
                lines.append(
                    f"{inner}mdl.add_constraint({lhs} {sense_op} {rhs}, "
                    f"ctname=f\"{cname}_{name_parts}\")"
                )

        lines += [
            "",
            "    # --- Solve ---",
            "    if time_limit is not None:",
            "        mdl.set_time_limit(time_limit)",
            "    sol = mdl.solve(log_output=show_solver_log)",
            "    try:",
            "        mdl.export_as_lp('model.lp')",
            "    except Exception:",
            "        pass  # LP export is best-effort",
            "    if sol is None:",
            "        _status = 'infeasible'",
            "    else:",
            "        _ds = str(mdl.solve_details.status).lower()",
            "        if 'optimal' in _ds:",
            "            _status = 'optimal'",
            "        elif 'feasible' in _ds:",
            "            _status = 'feasible'",
            "        elif 'infeasible' in _ds:",
            "            _status = 'infeasible'",
            "        elif 'unbounded' in _ds:",
            "            _status = 'unbounded'",
            "        else:",
            "            _status = 'error'",
            "    result = {",
            "        'status': _status,",
            "        'objective_value': None,",
            "        'variables': {},",
            "        'variable_groups': [],",
            "    }",
            "    if _status in ('optimal', 'feasible') and sol is not None:",
            "        result['objective_value'] = mdl.objective_value",
        ]

        ext_lines_cplex: list[str] = []
        for var_name, meta in variables.items():
            domain = meta.get("domain", [])
            excl = meta.get("exclude_diagonal", False)
            ext_lines_cplex.append(f"    # extract {var_name}")
            if not domain:
                ext_lines_cplex.append(
                    f"    result['variables'][{var_name!r}] = {var_name}.solution_value"
                )
            elif len(domain) == 1:
                idx0 = index_map[domain[0]]
                df_cond = self._domain_filter_cond(meta, parameters, [idx0], domain)
                ext_lines_cplex.append(f"    for {idx0} in {domain[0]}:")
                if df_cond:
                    ext_lines_cplex.append(f"        if not {df_cond}: continue")
                ext_lines_cplex.append(
                    f"        result['variables'][f\"{var_name}\\x1f{{{idx0}}}\"] = "
                    f"{var_name}[{idx0}].solution_value"
                )
            elif len(domain) == 2:
                idx0, idx1 = self._domain_idx_vars(domain, index_map)
                df_cond = self._domain_filter_cond(meta, parameters, [idx0, idx1], domain)
                ext_lines_cplex.append(f"    for {idx0} in {domain[0]}:")
                ext_lines_cplex.append(f"        for {idx1} in {domain[1]}:")
                if excl:
                    ext_lines_cplex.append(f"            if {idx0} == {idx1}: continue")
                if df_cond:
                    ext_lines_cplex.append(f"            if not {df_cond}: continue")
                ext_lines_cplex.append(
                    f"            result['variables'][f\"{var_name}\\x1f{{{idx0}}}\\x1f{{{idx1}}}\"] = "
                    f"{var_name}[({idx0}, {idx1})].solution_value"
                )
            else:
                idx_vars = self._domain_idx_vars(domain, index_map)
                for k, (iv, s) in enumerate(zip(idx_vars, domain)):
                    ext_lines_cplex.append(f"    {'    ' * k}for {iv} in {s}:")
                inner = "    " * (len(domain) + 1)
                if excl and len(idx_vars) >= 2:
                    _da, _db = self._excl_diag_pair(domain, idx_vars)
                    ext_lines_cplex.append(f"{inner}if {_da} == {_db}: continue")
                df_cond = self._domain_filter_cond(meta, parameters, idx_vars, domain)
                if df_cond:
                    ext_lines_cplex.append(f"{inner}if not {df_cond}: continue")
                idx_tuple = "(" + ", ".join(idx_vars) + ")"
                name_parts = "\\x1f".join(f"{{{iv}}}" for iv in idx_vars)
                ext_lines_cplex.append(
                    f"{inner}result['variables'][f\"{var_name}\\x1f{name_parts}\"] = "
                    f"{var_name}[{idx_tuple}].solution_value"
                )

        # Nest extraction inside the status check to prevent errors on failed solves
        for _el in ext_lines_cplex:
            lines.append("    " + _el)

        lines += self._emit_variable_groups(variables)
        lines.append("    return result")
        return "\n".join(lines) + "\n"

    # ------------------------------------------------------------------
    # Pyomo backend
    # ------------------------------------------------------------------

    def _compile_pyomo(self, ir: dict) -> str:
        sets = ir.get("sets", {})
        parameters = ir.get("parameters", {})
        variables = ir.get("variables", {})
        constraints = ir.get("constraints", {})
        objective = ir.get("objective", {})
        sense = ir.get("sense", "minimize")
        problem_class = ir.get("problem_class", "Model")

        index_map: dict[str, str] = {n: m["index_symbol"] for n, m in sets.items()}
        set_column: dict[str, str | None] = {n: m.get("column") for n, m in sets.items()}

        lines: list[str] = [
            "import pyomo.environ as pyo",
            "",
            "",
            "def solve(data: dict, time_limit: int | None = None, show_solver_log: bool = False) -> dict:",
            "    # --- Load sets ---",
        ]
        lines += self._emit_set_loading(sets, set_column)
        lines.append("")
        lines.append("    # --- Load parameters ---")
        lines += self._emit_parameter_loading(parameters, set_column, index_map)

        pyo_sense = "pyo.minimize" if sense == "minimize" else "pyo.maximize"
        lines += [
            "",
            "    # --- Build model ---",
            f"    model = pyo.ConcreteModel(name={problem_class!r})",
            "",
            "    # --- Decision variables ---",
        ]

        for var_name, meta in variables.items():
            domain = meta.get("domain", [])
            vtype = meta.get("type", "continuous")
            lb = meta.get("lower_bound")
            ub = meta.get("upper_bound")
            lb_str = str(lb) if lb is not None else "0"
            if ub is not None:
                ub_str = str(ub)
            elif meta.get("upper_bound_set"):
                ub_str = f"len({meta['upper_bound_set']})"
            else:
                ub_str = "None"
            excl = meta.get("exclude_diagonal", False)

            if vtype == "binary":
                within = "pyo.Binary"
                bounds = ""
            elif vtype == "integer":
                within = "pyo.Integers"
                bounds = f", bounds=({lb_str}, {ub_str})"
            else:
                within = "pyo.Reals"
                bounds = f", bounds=({lb_str}, {ub_str})"

            if not domain:
                lines.append(
                    f"    model.{var_name} = pyo.Var(within={within}{bounds})"
                )
            elif len(domain) == 1:
                idx0 = index_map[domain[0]]
                df_cond = self._domain_filter_cond(meta, parameters, [idx0], domain)
                if df_cond:
                    lines.append(
                        f"    model.{var_name} = pyo.Var("
                        f"[{idx0} for {idx0} in {domain[0]} if {df_cond}], within={within}{bounds})"
                    )
                else:
                    lines.append(
                        f"    model.{var_name} = pyo.Var({domain[0]}, within={within}{bounds})"
                    )
            elif len(domain) == 2:
                s0, s1 = domain[0], domain[1]
                idx0, idx1 = self._domain_idx_vars(domain, index_map)
                diag_guard = f" if {idx0} != {idx1}" if excl else ""
                df_cond = self._domain_filter_cond(meta, parameters, [idx0, idx1], domain)
                df_guard = f" if {df_cond}" if df_cond else ""
                if excl or df_cond:
                    lines.append(
                        f"    model.{var_name} = pyo.Var("
                        f"[({idx0}, {idx1}) for {idx0} in {s0} for {idx1} in {s1}{diag_guard}{df_guard}], "
                        f"within={within}{bounds})"
                    )
                else:
                    lines.append(
                        f"    model.{var_name} = pyo.Var({s0}, {s1}, within={within}{bounds})"
                    )
            else:
                idx_vars = self._domain_idx_vars(domain, index_map)
                iterators = " ".join(f"for {iv} in {s}" for iv, s in zip(idx_vars, domain))
                idx_tuple = "(" + ", ".join(idx_vars) + ")"
                diag_guard = self._excl_diag_guard(domain, idx_vars) if excl else ""
                df_cond = self._domain_filter_cond(meta, parameters, idx_vars, domain)
                df_guard = f" if {df_cond}" if df_cond else ""
                if excl or df_cond:
                    lines.append(
                        f"    model.{var_name} = pyo.Var("
                        f"[{idx_tuple} {iterators}{diag_guard}{df_guard}], within={within}{bounds})"
                    )
                else:
                    set_args = ", ".join(domain)
                    lines.append(
                        f"    model.{var_name} = pyo.Var({set_args}, within={within}{bounds})"
                    )

        lines.append("")
        lines.append("    # --- Objective ---")
        terms = self._flatten_obj_terms(objective["expression"])
        lines.append("    model.obj = pyo.Objective(expr=(")
        for i, (sign, node) in enumerate(terms):
            expr = self._emit_pyomo_expr(node, index_map, variables, parameters)
            is_last = i == len(terms) - 1
            suffix = "," if is_last else ""
            if i == 0:
                prefix = "        " if sign == 1 else "        -"
            else:
                prefix = "        + " if sign == 1 else "        - "
            lines.append(f"{prefix}{expr}{suffix}")
        lines.append(f"    ), sense={pyo_sense})")

        lines.append("")
        lines.append("    # --- Constraints ---")
        for cname, cmeta in constraints.items():
            domain = cmeta.get("domain", [])
            sense_op = {"<=": "<=", ">=": ">=", "=": "=="}.get(cmeta.get("sense", "<="), "<=")
            domain_loop_vars = set(self._domain_idx_vars(domain, index_map)) if domain else set()
            lag_syms: dict[str, int] = {}
            lag_syms.update(self._collect_lag_symbols(cmeta["expression"], variables, parameters, sets))
            lag_syms.update(self._collect_lag_symbols(cmeta["rhs"], variables, parameters, sets))
            lag_ctx = self._build_lag_context(lag_syms, sets) if lag_syms else {}
            lhs = self._emit_pyomo_expr(cmeta["expression"], index_map, variables, parameters, extra_known=domain_loop_vars, lag_context=lag_ctx or None)
            rhs = self._emit_pyomo_expr(cmeta["rhs"], index_map, variables, parameters, extra_known=domain_loop_vars, lag_context=lag_ctx or None)
            sparse_filter = cmeta.get("sparse_filter")
            sf_precompute, sf_guard = self._sparse_filter_guard(sparse_filter, parameters, index_map, domain)
            if sf_precompute:
                lines.append(sf_precompute)

            lines.append(f"    model.{cname} = pyo.ConstraintList()")
            if not domain:
                lines.append(f"    model.{cname}.add({lhs} {sense_op} {rhs})")
            elif len(domain) == 1:
                idx0 = index_map[domain[0]]
                if idx0 in lag_ctx:
                    _, _ev0 = lag_ctx[idx0]
                    _lv0 = lag_syms[idx0]
                    lines.append(f"    for {_ev0}, {idx0} in enumerate({domain[0]}):")
                    if _lv0 < 0:
                        lines.append(f"        if {_ev0} < {-_lv0}: continue")
                    else:
                        lines.append(f"        if {_ev0} + {_lv0} >= len({domain[0]}): continue")
                else:
                    lines.append(f"    for {idx0} in {domain[0]}:")
                if sf_guard:
                    lines.append(f"        if {sf_guard}: continue")
                lines.append(f"        model.{cname}.add({lhs} {sense_op} {rhs})")
            elif len(domain) == 2:
                idx0, idx1 = self._domain_idx_vars(domain, index_map)
                if idx0 in lag_ctx:
                    _, _ev0 = lag_ctx[idx0]
                    _lv0 = lag_syms[idx0]
                    lines.append(f"    for {_ev0}, {idx0} in enumerate({domain[0]}):")
                    if _lv0 < 0:
                        lines.append(f"        if {_ev0} < {-_lv0}: continue")
                    else:
                        lines.append(f"        if {_ev0} + {_lv0} >= len({domain[0]}): continue")
                else:
                    lines.append(f"    for {idx0} in {domain[0]}:")
                if idx1 in lag_ctx:
                    _, _ev1 = lag_ctx[idx1]
                    _lv1 = lag_syms[idx1]
                    lines.append(f"        for {_ev1}, {idx1} in enumerate({domain[1]}):")
                    if _lv1 < 0:
                        lines.append(f"            if {_ev1} < {-_lv1}: continue")
                    else:
                        lines.append(f"            if {_ev1} + {_lv1} >= len({domain[1]}): continue")
                else:
                    lines.append(f"        for {idx1} in {domain[1]}:")
                guard = self._constraint_diagonal_guard(domain, [idx0, idx1])
                if guard:
                    lines.append(f"            if {guard}: continue")
                if sf_guard:
                    lines.append(f"            if {sf_guard}: continue")
                lines.append(f"            model.{cname}.add({lhs} {sense_op} {rhs})")
            else:
                idx_vars = self._domain_idx_vars(domain, index_map)
                for k, (iv, s) in enumerate(zip(idx_vars, domain)):
                    if iv in lag_ctx:
                        _, _ev = lag_ctx[iv]
                        _lv = lag_syms[iv]
                        lines.append(f"    {'    ' * k}for {_ev}, {iv} in enumerate({s}):")
                        if _lv < 0:
                            lines.append(f"    {'    ' * (k + 1)}if {_ev} < {-_lv}: continue")
                        else:
                            lines.append(f"    {'    ' * (k + 1)}if {_ev} + {_lv} >= len({s}): continue")
                    else:
                        lines.append(f"    {'    ' * k}for {iv} in {s}:")
                inner = "    " * (len(domain) + 1)
                guard = self._constraint_diagonal_guard(domain, idx_vars)
                if guard:
                    lines.append(f"{inner}if {guard}: continue")
                if sf_guard:
                    lines.append(f"{inner}if {sf_guard}: continue")
                lines.append(f"{inner}model.{cname}.add({lhs} {sense_op} {rhs})")

        lines += [
            "",
            "    # --- Solve ---",
            "    print('[ORPilot] Generating LP file...')",
            "    try:",
            "        model.write('model.lp', io_options={'symbolic_solver_labels': True})",
            "    except Exception:",
            "        try:",
            "            model.write('model.lp')",
            "        except Exception:",
            "            pass  # LP write is best-effort",
            "    _solver = None",
            "    for _sname in ['appsi_highs', 'glpk', 'cbc']:",
            "        _s = pyo.SolverFactory(_sname)",
            "        if _s.available(exception_flag=False):",
            "            _solver = _s",
            "            break",
            "    if _solver is None:",
            "        raise RuntimeError(",
            "            'No Pyomo solver found. Install HiGHS (pip install highspy), GLPK, or CBC.'",
            "        )",
            "    print(f'[ORPilot] Using Pyomo solver: {_sname}')",
            "    if time_limit is not None:",
            "        _sname_used = type(_solver).__name__.lower()",
            "        if 'highs' in _sname_used or 'appsi' in _sname_used:",
            "            _solver.options['time_limit'] = time_limit",
            "        elif 'glpk' in _sname_used:",
            "            _solver.options['tmlim'] = time_limit",
            "        else:  # cbc",
            "            _solver.options['sec'] = time_limit",
            "    _results = _solver.solve(model, tee=show_solver_log)",
            "",
            "    _tc = str(_results.solver.termination_condition).lower()",
            "    if 'optimal' in _tc:",
            "        _status = 'optimal'",
            "    elif 'feasible' in _tc:",
            "        _status = 'feasible'",
            "    elif 'infeasible' in _tc:",
            "        _status = 'infeasible'",
            "    elif 'unbounded' in _tc:",
            "        _status = 'unbounded'",
            "    else:",
            "        _status = 'error'",
            "    result = {",
            "        'status': _status,",
            "        'objective_value': None,",
            "        'variables': {},",
            "        'variable_groups': [],",
            "    }",
            "    if _status in ('optimal', 'feasible'):",
            "        result['objective_value'] = pyo.value(model.obj)",
        ]

        for var_name, meta in variables.items():
            domain = meta.get("domain", [])
            excl = meta.get("exclude_diagonal", False)
            lines.append(f"    # extract {var_name}")
            if not domain:
                lines.append(
                    f"    result['variables'][{var_name!r}] = pyo.value(model.{var_name})"
                )
            elif len(domain) == 1:
                idx0 = index_map[domain[0]]
                df_cond = self._domain_filter_cond(meta, parameters, [idx0], domain)
                lines.append(f"    for {idx0} in {domain[0]}:")
                if df_cond:
                    lines.append(f"        if not {df_cond}: continue")
                lines.append(
                    f"        result['variables'][f\"{var_name}\\x1f{{{idx0}}}\"] = "
                    f"pyo.value(model.{var_name}[{idx0}])"
                )
            elif len(domain) == 2:
                idx0, idx1 = self._domain_idx_vars(domain, index_map)
                df_cond = self._domain_filter_cond(meta, parameters, [idx0, idx1], domain)
                lines.append(f"    for {idx0} in {domain[0]}:")
                lines.append(f"        for {idx1} in {domain[1]}:")
                if excl:
                    lines.append(f"            if {idx0} == {idx1}: continue")
                if df_cond:
                    lines.append(f"            if not {df_cond}: continue")
                lines.append(
                    f"            result['variables'][f\"{var_name}\\x1f{{{idx0}}}\\x1f{{{idx1}}}\"] = "
                    f"pyo.value(model.{var_name}[{idx0}, {idx1}])"
                )
            else:
                idx_vars = self._domain_idx_vars(domain, index_map)
                for k, (iv, s) in enumerate(zip(idx_vars, domain)):
                    lines.append(f"    {'    ' * k}for {iv} in {s}:")
                inner = "    " * (len(domain) + 1)
                if excl and len(idx_vars) >= 2:
                    _da, _db = self._excl_diag_pair(domain, idx_vars)
                    lines.append(f"{inner}if {_da} == {_db}: continue")
                df_cond = self._domain_filter_cond(meta, parameters, idx_vars, domain)
                if df_cond:
                    lines.append(f"{inner}if not {df_cond}: continue")
                idx_str = ", ".join(idx_vars)
                name_parts = "\\x1f".join(f"{{{iv}}}" for iv in idx_vars)
                lines.append(
                    f"{inner}result['variables'][f\"{var_name}\\x1f{name_parts}\"] = "
                    f"pyo.value(model.{var_name}[{idx_str}])"
                )

        lines += self._emit_variable_groups(variables)
        lines.append("    return result")
        return "\n".join(lines) + "\n"

    # ------------------------------------------------------------------
    # OR-Tools backend
    # ------------------------------------------------------------------

    def _compile_ortools(self, ir: dict) -> str:
        sets = ir.get("sets", {})
        parameters = ir.get("parameters", {})
        variables = ir.get("variables", {})
        constraints = ir.get("constraints", {})
        objective = ir.get("objective", {})
        sense = ir.get("sense", "minimize")
        problem_class = ir.get("problem_class", "Model")
        model_type = ir.get("model_type", "Linear Program")

        index_map: dict[str, str] = {n: m["index_symbol"] for n, m in sets.items()}
        set_column: dict[str, str | None] = {n: m.get("column") for n, m in sets.items()}

        is_mip = "Integer" in model_type or "Mixed" in model_type
        solver_type = "'SCIP'" if is_mip else "'GLOP'"

        lines: list[str] = [
            "from ortools.linear_solver import pywraplp",
            "",
            "",
            "def solve(data: dict, time_limit: int | None = None, show_solver_log: bool = False) -> dict:",
            "    # --- Load sets ---",
        ]
        lines += self._emit_set_loading(sets, set_column)
        lines.append("")
        lines.append("    # --- Load parameters ---")
        lines += self._emit_parameter_loading(parameters, set_column, index_map)

        lines += [
            "",
            "    # --- Build solver ---",
            f"    solver = pywraplp.Solver.CreateSolver({solver_type})",
            "    if not solver:",
            "        solver = pywraplp.Solver.CreateSolver('SCIP')",
            "    if not solver:",
            f'        raise RuntimeError("OR-Tools solver {solver_type} not available")',
            f"    solver.SetSolverSpecificParametersAsString('')",
            "",
            "    # --- Decision variables ---",
        ]

        for var_name, meta in variables.items():
            domain = meta.get("domain", [])
            vtype = meta.get("type", "continuous")
            lb = meta.get("lower_bound")
            ub = meta.get("upper_bound")
            lb_str = str(float(lb)) if lb is not None else "0.0"

            # Upper bound expression: solver.infinity() for continuous, int cap for integer
            if ub is not None:
                if vtype == "integer":
                    ub_str = str(int(ub))
                else:
                    ub_str = str(float(ub))
            elif vtype == "continuous":
                ub_str = "solver.infinity()"
            elif meta.get("upper_bound_set"):
                ub_str = f"len({meta['upper_bound_set']})"
            else:
                ub_str = "int(1e9)"

            if not domain:
                if vtype == "binary":
                    lines.append(f"    {var_name} = solver.BoolVar({var_name!r})")
                elif vtype == "integer":
                    lines.append(
                        f"    {var_name} = solver.IntVar({lb_str}, {ub_str}, {var_name!r})"
                    )
                else:
                    lines.append(
                        f"    {var_name} = solver.NumVar({lb_str}, {ub_str}, {var_name!r})"
                    )
                continue

            lines.append(f"    {var_name} = {{}}")
            if len(domain) == 1:
                idx0 = index_map[domain[0]]
                df_cond = self._domain_filter_cond(meta, parameters, [idx0], domain)
                lines.append(f"    for {idx0} in {domain[0]}:")
                if df_cond:
                    lines.append(f"        if not {df_cond}: continue")
                if vtype == "binary":
                    lines.append(
                        f"        {var_name}[{idx0}] = solver.BoolVar(f'{var_name}_{{{idx0}}}')"
                    )
                elif vtype == "integer":
                    lines.append(
                        f"        {var_name}[{idx0}] = solver.IntVar("
                        f"{lb_str}, {ub_str}, f'{var_name}_{{{idx0}}}')"
                    )
                else:
                    lines.append(
                        f"        {var_name}[{idx0}] = solver.NumVar("
                        f"{lb_str}, {ub_str}, f'{var_name}_{{{idx0}}}')"
                    )
            elif len(domain) == 2:
                idx0, idx1 = self._domain_idx_vars(domain, index_map)
                excl = meta.get("exclude_diagonal", False)
                df_cond = self._domain_filter_cond(meta, parameters, [idx0, idx1], domain)
                lines.append(f"    for {idx0} in {domain[0]}:")
                lines.append(f"        for {idx1} in {domain[1]}:")
                if excl:
                    lines.append(f"            if {idx0} == {idx1}: continue")
                if df_cond:
                    lines.append(f"            if not {df_cond}: continue")
                if vtype == "binary":
                    lines.append(
                        f"            {var_name}[({idx0}, {idx1})] = "
                        f"solver.BoolVar(f'{var_name}_{{{idx0}}}_{{{idx1}}}')"
                    )
                elif vtype == "integer":
                    lines.append(
                        f"            {var_name}[({idx0}, {idx1})] = solver.IntVar("
                        f"{lb_str}, {ub_str}, f'{var_name}_{{{idx0}}}_{{{idx1}}}')"
                    )
                else:
                    lines.append(
                        f"            {var_name}[({idx0}, {idx1})] = solver.NumVar("
                        f"{lb_str}, {ub_str}, f'{var_name}_{{{idx0}}}_{{{idx1}}}')"
                    )
            else:
                excl = meta.get("exclude_diagonal", False)
                idx_vars = self._domain_idx_vars(domain, index_map)
                for k, (iv, s) in enumerate(zip(idx_vars, domain)):
                    lines.append(f"    {'    ' * k}for {iv} in {s}:")
                skip_pad = "    " * (len(idx_vars) + 1)
                # Emit diagonal guard as a continue statement inside the innermost loop
                if excl and len(idx_vars) >= 2:
                    _da, _db = self._excl_diag_pair(domain, idx_vars)
                    lines.append(f"{skip_pad}if {_da} == {_db}: continue")
                df_cond = self._domain_filter_cond(meta, parameters, idx_vars, domain)
                if df_cond:
                    lines.append(f"{skip_pad}if not {df_cond}: continue")
                idx_tuple = "(" + ", ".join(idx_vars) + ")"
                name_str = "_".join(f"{{{iv}}}" for iv in idx_vars)
                inner = "    " * (len(domain) + 1)
                if vtype == "binary":
                    lines.append(
                        f"{inner}{var_name}[{idx_tuple}] = solver.BoolVar(f'{var_name}_{name_str}')"
                    )
                elif vtype == "integer":
                    lines.append(
                        f"{inner}{var_name}[{idx_tuple}] = solver.IntVar("
                        f"{lb_str}, {ub_str}, f'{var_name}_{name_str}')"
                    )
                else:
                    lines.append(
                        f"{inner}{var_name}[{idx_tuple}] = solver.NumVar("
                        f"{lb_str}, {ub_str}, f'{var_name}_{name_str}')"
                    )

        # Objective
        lines += [
            "",
            "    # --- Objective ---",
            "    objective = solver.Objective()",
        ]
        self._emit_ortools_coefficients(
            objective["expression"], "objective", index_map, variables, parameters, lines, indent=1
        )
        if sense == "minimize":
            lines.append("    objective.SetMinimization()")
        else:
            lines.append("    objective.SetMaximization()")

        # Constraints
        lines.append("")
        lines.append("    # --- Constraints ---")
        for cname, cmeta in constraints.items():
            domain = cmeta.get("domain", [])
            sense_c = cmeta.get("sense", "<=")
            rhs_node = cmeta["rhs"]
            domain_loop_vars = set(self._domain_idx_vars(domain, index_map)) if domain else set()
            lag_syms: dict[str, int] = {}
            lag_syms.update(self._collect_lag_symbols(cmeta["expression"], variables, parameters, sets))
            lag_syms.update(self._collect_lag_symbols(rhs_node, variables, parameters, sets))
            lag_ctx = self._build_lag_context(lag_syms, sets) if lag_syms else {}
            sparse_filter = cmeta.get("sparse_filter")
            sf_precompute, sf_guard = self._sparse_filter_guard(sparse_filter, parameters, index_map, domain)
            if sf_precompute:
                lines.append(sf_precompute)

            def _emit_ct_body(
                rhs_expr: str,
                ct_indent: int,
                name_expr: str,
                extra_known: set[str] | None = None,
                _lc: dict | None = None,
            ) -> None:
                pad = "    " * ct_indent
                if sense_c == "<=":
                    lines.append(
                        f"{pad}ct = solver.Constraint(-solver.infinity(), "
                        f"float({rhs_expr}), {name_expr})"
                    )
                elif sense_c == ">=":
                    lines.append(
                        f"{pad}ct = solver.Constraint("
                        f"float({rhs_expr}), solver.infinity(), {name_expr})"
                    )
                else:  # "="
                    lines.append(
                        f"{pad}ct = solver.Constraint("
                        f"float({rhs_expr}), float({rhs_expr}), {name_expr})"
                    )
                self._emit_ortools_coefficients(
                    cmeta["expression"],
                    "ct",
                    index_map,
                    variables,
                    parameters,
                    lines,
                    ct_indent,
                    extra_known=extra_known,
                    lag_context=_lc,
                )

            if not domain:
                rhs_expr = self._emit_expr(rhs_node, index_map, variables, parameters, lag_context=lag_ctx or None)
                _emit_ct_body(rhs_expr, ct_indent=1, name_expr=repr(cname), _lc=lag_ctx or None)
            elif len(domain) == 1:
                idx0 = index_map[domain[0]]
                if idx0 in lag_ctx:
                    _, _ev0 = lag_ctx[idx0]
                    _lv0 = lag_syms[idx0]
                    lines.append(f"    for {_ev0}, {idx0} in enumerate({domain[0]}):")
                    if _lv0 < 0:
                        lines.append(f"        if {_ev0} < {-_lv0}: continue")
                    else:
                        lines.append(f"        if {_ev0} + {_lv0} >= len({domain[0]}): continue")
                else:
                    lines.append(f"    for {idx0} in {domain[0]}:")
                if sf_guard:
                    lines.append(f"        if {sf_guard}: continue")
                rhs_expr = self._emit_expr(rhs_node, index_map, variables, parameters, extra_known=domain_loop_vars, lag_context=lag_ctx or None)
                _emit_ct_body(rhs_expr, ct_indent=2, name_expr=f"f\"{cname}_{{{idx0}}}\"", extra_known=domain_loop_vars, _lc=lag_ctx or None)
            elif len(domain) == 2:
                idx0, idx1 = self._domain_idx_vars(domain, index_map)
                if idx0 in lag_ctx:
                    _, _ev0 = lag_ctx[idx0]
                    _lv0 = lag_syms[idx0]
                    lines.append(f"    for {_ev0}, {idx0} in enumerate({domain[0]}):")
                    if _lv0 < 0:
                        lines.append(f"        if {_ev0} < {-_lv0}: continue")
                    else:
                        lines.append(f"        if {_ev0} + {_lv0} >= len({domain[0]}): continue")
                else:
                    lines.append(f"    for {idx0} in {domain[0]}:")
                if idx1 in lag_ctx:
                    _, _ev1 = lag_ctx[idx1]
                    _lv1 = lag_syms[idx1]
                    lines.append(f"        for {_ev1}, {idx1} in enumerate({domain[1]}):")
                    if _lv1 < 0:
                        lines.append(f"            if {_ev1} < {-_lv1}: continue")
                    else:
                        lines.append(f"            if {_ev1} + {_lv1} >= len({domain[1]}): continue")
                else:
                    lines.append(f"        for {idx1} in {domain[1]}:")
                guard = self._constraint_diagonal_guard(domain, [idx0, idx1])
                if guard:
                    lines.append(f"            if {guard}: continue")
                if sf_guard:
                    lines.append(f"            if {sf_guard}: continue")
                rhs_expr = self._emit_expr(rhs_node, index_map, variables, parameters, extra_known=domain_loop_vars, lag_context=lag_ctx or None)
                _emit_ct_body(
                    rhs_expr,
                    ct_indent=3,
                    name_expr=f"f\"{cname}_{{{idx0}}}_{{{idx1}}}\"",
                    extra_known=domain_loop_vars,
                    _lc=lag_ctx or None,
                )
            else:
                idx_vars = self._domain_idx_vars(domain, index_map)
                for k, (iv, s) in enumerate(zip(idx_vars, domain)):
                    if iv in lag_ctx:
                        _, _ev = lag_ctx[iv]
                        _lv = lag_syms[iv]
                        lines.append(f"    {'    ' * k}for {_ev}, {iv} in enumerate({s}):")
                        if _lv < 0:
                            lines.append(f"    {'    ' * (k + 1)}if {_ev} < {-_lv}: continue")
                        else:
                            lines.append(f"    {'    ' * (k + 1)}if {_ev} + {_lv} >= len({s}): continue")
                    else:
                        lines.append(f"    {'    ' * k}for {iv} in {s}:")
                name_parts = "\\x1f".join(f"{{{iv}}}" for iv in idx_vars)
                inner_pad = "    " * (len(domain) + 1)
                guard = self._constraint_diagonal_guard(domain, idx_vars)
                if guard:
                    lines.append(f"{inner_pad}if {guard}: continue")
                if sf_guard:
                    lines.append(f"{inner_pad}if {sf_guard}: continue")
                rhs_expr = self._emit_expr(rhs_node, index_map, variables, parameters, extra_known=domain_loop_vars, lag_context=lag_ctx or None)
                _emit_ct_body(
                    rhs_expr,
                    ct_indent=len(domain) + 1,
                    name_expr=f"f\"{cname}_{name_parts}\"",
                    extra_known=domain_loop_vars,
                    _lc=lag_ctx or None,
                )

        # LP export + solve + result
        lines += [
            "",
            "    # --- Solve ---",
            "    try:",
            "        with open('model.lp', 'w') as _lp_f:",
            "            _lp_f.write(solver.ExportModelAsLpFormat(False))",
            "    except Exception:",
            "        pass  # LP export is best-effort",
            "    if show_solver_log:",
            "        solver.EnableOutput()",
            "    else:",
            "        solver.SuppressOutput()",
            "    if time_limit is not None:",
            "        solver.set_time_limit(time_limit * 1000)  # OR-Tools expects milliseconds",
            "    _status_int = solver.Solve()",
            "    _status_map = {",
            "        pywraplp.Solver.OPTIMAL: 'optimal',",
            "        pywraplp.Solver.FEASIBLE: 'feasible',",
            "        pywraplp.Solver.INFEASIBLE: 'infeasible',",
            "        pywraplp.Solver.UNBOUNDED: 'unbounded',",
            "    }",
            "    result = {",
            "        'status': _status_map.get(_status_int, 'error'),",
            "        'objective_value': None,",
            "        'variables': {},",
            "        'variable_groups': [],",
            "    }",
            "    if _status_int in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):",
            "        result['objective_value'] = solver.Objective().Value()",
        ]

        for var_name, meta in variables.items():
            domain = meta.get("domain", [])
            excl = meta.get("exclude_diagonal", False)
            lines.append(f"    # extract {var_name}")
            if not domain:
                lines.append(
                    f"    result['variables'][{var_name!r}] = {var_name}.solution_value()"
                )
            elif len(domain) == 1:
                idx0 = index_map[domain[0]]
                df_cond = self._domain_filter_cond(meta, parameters, [idx0], domain)
                lines.append(f"    for {idx0} in {domain[0]}:")
                if df_cond:
                    lines.append(f"        if not {df_cond}: continue")
                lines.append(
                    f"        result['variables'][f\"{var_name}\\x1f{{{idx0}}}\"] = "
                    f"{var_name}[{idx0}].solution_value()"
                )
            elif len(domain) == 2:
                idx0, idx1 = self._domain_idx_vars(domain, index_map)
                df_cond = self._domain_filter_cond(meta, parameters, [idx0, idx1], domain)
                lines.append(f"    for {idx0} in {domain[0]}:")
                lines.append(f"        for {idx1} in {domain[1]}:")
                if excl:
                    lines.append(f"            if {idx0} == {idx1}: continue")
                if df_cond:
                    lines.append(f"            if not {df_cond}: continue")
                lines.append(
                    f"            result['variables'][f\"{var_name}\\x1f{{{idx0}}}\\x1f{{{idx1}}}\"] = "
                    f"{var_name}[({idx0}, {idx1})].solution_value()"
                )
            else:
                idx_vars = self._domain_idx_vars(domain, index_map)
                for k, (iv, s) in enumerate(zip(idx_vars, domain)):
                    lines.append(f"    {'    ' * k}for {iv} in {s}:")
                inner = "    " * (len(domain) + 1)
                if excl and len(idx_vars) >= 2:
                    _da, _db = self._excl_diag_pair(domain, idx_vars)
                    lines.append(f"{inner}if {_da} == {_db}: continue")
                df_cond = self._domain_filter_cond(meta, parameters, idx_vars, domain)
                if df_cond:
                    lines.append(f"{inner}if not {df_cond}: continue")
                idx_tuple = "(" + ", ".join(idx_vars) + ")"
                name_parts = "\\x1f".join(f"{{{iv}}}" for iv in idx_vars)
                lines.append(
                    f"{inner}result['variables'][f\"{var_name}\\x1f{name_parts}\"] = "
                    f"{var_name}[{idx_tuple}].solution_value()"
                )

        lines += self._emit_variable_groups(variables)
        lines.append("    return result")
        return "\n".join(lines) + "\n"
