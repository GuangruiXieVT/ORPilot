"""Deterministic IR → Python compiler (PuLP, Pyomo, OR-Tools backends)."""

from __future__ import annotations

from pathlib import Path


class IRCompiler:
    """Compiles a JSON IR dict into a solver-specific Python solve(data) function."""

    def compile(self, ir: dict, solver_framework: str = "pulp") -> str:
        if solver_framework == "pulp":
            return self._compile_pulp(ir)
        if solver_framework == "pyomo":
            return self._compile_pyomo(ir)
        if solver_framework in ("ortools", "or-tools"):
            return self._compile_ortools(ir)
        raise NotImplementedError(f"Solver framework '{solver_framework}' is not yet supported.")

    # ------------------------------------------------------------------
    # Shared helpers — set/parameter loading, variable groups
    # ------------------------------------------------------------------

    def _emit_set_loading(self, sets: dict, set_column: dict) -> list[str]:
        """Emit lines that load each set's members from data."""
        lines = []
        for set_name, meta in sets.items():
            source = meta.get("source")
            column = meta.get("column")
            table_stem = Path(source).stem if source else None

            if table_stem and column:
                lines.append(
                    f"    {set_name} = list(dict.fromkeys("
                    f"row[{column!r}] for row in data[{table_stem!r}]))"
                )
            elif table_stem:
                lines.append(
                    f"    {set_name} = list(dict.fromkeys("
                    f"next(iter(row.values())) for row in data[{table_stem!r}]))"
                )
            else:
                lines.append(
                    f"    {set_name} = []  # TODO: no source specified for set {set_name!r}"
                )
        return lines

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

            col_names = [set_column.get(s) or index_map.get(s, s.lower()) for s in domain]
            # Column that holds the parameter's value in the CSV.
            # Use the explicit "column" from the IR if present; fall back to param_name.
            value_col = meta.get("column") or param_name

            # Scalar parameter (no domain): read directly from the first CSV row.
            if not domain:
                lines.append(
                    f"    {param_name} = float(data[{table_stem!r}][0][{value_col!r}])"
                )
                continue

            lines.append(f"    {param_name} = {{}}")
            lines.append(f"    for _row in data[{table_stem!r}]:")

            if len(domain) == 1:
                c0 = col_names[0]
                lines.append(
                    f"        _key = _row.get({c0!r}) or next("
                    f"v for k, v in _row.items() if k != {value_col!r})"
                )
                lines.append(f"        {param_name}[_key] = float(_row[{value_col!r}])")
            elif len(domain) == 2:
                c0, c1 = col_names[0], col_names[1]
                lines.append(
                    f"        _k1 = _row.get({c0!r}) or _row.get({domain[0].lower()!r})"
                )
                lines.append(
                    f"        _k2 = _row.get({c1!r}) or _row.get({domain[1].lower()!r})"
                )
                lines.append(
                    f"        {param_name}[(_k1, _k2)] = float(_row[{value_col!r}])"
                )
            else:
                lines.append(f"        pass  # TODO: domain {domain!r} not supported")
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
                f"        if k.startswith({var_name!r} + '_')",
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

    def _var_ref(self, name: str, indices: list[str], domain: list[str]) -> str:
        """Return the dict-style variable reference used by PuLP and OR-Tools."""
        if not indices or not domain:
            return name
        if len(domain) == 1:
            return f"{name}[{indices[0]}]"
        idx_tuple = ", ".join(indices[: len(domain)])
        return f"{name}[({idx_tuple})]"

    def _emit_expr(
        self,
        node: dict,
        index_map: dict[str, str],
        variables: dict,
        parameters: dict,
    ) -> str:
        """Emit a Python expression string (PuLP lpSum / plain Python for OR-Tools RHS)."""
        node_type = node.get("type")
        operation = node.get("operation")

        if node_type == "constant":
            return str(node["value"])

        if node_type == "variable":
            name = node["name"]
            indices = node.get("indices", [])
            domain = variables.get(name, {}).get("domain", [])
            return self._var_ref(name, indices, domain)

        if node_type == "parameter":
            name = node["name"]
            indices = node.get("indices", [])
            domain = parameters.get(name, {}).get("domain", [])
            if not indices or not domain:
                return name
            if len(domain) == 1:
                return f"{name}[{indices[0]}]"
            idx_tuple = ", ".join(indices[: len(domain)])
            return f"{name}[({idx_tuple})]"

        if operation in ("sum", "subtract", "multiply"):
            left = self._emit_expr(node["left"], index_map, variables, parameters)
            right = self._emit_expr(node["right"], index_map, variables, parameters)
            if operation == "multiply":
                return f"{left} * {right}"
            op = "+" if operation == "sum" else "-"
            return f"({left} {op} {right})"

        if operation == "indexed_sum":
            over = node.get("over", [])
            body = self._emit_expr(node["body"], index_map, variables, parameters)
            iterators = " ".join(f"for {index_map[s]} in {s}" for s in over)
            return f"pulp.lpSum({body} {iterators})"

        return "0"

    def _emit_pyomo_expr(
        self,
        node: dict,
        index_map: dict[str, str],
        variables: dict,
        parameters: dict,
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

        if node_type == "variable":
            name = node["name"]
            indices = node.get("indices", [])
            domain = variables.get(name, {}).get("domain", [])
            if not indices or not domain:
                return f"model.{name}"
            if len(domain) == 1:
                return f"model.{name}[{indices[0]}]"
            idx_str = ", ".join(indices[: len(domain)])
            return f"model.{name}[{idx_str}]"

        if node_type == "parameter":
            name = node["name"]
            indices = node.get("indices", [])
            domain = parameters.get(name, {}).get("domain", [])
            if not indices or not domain:
                return name
            if len(domain) == 1:
                return f"{name}[{indices[0]}]"
            idx_tuple = ", ".join(indices[: len(domain)])
            return f"{name}[({idx_tuple})]"

        if operation in ("sum", "subtract", "multiply"):
            left = self._emit_pyomo_expr(node["left"], index_map, variables, parameters)
            right = self._emit_pyomo_expr(node["right"], index_map, variables, parameters)
            if operation == "multiply":
                return f"{left} * {right}"
            op = "+" if operation == "sum" else "-"
            return f"({left} {op} {right})"

        if operation == "indexed_sum":
            over = node.get("over", [])
            body = self._emit_pyomo_expr(node["body"], index_map, variables, parameters)
            iterators = " ".join(f"for {index_map[s]} in {s}" for s in over)
            return f"sum({body} {iterators})"

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
    ) -> None:
        """Append OR-Tools SetCoefficient calls to *lines* for a linear expression node.

        *target* is the Python name of the OR-Tools objective or constraint object.
        *indent* is the current indentation level (1 = 4 spaces = inside solve()).
        *sign* is +1 or -1, accumulated through subtract nodes.
        """
        pad = "    " * indent
        op = node.get("operation")
        ntype = node.get("type")

        if op == "indexed_sum":
            for s in node["over"]:
                pad = "    " * indent
                lines.append(f"{pad}for {index_map[s]} in {s}:")
                indent += 1
            self._emit_ortools_coefficients(
                node["body"], target, index_map, variables, parameters, lines, indent, sign
            )
            return

        if ntype == "variable":
            name = node["name"]
            indices = node.get("indices", [])
            domain = variables.get(name, {}).get("domain", [])
            var_ref = self._var_ref(name, indices, domain)
            coeff = "1.0" if sign == 1 else "-1.0"
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
            var_ref = self._var_ref(name, indices, domain)
            coeff_str = self._emit_expr(coeff_node, index_map, variables, parameters)
            if sign == -1:
                coeff_str = f"-({coeff_str})"
            lines.append(f"{pad}{target}.SetCoefficient({var_ref}, {coeff_str})")
            return

        if op == "sum":
            self._emit_ortools_coefficients(
                node["left"], target, index_map, variables, parameters, lines, indent, sign
            )
            self._emit_ortools_coefficients(
                node["right"], target, index_map, variables, parameters, lines, indent, sign
            )
            return

        if op == "subtract":
            self._emit_ortools_coefficients(
                node["left"], target, index_map, variables, parameters, lines, indent, sign
            )
            self._emit_ortools_coefficients(
                node["right"], target, index_map, variables, parameters, lines, indent, -sign
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
            "def solve(data: dict) -> dict:",
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
            ub_str = str(ub) if ub is not None else "None"

            if not domain:
                lines.append(
                    f"    {var_name} = pulp.LpVariable({var_name!r}, "
                    f"lowBound={lb_str}, upBound={ub_str}, cat={cat})"
                )
            elif len(domain) == 1:
                lines.append(
                    f"    {var_name} = pulp.LpVariable.dicts("
                    f"{var_name!r}, {domain[0]}, lowBound={lb_str}, upBound={ub_str}, cat={cat})"
                )
            elif len(domain) == 2:
                s0, s1 = domain[0], domain[1]
                idx0, idx1 = index_map[s0], index_map[s1]
                lines.append(
                    f"    {var_name} = pulp.LpVariable.dicts("
                    f"{var_name!r}, [({idx0}, {idx1}) for {idx0} in {s0} for {idx1} in {s1}], "
                    f"lowBound={lb_str}, upBound={ub_str}, cat={cat})"
                )
            else:
                lines.append(
                    f"    {var_name} = {{}}  # TODO: domain {domain!r} > 2 sets"
                )

        lines.append("")
        lines.append("    # --- Objective ---")
        obj_expr = self._emit_expr(objective["expression"], index_map, variables, parameters)
        lines.append(f"    prob += {obj_expr}, 'objective'")

        lines.append("")
        lines.append("    # --- Constraints ---")
        for cname, cmeta in constraints.items():
            domain = cmeta.get("domain", [])
            sense_op = {"<=": "<=", ">=": ">=", "=": "=="}.get(cmeta.get("sense", "<="), "<=")
            lhs = self._emit_expr(cmeta["expression"], index_map, variables, parameters)
            rhs = self._emit_expr(cmeta["rhs"], index_map, variables, parameters)

            if not domain:
                lines.append(f"    prob += {lhs} {sense_op} {rhs}, {cname!r}")
            elif len(domain) == 1:
                idx0 = index_map[domain[0]]
                lines.append(f"    for {idx0} in {domain[0]}:")
                lines.append(f"        prob += {lhs} {sense_op} {rhs}, f\"{cname}_{{{idx0}}}\"")
            elif len(domain) == 2:
                idx0, idx1 = index_map[domain[0]], index_map[domain[1]]
                lines.append(f"    for {idx0} in {domain[0]}:")
                lines.append(f"        for {idx1} in {domain[1]}:")
                lines.append(
                    f"            prob += {lhs} {sense_op} {rhs}, "
                    f"f\"{cname}_{{{idx0}}}_{{{idx1}}}\""
                )
            else:
                lines.append(f"    # TODO: constraint {cname!r} domain > 2 skipped")

        lines += [
            "",
            "    # --- Solve ---",
            "    prob.writeLP('model.lp')",
            "    prob.solve(pulp.PULP_CBC_CMD(msg=0))",
            "",
            "    status_map = {1: 'optimal', -1: 'infeasible', -2: 'unbounded', -3: 'error', 0: 'error'}",
            "    result = {",
            "        'status': status_map.get(prob.status, 'error'),",
            "        'objective_value': None,",
            "        'variables': {},",
            "        'variable_groups': [],",
            "    }",
            "    if prob.status == 1:",
            "        result['objective_value'] = pulp.value(prob.objective)",
        ]

        for var_name, meta in variables.items():
            domain = meta.get("domain", [])
            lines.append(f"    # extract {var_name}")
            if not domain:
                lines.append(
                    f"    result['variables'][{var_name!r}] = pulp.value({var_name})"
                )
            elif len(domain) == 1:
                idx0 = index_map[domain[0]]
                lines.append(f"    for {idx0} in {domain[0]}:")
                lines.append(
                    f"        result['variables'][f\"{var_name}_{{{idx0}}}\"] = "
                    f"pulp.value({var_name}[{idx0}])"
                )
            elif len(domain) == 2:
                idx0, idx1 = index_map[domain[0]], index_map[domain[1]]
                lines.append(f"    for {idx0} in {domain[0]}:")
                lines.append(f"        for {idx1} in {domain[1]}:")
                lines.append(
                    f"            result['variables'][f\"{var_name}_{{{idx0}}}_{{{idx1}}}\"] = "
                    f"pulp.value({var_name}[({idx0}, {idx1})])"
                )

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
            "def solve(data: dict) -> dict:",
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
            ub_str = str(ub) if ub is not None else "None"

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
                lines.append(
                    f"    model.{var_name} = pyo.Var({domain[0]}, within={within}{bounds})"
                )
            elif len(domain) == 2:
                lines.append(
                    f"    model.{var_name} = pyo.Var("
                    f"{domain[0]}, {domain[1]}, within={within}{bounds})"
                )
            else:
                lines.append(
                    f"    # TODO: domain {domain!r} > 2 sets not compiled for {var_name}"
                )

        lines.append("")
        lines.append("    # --- Objective ---")
        obj_expr = self._emit_pyomo_expr(
            objective["expression"], index_map, variables, parameters
        )
        lines.append(f"    model.obj = pyo.Objective(expr={obj_expr}, sense={pyo_sense})")

        lines.append("")
        lines.append("    # --- Constraints ---")
        for cname, cmeta in constraints.items():
            domain = cmeta.get("domain", [])
            sense_op = {"<=": "<=", ">=": ">=", "=": "=="}.get(cmeta.get("sense", "<="), "<=")
            lhs = self._emit_pyomo_expr(cmeta["expression"], index_map, variables, parameters)
            rhs = self._emit_pyomo_expr(cmeta["rhs"], index_map, variables, parameters)

            lines.append(f"    model.{cname} = pyo.ConstraintList()")
            if not domain:
                lines.append(f"    model.{cname}.add({lhs} {sense_op} {rhs})")
            elif len(domain) == 1:
                idx0 = index_map[domain[0]]
                lines.append(f"    for {idx0} in {domain[0]}:")
                lines.append(f"        model.{cname}.add({lhs} {sense_op} {rhs})")
            elif len(domain) == 2:
                idx0, idx1 = index_map[domain[0]], index_map[domain[1]]
                lines.append(f"    for {idx0} in {domain[0]}:")
                lines.append(f"        for {idx1} in {domain[1]}:")
                lines.append(f"            model.{cname}.add({lhs} {sense_op} {rhs})")
            else:
                lines.append(f"    # TODO: constraint {cname!r} domain > 2 skipped")

        lines += [
            "",
            "    # --- Solve ---",
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
            "    _results = _solver.solve(model, tee=False)",
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
            lines.append(f"    # extract {var_name}")
            if not domain:
                lines.append(
                    f"    result['variables'][{var_name!r}] = pyo.value(model.{var_name})"
                )
            elif len(domain) == 1:
                idx0 = index_map[domain[0]]
                lines.append(f"    for {idx0} in {domain[0]}:")
                lines.append(
                    f"        result['variables'][f\"{var_name}_{{{idx0}}}\"] = "
                    f"pyo.value(model.{var_name}[{idx0}])"
                )
            elif len(domain) == 2:
                idx0, idx1 = index_map[domain[0]], index_map[domain[1]]
                lines.append(f"    for {idx0} in {domain[0]}:")
                lines.append(f"        for {idx1} in {domain[1]}:")
                lines.append(
                    f"            result['variables'][f\"{var_name}_{{{idx0}}}_{{{idx1}}}\"] = "
                    f"pyo.value(model.{var_name}[{idx0}, {idx1}])"
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
            "def solve(data: dict) -> dict:",
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
            else:
                ub_str = "solver.infinity()" if vtype == "continuous" else "int(1e9)"

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
                lines.append(f"    for {idx0} in {domain[0]}:")
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
                idx0, idx1 = index_map[domain[0]], index_map[domain[1]]
                lines.append(f"    for {idx0} in {domain[0]}:")
                lines.append(f"        for {idx1} in {domain[1]}:")
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
                lines.append(
                    f"    # TODO: domain {domain!r} > 2 sets not compiled for {var_name}"
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

            # Helper to emit the constraint body (ct declaration + SetCoefficient calls)
            # at a given indent level, with rhs already in scope as a Python expression.
            def _emit_ct_body(rhs_expr: str, ct_indent: int, name_expr: str) -> None:
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
                )

            if not domain:
                rhs_expr = self._emit_expr(rhs_node, index_map, variables, parameters)
                _emit_ct_body(rhs_expr, ct_indent=1, name_expr=repr(cname))
            elif len(domain) == 1:
                idx0 = index_map[domain[0]]
                lines.append(f"    for {idx0} in {domain[0]}:")
                rhs_expr = self._emit_expr(rhs_node, index_map, variables, parameters)
                _emit_ct_body(rhs_expr, ct_indent=2, name_expr=f"f\"{cname}_{{{idx0}}}\"")
            elif len(domain) == 2:
                idx0, idx1 = index_map[domain[0]], index_map[domain[1]]
                lines.append(f"    for {idx0} in {domain[0]}:")
                lines.append(f"        for {idx1} in {domain[1]}:")
                rhs_expr = self._emit_expr(rhs_node, index_map, variables, parameters)
                _emit_ct_body(
                    rhs_expr,
                    ct_indent=3,
                    name_expr=f"f\"{cname}_{{{idx0}}}_{{{idx1}}}\"",
                )
            else:
                lines.append(f"    # TODO: constraint {cname!r} domain > 2 skipped")

        # LP export + solve + result
        lines += [
            "",
            "    # --- Solve ---",
            "    try:",
            "        with open('model.lp', 'w') as _lp_f:",
            "            _lp_f.write(solver.ExportModelAsLpFormat(False))",
            "    except Exception:",
            "        pass  # LP export is best-effort",
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
            lines.append(f"    # extract {var_name}")
            if not domain:
                lines.append(
                    f"    result['variables'][{var_name!r}] = {var_name}.solution_value()"
                )
            elif len(domain) == 1:
                idx0 = index_map[domain[0]]
                lines.append(f"    for {idx0} in {domain[0]}:")
                lines.append(
                    f"        result['variables'][f\"{var_name}_{{{idx0}}}\"] = "
                    f"{var_name}[{idx0}].solution_value()"
                )
            elif len(domain) == 2:
                idx0, idx1 = index_map[domain[0]], index_map[domain[1]]
                lines.append(f"    for {idx0} in {domain[0]}:")
                lines.append(f"        for {idx1} in {domain[1]}:")
                lines.append(
                    f"            result['variables'][f\"{var_name}_{{{idx0}}}_{{{idx1}}}\"] = "
                    f"{var_name}[({idx0}, {idx1})].solution_value()"
                )

        lines += self._emit_variable_groups(variables)
        lines.append("    return result")
        return "\n".join(lines) + "\n"
