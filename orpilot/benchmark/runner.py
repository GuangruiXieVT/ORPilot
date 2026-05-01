"""BenchmarkRunner — orchestrates the three benchmark modes."""

from __future__ import annotations

import csv
import time
import tempfile
import traceback
from pathlib import Path
from typing import Any

from orpilot.benchmark.case import BenchmarkCase, BenchmarkResult
from orpilot.benchmark.ingestor import TextIngestor
from orpilot.codegen.executor import CodeExecutor
from orpilot.codegen.ir_compiler import IRCompiler
from orpilot.llm.base import BaseLLM
from orpilot.models.data import CsvColumnSpec, CsvFileSpec, UserData
from orpilot.models.problem import ProblemDefinition
from orpilot.workflow.nodes.direct_code_gen import direct_code_gen_node
from orpilot.workflow.nodes.ir_builder import ir_builder_node, ir_builder_on_demand_node
from orpilot.workflow.nodes.param_computation import param_computation_node
from orpilot.workflow.state import WorkflowState


def _tables_to_data(tables: dict[str, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    """Pass tables through as-is (already in raw_tables format)."""
    return dict(tables)


def _infer_csv_specs(tables: dict[str, list[dict[str, Any]]]) -> list[CsvFileSpec]:
    """Build minimal CsvFileSpec objects from table data (no type coercion needed)."""
    specs = []
    for stem, rows in tables.items():
        if not rows:
            specs.append(CsvFileSpec(filename=f"{stem}.csv"))
            continue
        cols = [
            CsvColumnSpec(
                name=col,
                dtype="float" if isinstance(val, float) else ("int" if isinstance(val, int) else "str"),
            )
            for col, val in rows[0].items()
        ]
        specs.append(CsvFileSpec(filename=f"{stem}.csv", columns=cols))
    return specs


def _write_tables_to_dir(tables: dict[str, list[dict[str, Any]]], directory: Path) -> None:
    """Write each table as a CSV file under *directory*."""
    for stem, rows in tables.items():
        if not rows:
            continue
        path = directory / f"{stem}.csv"
        all_keys = list(dict.fromkeys(k for row in rows for k in row))
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=all_keys, extrasaction="ignore", restval="")
            writer.writeheader()
            writer.writerows(rows)


def _extract_result(exec_result: dict) -> tuple[str, float | None, str | None, str]:
    """Extract (status, objective_value, error, lp_content) from CodeExecutor output."""
    lp_content = exec_result.get("lp_content") or ""
    error = exec_result.get("error")
    if error:
        return "error", None, error, lp_content
    result = exec_result.get("result") or {}
    status = result.get("status", "unknown")
    obj = result.get("objective_value")
    if obj is not None:
        try:
            obj = float(obj)
        except (TypeError, ValueError):
            obj = None
    # The executor wrapper serialises exceptions as {"status": "error", "error": "..."}
    # inside the result dict — surface that message so callers can report it.
    err_msg = result.get("error") if status == "error" else None
    return status, obj, err_msg, lp_content


class BenchmarkRunner:
    """Run benchmark cases through the ORPilot pipeline."""

    def __init__(self, timeout: int = 120) -> None:
        self.timeout = timeout

    # ------------------------------------------------------------------
    # Mode C — compiler only (no LLM)
    # ------------------------------------------------------------------

    def run_compiler_only(
        self,
        case: BenchmarkCase,
        ir_model: dict,
        tables: dict[str, list[dict[str, Any]]],
        solver: str = "pulp",
    ) -> BenchmarkResult:
        """Compile *ir_model* and execute; no LLM calls."""
        start = time.monotonic()
        try:
            code = IRCompiler().compile(ir_model, solver)
        except Exception as exc:
            return BenchmarkResult(
                case_name=case.name,
                solver=solver,
                mode="compiler",
                status="error",
                objective_value=None,
                expected_objective=case.expected_objective,
                objective_tolerance=case.objective_tolerance,
                ir_model=ir_model,
                error=f"Compilation error: {exc}",
                solve_time=time.monotonic() - start,
            )

        data = _tables_to_data(tables)
        exec_result = CodeExecutor(timeout=self.timeout).execute(code, data)
        elapsed = time.monotonic() - start
        status, obj, err, lp_content = _extract_result(exec_result)
        return BenchmarkResult(
            case_name=case.name,
            solver=solver,
            mode="compiler",
            status=status,
            objective_value=obj,
            expected_objective=case.expected_objective,
            objective_tolerance=case.objective_tolerance,
            ir_model=ir_model,
            generated_code=code,
            lp_content=lp_content,
            error=err,
            solve_time=elapsed,
        )

    # ------------------------------------------------------------------
    # Mode B — IR builder (1 LLM call)
    # ------------------------------------------------------------------

    def run_with_ir_builder(
        self,
        case: BenchmarkCase,
        tables: dict[str, list[dict[str, Any]]],
        llm: BaseLLM,
        solver: str = "pulp",
        max_retries: int = 3,
        generate_ir: bool = False,
    ) -> BenchmarkResult:
        """Build IR with LLM from pre-extracted tables, then compile and run.

        Retries up to *max_retries* times when compilation fails or the solver
        returns error, infeasible, or unbounded — feeding the error back to the
        IR builder so it can correct the model.
        """
        start = time.monotonic()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            _write_tables_to_dir(tables, tmpdir_path)

            csv_specs = _infer_csv_specs(tables)
            csv_file_paths = {
                Path(spec.filename).stem: str(tmpdir_path / spec.filename)
                for spec in csv_specs
            }

            problem = ProblemDefinition(
                description=case.problem_text,
                csv_file_paths=csv_file_paths,
            )
            user_data = UserData(
                raw_tables=tables,
                csv_specs=csv_specs,
                csv_dir=str(tmpdir_path),
            )

            state: WorkflowState = {
                "messages": [],
                "problem": problem,
                "user_data": user_data,
                "ir_model": None,
                "generated_code": "",
                "solution": None,
                "report": "",
                "current_node": "ir_builder",
                "solver_name": solver,
                "retry_count": 0,
                "max_retries": max_retries,
                "error_context": "",
                "needs_user_input": False,
                "user_input": "",
                "llm_config": {},
                "data_dir": str(tmpdir_path),
                "csv_specs": csv_specs,
                "output_dir": "",
                "solver_time_limit": None,
                "show_solver_log": False,
            }

            # param_computation runs once before the retry loop.
            state = param_computation_node(state, llm)
            updated_user_data = state.get("user_data")
            data = _tables_to_data(
                updated_user_data.raw_tables if updated_user_data is not None else tables
            )

            ir_model: dict | None = None
            code: str = ""
            status, obj, err, lp_content = "error", None, "No IR produced", ""

            for attempt in range(max_retries + 1):
                # ir_builder_node uses error_context + ir_model already in state
                # to give the LLM error feedback on retries.
                state = ir_builder_node(state, llm)
                ir_model = state.get("ir_model")
                if not ir_model:
                    err = "IR builder returned no model"
                    break

                # Compile IR → solver code
                try:
                    code = IRCompiler().compile(ir_model, solver)
                    state = {**state, "generated_code": code, "error_context": ""}
                except Exception:
                    error_msg = "IR compilation failed:\n" + traceback.format_exc()
                    if attempt < max_retries:
                        state = {
                            **state,
                            "error_context": error_msg,
                            "retry_count": attempt + 1,
                        }
                        continue
                    status, obj, err = "error", None, error_msg
                    break

                # Execute and check result
                exec_result = CodeExecutor(timeout=self.timeout).execute(code, data)
                status, obj, err, lp_content = _extract_result(exec_result)

                if status in ("optimal", "feasible"):
                    break

                if attempt < max_retries:
                    if status == "unbounded":
                        error_detail = (
                            "The model is unbounded — the objective can grow to infinity. "
                            "A variable or combination of variables is unconstrained in the "
                            "objective direction. Check that all variables are bounded by "
                            "constraints (e.g. warehouse capacity limits purchases, demand "
                            "limits production). Add any missing upper-bound constraints."
                        )
                    else:
                        error_detail = err or f"status={status}"
                    lp_snippet = f"\n\nLP file:\n{lp_content[:2000]}" if lp_content else ""
                    state = {
                        **state,
                        "error_context": (
                            f"Solve failed with status={status}. "
                            f"Error: {error_detail}{lp_snippet}"
                        ),
                        "generated_code": code,
                        "retry_count": attempt + 1,
                    }

        elapsed = time.monotonic() - start
        return BenchmarkResult(
            case_name=case.name,
            solver=solver,
            mode="ir_builder",
            status=status,
            objective_value=obj,
            expected_objective=case.expected_objective,
            objective_tolerance=case.objective_tolerance,
            ir_model=ir_model,
            generated_code=code,
            lp_content=lp_content,
            error=err,
            solve_time=elapsed,
        )

    # ------------------------------------------------------------------
    # Mode D — direct code gen (1 LLM call, no IR)
    # ------------------------------------------------------------------

    def run_direct_code_gen(
        self,
        case: BenchmarkCase,
        tables: dict[str, list[dict[str, Any]]],
        llm: BaseLLM,
        solver: str = "pulp",
        max_retries: int = 3,
        generate_ir: bool = False,
    ) -> BenchmarkResult:
        """Generate solver code directly with LLM from pre-extracted tables and run it.

        Retries up to *max_retries* times when execution fails, feeding the
        error back to the LLM for correction.
        If *generate_ir* is True, generates an IR blueprint after a successful solve.
        """
        start = time.monotonic()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            _write_tables_to_dir(tables, tmpdir_path)

            csv_specs = _infer_csv_specs(tables)
            csv_file_paths = {
                Path(spec.filename).stem: str(tmpdir_path / spec.filename)
                for spec in csv_specs
            }

            problem = ProblemDefinition(
                description=case.problem_text,
                csv_file_paths=csv_file_paths,
            )
            user_data = UserData(
                raw_tables=tables,
                csv_specs=csv_specs,
                csv_dir=str(tmpdir_path),
            )

            state: WorkflowState = {
                "messages": [],
                "problem": problem,
                "user_data": user_data,
                "ir_model": None,
                "generated_code": "",
                "solution": None,
                "report": "",
                "current_node": "direct_code_gen",
                "solver_name": solver,
                "retry_count": 0,
                "max_retries": max_retries,
                "error_context": "",
                "needs_user_input": False,
                "user_input": "",
                "llm_config": {},
                "data_dir": str(tmpdir_path),
                "csv_specs": csv_specs,
                "output_dir": "",
                "solver_time_limit": None,
                "show_solver_log": False,
                "use_ir": False,
            }

            state = param_computation_node(state, llm)
            updated_user_data = state.get("user_data")
            data = _tables_to_data(
                updated_user_data.raw_tables if updated_user_data is not None else tables
            )

            code: str = ""
            status, obj, err, lp_content = "error", None, "No code produced", ""

            for attempt in range(max_retries + 1):
                state = direct_code_gen_node(state, llm)

                # If node short-circuited to reporter, give up
                if state.get("current_node") == "reporter":
                    err = state.get("report", "Direct code gen failed")
                    break

                code = state.get("generated_code", "")
                if not code:
                    err = "Direct code gen returned no code"
                    break

                # Execute and check result
                exec_result = CodeExecutor(timeout=self.timeout).execute(code, data)
                status, obj, err, lp_content = _extract_result(exec_result)

                if status in ("optimal", "feasible"):
                    break

                if attempt < max_retries:
                    if status == "unbounded":
                        error_detail = (
                            "The model is unbounded — the objective can grow to infinity. "
                            "A variable or combination of variables is unconstrained in the "
                            "objective direction. Check that all variables are bounded by "
                            "constraints (e.g. warehouse capacity limits purchases, demand "
                            "limits production). Add any missing upper-bound constraints."
                        )
                    else:
                        error_detail = err or f"status={status}"
                    lp_snippet = f"\n\nLP file:\n{lp_content[:2000]}" if lp_content else ""
                    state = {
                        **state,
                        "error_context": (
                            f"Solve failed with status={status}. "
                            f"Error: {error_detail}{lp_snippet}"
                        ),
                        "generated_code": code,
                        "retry_count": attempt + 1,
                    }

        # Optionally generate IR blueprint after a successful solve
        ir_model = None
        if generate_ir and status in ("optimal", "feasible") and code:
            ir_state = ir_builder_on_demand_node({**state, "generated_code": code}, llm)
            ir_model = ir_state.get("ir_model")

        elapsed = time.monotonic() - start
        return BenchmarkResult(
            case_name=case.name,
            solver=solver,
            mode="direct_code_gen",
            status=status,
            objective_value=obj,
            expected_objective=case.expected_objective,
            objective_tolerance=case.objective_tolerance,
            ir_model=ir_model,
            generated_code=code,
            lp_content=lp_content,
            error=err,
            solve_time=elapsed,
        )

    # ------------------------------------------------------------------
    # Mode D-full — direct code gen + ingestor (2 LLM calls, no IR)
    # ------------------------------------------------------------------

    def run_direct_pipeline(
        self,
        case: BenchmarkCase,
        llm: BaseLLM,
        solver: str = "pulp",
        generate_ir: bool = False,
    ) -> BenchmarkResult:
        """Ingest text with LLM to extract tables, then generate solver code directly.

        Use this when tables are not pre-extracted (e.g. HuggingFace dataset cases).
        Mirrors run_full_pipeline but calls run_direct_code_gen instead of
        run_with_ir_builder.
        """
        start = time.monotonic()

        try:
            _problem_def, tables = TextIngestor().ingest(case.problem_text, llm)
        except Exception as exc:
            return BenchmarkResult(
                case_name=case.name,
                solver=solver,
                mode="direct_pipeline",
                status="error",
                objective_value=None,
                expected_objective=case.expected_objective,
                objective_tolerance=case.objective_tolerance,
                error=f"Ingestor error: {exc}",
                solve_time=time.monotonic() - start,
            )

        result = self.run_direct_code_gen(
            BenchmarkCase(
                name=case.name,
                problem_text=case.problem_text,
                tables=tables,
                expected_objective=case.expected_objective,
                expected_status=case.expected_status,
                objective_tolerance=case.objective_tolerance,
                source=case.source,
                tags=case.tags,
            ),
            tables=tables,
            llm=llm,
            solver=solver,
            generate_ir=generate_ir,
        )
        result.solve_time = time.monotonic() - start
        result.mode = "direct_pipeline"
        result.tables = tables
        return result

    # ------------------------------------------------------------------
    # Mode A — full pipeline (2 LLM calls)
    # ------------------------------------------------------------------

    def run_full_pipeline(
        self,
        case: BenchmarkCase,
        llm: BaseLLM,
        solver: str = "pulp",
    ) -> BenchmarkResult:
        """Ingest text with LLM, build IR with LLM, compile, and run."""
        start = time.monotonic()
        try:
            _problem_def, tables = TextIngestor().ingest(case.problem_text, llm)
        except Exception as exc:
            return BenchmarkResult(
                case_name=case.name,
                solver=solver,
                mode="full",
                status="error",
                objective_value=None,
                expected_objective=case.expected_objective,
                objective_tolerance=case.objective_tolerance,
                error=f"Ingestor error: {exc}",
                solve_time=time.monotonic() - start,
            )

        result = self.run_with_ir_builder(
            BenchmarkCase(
                name=case.name,
                problem_text=case.problem_text,
                tables=tables,
                expected_objective=case.expected_objective,
                expected_status=case.expected_status,
                objective_tolerance=case.objective_tolerance,
                source=case.source,
                tags=case.tags,
            ),
            tables=tables,
            llm=llm,
            solver=solver,
        )
        # Adjust elapsed time to include ingestor time
        result.solve_time = time.monotonic() - start
        result.mode = "full"
        result.tables = tables
        return result

    # ------------------------------------------------------------------
    # Convenience method — auto-select mode
    # ------------------------------------------------------------------

    def run(
        self,
        case: BenchmarkCase,
        llm: BaseLLM | None = None,
        solver: str = "pulp",
    ) -> BenchmarkResult:
        """Run the case, automatically choosing the mode based on available artifacts.

        - Mode C (compiler only): ``case.ir_model`` and ``case.tables`` are set
        - Mode B (IR builder):    only ``case.tables`` is set
        - Mode A (full pipeline): neither is set (requires LLM)
        """
        if case.ir_model is not None and case.tables is not None:
            return self.run_compiler_only(case, case.ir_model, case.tables, solver)
        if case.tables is not None:
            if llm is None:
                raise ValueError("Mode B (run_with_ir_builder) requires an LLM instance.")
            return self.run_with_ir_builder(case, case.tables, llm, solver)
        if llm is None:
            raise ValueError("Mode A (run_full_pipeline) requires an LLM instance.")
        return self.run_full_pipeline(case, llm, solver)
