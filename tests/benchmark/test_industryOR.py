"""IndustryOR HuggingFace dataset benchmark tests — require a live LLM API key."""

from __future__ import annotations

import datetime
import json
import logging
import time
from pathlib import Path

import pytest

from orpilot.benchmark.runner import BenchmarkRunner

log = logging.getLogger(__name__)


def _save_artifacts(result, case, out_dir: Path) -> None:
    """Write ir.json, model.py, and model.lp for a case into out_dir/<case_name>/."""
    case_dir = out_dir / case.name
    case_dir.mkdir(parents=True, exist_ok=True)
    if result.ir_model:
        (case_dir / "ir.json").write_text(json.dumps(result.ir_model, indent=2), encoding="utf-8")
    if result.generated_code:
        (case_dir / "model.py").write_text(result.generated_code, encoding="utf-8")
    if result.lp_content:
        (case_dir / "model.lp").write_text(result.lp_content, encoding="utf-8")


@pytest.mark.llm
@pytest.mark.industryOR
def test_industryOR_sample(llm_fixture, save_dir, generate_ir, difficulty, limit, solver):
    """Load IndustryOR cases and run the direct code gen pipeline for each.

    Pass --difficulty Easy|Medium|Hard to select the difficulty tier (default: Easy).
    Pass --limit N to cap the number of cases (default: all).
    Pass --generate-ir to also produce a solver-agnostic IR blueprint after each
    successful solve (or set generate_ir=true in orpilot.toml).
    """
    pytest.importorskip("datasets", reason="pip install 'orpilot[hf]' to run IndustryOR tests")

    from orpilot.benchmark.loader_hf import load_hf_cases

    cases = load_hf_cases("CardinalOperations/IndustryOR", difficulty=difficulty, limit=limit)
    assert cases, f"No {difficulty} cases loaded from CardinalOperations/IndustryOR"

    runner = BenchmarkRunner(timeout=180)
    passed = 0
    failures: list[str] = []
    solve_times: list[float] = []
    results: list = []

    mode_label = "direct+ir" if generate_ir else "direct"
    log.info("Running %d IndustryOR %s cases (mode=%s)", len(cases), difficulty, mode_label)
    suite_start = time.monotonic()
    for i, case in enumerate(cases, 1):
        log.info("[%d/%d] %s (expected %s) ...", i, len(cases), case.name, case.expected_objective)

        # Reset token counters before each case so we capture per-case usage
        if hasattr(llm_fixture, "reset_usage"):
            llm_fixture.reset_usage()

        result = runner.run_direct_pipeline(case, llm_fixture, solver=solver, generate_ir=generate_ir)

        # Attach per-case token usage
        if hasattr(llm_fixture, "get_usage"):
            usage = llm_fixture.get_usage()
            result.metrics = {
                "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
                "latency_s": round(result.solve_time or 0.0, 2),
            }

        results.append(result)
        solve_times.append(result.solve_time)
        if save_dir:
            _save_artifacts(result, case, save_dir)
        if result.passed:
            passed += 1
            log.info("  PASS  (obj=%s, time=%.1fs)", result.objective_value, result.solve_time)
        else:
            msg = (
                f"status={result.status}, "
                f"obj={result.objective_value} (expected {case.expected_objective}), "
                f"error={result.error}"
            )
            log.info("  FAIL  (%s)", msg)
            failures.append(f"{case.name}: {msg}")
        print(f"[{i}/{len(cases)}] {'PASS' if result.passed else 'FAIL'}  passed={passed}  failed={len(failures)}  ({result.solve_time:.1f}s)", flush=True)

    total = len(cases)
    pct = 100.0 * passed / total if total else 0.0
    total_wall = time.monotonic() - suite_start
    avg_t = sum(solve_times) / len(solve_times) if solve_times else 0.0
    min_t = min(solve_times) if solve_times else 0.0
    max_t = max(solve_times) if solve_times else 0.0
    log.info("=" * 60)
    log.info("  RESULT: %d/%d passed (%.1f%%)", passed, total, pct)
    log.info("  TIME:   total=%.1fs  avg=%.1fs  min=%.1fs  max=%.1fs", total_wall, avg_t, min_t, max_t)
    if failures:
        log.info("  FAILED cases:")
        for f in failures:
            log.info("    - %s", f)
    log.info("=" * 60)

    # Write aggregate metrics.json (one file per batch run, not per case)
    if save_dir and results:
        total_input = sum(r.metrics.get("input_tokens", 0) for r in results)
        total_output = sum(r.metrics.get("output_tokens", 0) for r in results)
        aggregate = {
            "run_id": datetime.datetime.now().isoformat(timespec="seconds"),
            "solver": solver,
            "difficulty": difficulty,
            "total_cases": total,
            "passed": passed,
            "failed": total - passed,
            "totals": {
                "input_tokens": total_input,
                "output_tokens": total_output,
                "latency_s": round(total_wall, 2),
            },
            "cases": {
                r.case_name: {
                    "status": r.status,
                    "passed": r.passed,
                    "input_tokens": r.metrics.get("input_tokens", 0),
                    "output_tokens": r.metrics.get("output_tokens", 0),
                    "latency_s": r.metrics.get("latency_s", 0.0),
                    "expected_objective": r.expected_objective,
                    "validation_attempts": r.validation_attempts,
                    "first_validated_objective": r.first_validated_objective,
                    "code_gen_retries": r.code_gen_retries,
                    "validator_inconclusive": r.validator_inconclusive,
                }
                for r in results
            },
        }
        (save_dir / "metrics.json").write_text(
            json.dumps(aggregate, indent=2), encoding="utf-8"
        )
        log.info("  Wrote aggregate metrics to %s/metrics.json", save_dir)

    assert not failures, f"{len(failures)}/{total} IndustryOR cases failed:\n" + "\n".join(failures)


@pytest.mark.llm
@pytest.mark.industryOR
def test_industryOR_ir_pipeline(llm_fixture, save_dir, difficulty, limit, solver, embed_api_key, embed_model):
    """Run IndustryOR cases through the IR builder pipeline with RAG and solution validation.

    Pipeline: ingest → IR builder (RAG-augmented) → compile → solve → solution validator.
    The solution validator runs after each successful solve and feeds back to the IR builder
    when it detects a semantic error (e.g. a one-time cost charged per period).

    Pass --difficulty Easy|Medium|Hard to select the difficulty tier (default: Easy).
    Pass --limit N to cap the number of cases.
    """
    pytest.importorskip("datasets", reason="pip install 'orpilot[hf]' to run IndustryOR tests")

    from orpilot.benchmark.loader_hf import load_hf_cases

    cases = load_hf_cases("CardinalOperations/IndustryOR", difficulty=difficulty, limit=limit)
    assert cases, f"No {difficulty} cases loaded from CardinalOperations/IndustryOR"

    runner = BenchmarkRunner(timeout=180)
    passed = 0
    failures: list[str] = []
    solve_times: list[float] = []
    results: list = []

    log.info(
        "Running %d IndustryOR %s cases (mode=ir_pipeline, rag=%s)",
        len(cases), difficulty, "yes" if embed_api_key else "no",
    )
    suite_start = time.monotonic()
    for i, case in enumerate(cases, 1):
        log.info("[%d/%d] %s (expected %s) ...", i, len(cases), case.name, case.expected_objective)

        if hasattr(llm_fixture, "reset_usage"):
            llm_fixture.reset_usage()

        result = runner.run_ir_pipeline(
            case,
            llm_fixture,
            solver=solver,
            embed_api_key=embed_api_key,
            embed_model=embed_model,
        )

        if hasattr(llm_fixture, "get_usage"):
            usage = llm_fixture.get_usage()
            result.metrics = {
                "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
                "latency_s": round(result.solve_time or 0.0, 2),
            }

        results.append(result)
        solve_times.append(result.solve_time)
        if save_dir:
            _save_artifacts(result, case, save_dir)
        if result.passed:
            passed += 1
            log.info("  PASS  (obj=%s, time=%.1fs)", result.objective_value, result.solve_time)
        else:
            msg = (
                f"status={result.status}, "
                f"obj={result.objective_value} (expected {case.expected_objective}), "
                f"error={result.error}"
            )
            log.info("  FAIL  (%s)", msg)
            failures.append(f"{case.name}: {msg}")
        print(f"[{i}/{len(cases)}] {'PASS' if result.passed else 'FAIL'}  passed={passed}  failed={len(failures)}  ({result.solve_time:.1f}s)", flush=True)

    total = len(cases)
    pct = 100.0 * passed / total if total else 0.0
    total_wall = time.monotonic() - suite_start
    avg_t = sum(solve_times) / len(solve_times) if solve_times else 0.0
    min_t = min(solve_times) if solve_times else 0.0
    max_t = max(solve_times) if solve_times else 0.0
    log.info("=" * 60)
    log.info("  RESULT: %d/%d passed (%.1f%%)", passed, total, pct)
    log.info("  TIME:   total=%.1fs  avg=%.1fs  min=%.1fs  max=%.1fs", total_wall, avg_t, min_t, max_t)
    if failures:
        log.info("  FAILED cases:")
        for f in failures:
            log.info("    - %s", f)
    log.info("=" * 60)

    if save_dir and results:
        total_input = sum(r.metrics.get("input_tokens", 0) for r in results)
        total_output = sum(r.metrics.get("output_tokens", 0) for r in results)
        aggregate = {
            "run_id": datetime.datetime.now().isoformat(timespec="seconds"),
            "solver": solver,
            "difficulty": difficulty,
            "mode": "ir_pipeline",
            "total_cases": total,
            "passed": passed,
            "failed": total - passed,
            "totals": {
                "input_tokens": total_input,
                "output_tokens": total_output,
                "latency_s": round(total_wall, 2),
            },
            "cases": {
                r.case_name: {
                    "status": r.status,
                    "passed": r.passed,
                    "input_tokens": r.metrics.get("input_tokens", 0),
                    "output_tokens": r.metrics.get("output_tokens", 0),
                    "latency_s": r.metrics.get("latency_s", 0.0),
                    "expected_objective": r.expected_objective,
                    "validation_attempts": r.validation_attempts,
                    "first_validated_objective": r.first_validated_objective,
                    "code_gen_retries": r.code_gen_retries,
                    "validator_inconclusive": r.validator_inconclusive,
                }
                for r in results
            },
        }
        (save_dir / "metrics_ir_pipeline.json").write_text(
            json.dumps(aggregate, indent=2), encoding="utf-8"
        )
        log.info("  Wrote aggregate metrics to %s/metrics_ir_pipeline.json", save_dir)

    assert not failures, f"{len(failures)}/{total} IndustryOR IR-pipeline cases failed:\n" + "\n".join(failures)


@pytest.mark.llm
@pytest.mark.industryOR
def test_industryOR_direct_validated(llm_fixture, save_dir, difficulty, limit, solver):
    """Run IndustryOR cases through direct code gen + solution validation pipeline.

    Pipeline: ingest → param_computation → direct code gen → solve → solution validator.

    Pass --difficulty Easy|Medium|Hard to select the difficulty tier (default: Easy).
    Pass --limit N to cap the number of cases.
    """
    pytest.importorskip("datasets", reason="pip install 'orpilot[hf]' to run IndustryOR tests")

    from orpilot.benchmark.loader_hf import load_hf_cases

    cases = load_hf_cases("CardinalOperations/IndustryOR", difficulty=difficulty, limit=limit)
    assert cases, f"No {difficulty} cases loaded from CardinalOperations/IndustryOR"

    runner = BenchmarkRunner(timeout=180)
    passed = 0
    failures: list[str] = []
    solve_times: list[float] = []
    results: list = []

    log.info("Running %d IndustryOR %s cases (mode=direct_validated)", len(cases), difficulty)
    suite_start = time.monotonic()
    for i, case in enumerate(cases, 1):
        log.info("[%d/%d] %s (expected %s) ...", i, len(cases), case.name, case.expected_objective)

        if hasattr(llm_fixture, "reset_usage"):
            llm_fixture.reset_usage()

        result = runner.run_direct_pipeline_validated(case, llm_fixture, solver=solver)

        if hasattr(llm_fixture, "get_usage"):
            usage = llm_fixture.get_usage()
            result.metrics = {
                "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
                "latency_s": round(result.solve_time or 0.0, 2),
            }

        results.append(result)
        solve_times.append(result.solve_time)
        if save_dir:
            _save_artifacts(result, case, save_dir)
        if result.passed:
            passed += 1
            log.info("  PASS  (obj=%s, time=%.1fs)", result.objective_value, result.solve_time)
        else:
            msg = (
                f"status={result.status}, "
                f"obj={result.objective_value} (expected {case.expected_objective}), "
                f"error={result.error}"
            )
            log.info("  FAIL  (%s)", msg)
            failures.append(f"{case.name}: {msg}")
        print(f"[{i}/{len(cases)}] {'PASS' if result.passed else 'FAIL'}  passed={passed}  failed={len(failures)}  ({result.solve_time:.1f}s)", flush=True)

    total = len(cases)
    pct = 100.0 * passed / total if total else 0.0
    total_wall = time.monotonic() - suite_start
    avg_t = sum(solve_times) / len(solve_times) if solve_times else 0.0
    min_t = min(solve_times) if solve_times else 0.0
    max_t = max(solve_times) if solve_times else 0.0
    log.info("=" * 60)
    log.info("  RESULT: %d/%d passed (%.1f%%)", passed, total, pct)
    log.info("  TIME:   total=%.1fs  avg=%.1fs  min=%.1fs  max=%.1fs", total_wall, avg_t, min_t, max_t)
    if failures:
        log.info("  FAILED cases:")
        for f in failures:
            log.info("    - %s", f)
    log.info("=" * 60)

    if save_dir and results:
        total_input = sum(r.metrics.get("input_tokens", 0) for r in results)
        total_output = sum(r.metrics.get("output_tokens", 0) for r in results)
        aggregate = {
            "run_id": datetime.datetime.now().isoformat(timespec="seconds"),
            "solver": solver,
            "difficulty": difficulty,
            "mode": "direct_pipeline_validated",
            "total_cases": total,
            "passed": passed,
            "failed": total - passed,
            "totals": {
                "input_tokens": total_input,
                "output_tokens": total_output,
                "latency_s": round(total_wall, 2),
            },
            "cases": {
                r.case_name: {
                    "status": r.status,
                    "passed": r.passed,
                    "input_tokens": r.metrics.get("input_tokens", 0),
                    "output_tokens": r.metrics.get("output_tokens", 0),
                    "latency_s": r.metrics.get("latency_s", 0.0),
                    "expected_objective": r.expected_objective,
                    "validation_attempts": r.validation_attempts,
                    "first_validated_objective": r.first_validated_objective,
                    "code_gen_retries": r.code_gen_retries,
                    "validator_inconclusive": r.validator_inconclusive,
                }
                for r in results
            },
        }
        (save_dir / "metrics_direct_validated.json").write_text(
            json.dumps(aggregate, indent=2), encoding="utf-8"
        )
        log.info("  Wrote aggregate metrics to %s/metrics_direct_validated.json", save_dir)

    assert not failures, f"{len(failures)}/{total} IndustryOR direct-validated cases failed:\n" + "\n".join(failures)
